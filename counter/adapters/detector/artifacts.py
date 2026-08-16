"""Fetch, verify, and cache model artifacts."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from urllib.parse import urlparse

from counter.adapters.detector.catalog import ArtifactSpec
from counter.domain.errors import ModelLoadError

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024


class ArtifactStore:
    """Resolve local and S3 artifact URIs to verified files."""

    def __init__(self, cache_dir: str | Path = "var/models") -> None:
        self._cache_dir = Path(cache_dir)

    def resolve(self, spec: ArtifactSpec) -> Path:
        parsed = urlparse(spec.uri)
        scheme = parsed.scheme or "file"

        if scheme == "file":
            path = Path(parsed.path if parsed.scheme else spec.uri).expanduser()
            if not path.is_file():
                raise ModelLoadError(f"model artifact not found: {path}")
            self._verify(path, spec.sha256)
            return path

        if scheme == "s3":
            return self._fetch_s3(parsed.netloc, parsed.path.lstrip("/"), spec)

        raise ModelLoadError(f"unsupported artifact scheme '{scheme}' in {spec.uri}")

    def _fetch_s3(self, bucket: str, key: str, spec: ArtifactSpec) -> Path:
        if not bucket or not key:
            raise ModelLoadError(f"invalid S3 artifact URI: {spec.uri}")
        target = self._cache_path(spec, Path(key).name)
        if target.is_file():
            self._verify(target, spec.sha256)
            logger.info("model artifact served from cache", extra={"path": str(target)})
            return target

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise ModelLoadError(
                "boto3 is not installed; install the 's3' extra to fetch models from S3"
            ) from exc

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelLoadError(f"could not create model cache {target.parent}: {exc}") from exc
        staging = target.with_suffix(f"{target.suffix}.{uuid.uuid4().hex}.part")
        logger.info("downloading model artifact", extra={"bucket": bucket, "key": key})
        try:
            boto3.client("s3").download_file(bucket, key, str(staging))
        except Exception as exc:  # pragma: no cover - botocore raises bare Exception
            staging.unlink(missing_ok=True)
            raise ModelLoadError(f"could not download s3://{bucket}/{key}: {exc}") from exc

        try:
            self._verify(staging, spec.sha256)
            staging.replace(target)
        except ModelLoadError:
            staging.unlink(missing_ok=True)
            raise
        except OSError as exc:
            staging.unlink(missing_ok=True)
            raise ModelLoadError(f"could not cache model artifact at {target}: {exc}") from exc
        return target

    def _cache_path(self, spec: ArtifactSpec, filename: str) -> Path:
        prefix = (
            spec.sha256[:16] if spec.sha256 else hashlib.sha256(spec.uri.encode()).hexdigest()[:16]
        )
        return self._cache_dir / prefix / filename

    @staticmethod
    def _verify(path: Path, expected_sha256: str | None) -> None:
        if not expected_sha256:
            logger.warning("model artifact is not pinned to a digest", extra={"path": str(path)})
            return

        try:
            actual = sha256_of(path)
        except OSError as exc:
            raise ModelLoadError(f"could not read model artifact {path}: {exc}") from exc
        if actual.lower() != expected_sha256.lower():
            raise ModelLoadError(
                f"checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
            )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
