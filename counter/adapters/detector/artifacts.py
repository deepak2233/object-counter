"""Fetching and verifying model weights.

Internally trained models cannot be `wget`-ed from a public bucket in a README,
which is what the original setup did. They come from a private store, they are
pinned by digest, and they are cached on disk so a pod restart does not re-pull
gigabytes. See docs/MULTI_MODEL.md.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

from counter.adapters.detector.catalog import ArtifactSpec
from counter.domain.errors import ModelLoadError

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1024 * 1024


class ArtifactStore:
    """Resolves an artifact URI to a local file, verifying its digest.

    Supported schemes:
      * `file://` or a bare path — local or a mounted volume (PVC, NFS);
      * `s3://bucket/key` — any S3-compatible store (AWS, MinIO, Ceph), using
        boto3 credentials from the environment or the instance role.

    A new scheme is a new `_fetch_*` method; nothing else changes.
    """

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

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".part")
        logger.info("downloading model artifact", extra={"bucket": bucket, "key": key})
        try:
            boto3.client("s3").download_file(bucket, key, str(staging))
        except Exception as exc:  # pragma: no cover - botocore raises bare Exception
            staging.unlink(missing_ok=True)
            raise ModelLoadError(f"could not download s3://{bucket}/{key}: {exc}") from exc

        self._verify(staging, spec.sha256)
        # Rename last: a crashed download must never be picked up as a valid
        # cache entry by the next process to start.
        shutil.move(str(staging), str(target))
        return target

    def _cache_path(self, spec: ArtifactSpec, filename: str) -> Path:
        # Digest-addressed when pinned, so two versions never collide in cache.
        prefix = spec.sha256[:16] if spec.sha256 else "unpinned"
        return self._cache_dir / prefix / filename

    @staticmethod
    def _verify(path: Path, expected_sha256: str | None) -> None:
        if not expected_sha256:
            logger.warning("model artifact is not pinned to a digest", extra={"path": str(path)})
            return

        actual = sha256_of(path)
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
