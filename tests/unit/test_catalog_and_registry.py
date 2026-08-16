from __future__ import annotations

import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from counter.adapters.detector.artifacts import ArtifactStore
from counter.adapters.detector.catalog import ArtifactSpec, ModelCatalog
from counter.adapters.detector.fake import FakeObjectDetector
from counter.adapters.detector.registry import CatalogModelRegistry, StaticModelRegistry
from counter.domain.errors import ModelLoadError, ModelNotFoundError

pytestmark = pytest.mark.unit

CATALOG_YAML = """
default_model: fake
models:
  - name: fake
    framework: fake
  - name: rfcn
    framework: tensorflow-serving
    labels: mscoco_label_map.json
    endpoint: ${TEST_TFS_ENDPOINT}
"""


def write_catalog(tmp_path: Path, content: str, name: str = "models.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def tfs_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_TFS_ENDPOINT", "http://tfs.internal:8501")


class TestModelCatalog:
    def test_loads_models_from_yaml(self, tmp_path: Path) -> None:
        catalog = ModelCatalog.from_file(write_catalog(tmp_path, CATALOG_YAML))

        assert catalog.default_model == "fake"
        assert [spec.name for spec in catalog.models] == ["fake", "rfcn"]

    def test_expands_environment_variables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_TFS_ENDPOINT", "http://tfs.internal:8501")

        catalog = ModelCatalog.from_file(write_catalog(tmp_path, CATALOG_YAML))

        assert catalog.spec_for("rfcn").endpoint == "http://tfs.internal:8501"  # type: ignore[union-attr]

    def test_rejects_an_unresolved_environment_reference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MISSING_MODEL_HOST", raising=False)
        content = """
default_model: y
models:
  - name: y
    framework: tensorflow-serving
    endpoint: ${MISSING_MODEL_HOST}
"""

        with pytest.raises(ModelLoadError, match="unresolved environment reference"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_rejects_a_default_that_is_not_in_the_catalog(self, tmp_path: Path) -> None:
        content = "default_model: missing\nmodels:\n  - name: fake\n    framework: fake\n"

        with pytest.raises(ModelLoadError, match="default_model"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_rejects_duplicate_model_names(self, tmp_path: Path) -> None:
        content = (
            "default_model: fake\n"
            "models:\n  - name: fake\n    framework: fake\n  - name: fake\n    framework: fake\n"
        )

        with pytest.raises(ModelLoadError, match="duplicate model names"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_rejects_an_unknown_framework(self, tmp_path: Path) -> None:
        content = "default_model: x\nmodels:\n  - name: x\n    framework: caffe\n"

        with pytest.raises(ModelLoadError, match="invalid"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_requires_an_artifact_for_in_process_frameworks(self, tmp_path: Path) -> None:
        content = "default_model: y\nmodels:\n  - name: y\n    framework: onnx\n"

        with pytest.raises(ModelLoadError, match="needs an artifact"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_requires_s3_artifacts_to_be_pinned(self, tmp_path: Path) -> None:
        content = """
default_model: y
models:
  - name: y
    framework: onnx
    artifact:
      uri: s3://models/y.onnx
"""

        with pytest.raises(ModelLoadError, match="must pin"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_reads_the_torchvision_input_contract(self, tmp_path: Path) -> None:
        content = """
default_model: y
models:
  - name: y
    framework: torchscript
    input_contract: torchvision
    artifact:
      uri: /models/y.pt
"""

        catalog = ModelCatalog.from_file(write_catalog(tmp_path, content))

        assert catalog.models[0].input_contract == "torchvision"

    def test_requires_a_numeric_tensorflow_serving_version(self, tmp_path: Path) -> None:
        content = """
default_model: y
models:
  - name: y
    framework: tensorflow-serving
    version: latest
"""

        with pytest.raises(ModelLoadError, match="numeric TensorFlow Serving version"):
            ModelCatalog.from_file(write_catalog(tmp_path, content))

    def test_reports_a_missing_catalog_file(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="catalog not found"):
            ModelCatalog.from_file(tmp_path / "nope.yaml")

    def test_packaged_catalogs_are_valid(self) -> None:
        from counter.config import PACKAGED_CATALOGS, packaged_catalog_path

        for filename in set(PACKAGED_CATALOGS.values()):
            assert ModelCatalog.from_file(packaged_catalog_path(filename)).models


class TestArtifactStore:
    def test_resolves_a_local_file(self, tmp_path: Path) -> None:
        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"weights")

        resolved = ArtifactStore(tmp_path / "cache").resolve(ArtifactSpec(uri=str(artifact)))

        assert resolved == artifact

    def test_verifies_the_digest(self, tmp_path: Path) -> None:
        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"weights")
        digest = hashlib.sha256(b"weights").hexdigest()

        store = ArtifactStore(tmp_path / "cache")

        assert store.resolve(ArtifactSpec(uri=str(artifact), sha256=digest)) == artifact

    def test_refuses_an_artifact_whose_digest_does_not_match(self, tmp_path: Path) -> None:
        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"tampered")

        with pytest.raises(ModelLoadError, match="checksum mismatch"):
            ArtifactStore(tmp_path).resolve(ArtifactSpec(uri=str(artifact), sha256="0" * 64))

    def test_reports_a_missing_artifact(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="artifact not found"):
            ArtifactStore(tmp_path).resolve(ArtifactSpec(uri=str(tmp_path / "absent.onnx")))

    def test_rejects_an_unsupported_scheme(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="unsupported artifact scheme"):
            ArtifactStore(tmp_path).resolve(ArtifactSpec(uri="ftp://models/model.onnx"))

    def test_downloads_and_reuses_a_verified_s3_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        class S3Client:
            def download_file(self, bucket: str, key: str, filename: str) -> None:
                calls.append(f"{bucket}/{key}")
                Path(filename).write_bytes(b"weights")

        monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda _: S3Client()))
        spec = ArtifactSpec(
            uri="s3://models/model.onnx",
            sha256=hashlib.sha256(b"weights").hexdigest(),
        )
        store = ArtifactStore(tmp_path / "cache")

        first = store.resolve(spec)
        second = store.resolve(spec)

        assert first == second
        assert first.read_bytes() == b"weights"
        assert calls == ["models/model.onnx"]

    def test_parallel_s3_downloads_use_distinct_staging_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        barrier = threading.Barrier(2)
        staging_paths: list[str] = []

        class S3Client:
            def download_file(self, bucket: str, key: str, filename: str) -> None:
                staging_paths.append(filename)
                barrier.wait()
                Path(filename).write_bytes(b"weights")

        monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda _: S3Client()))
        spec = ArtifactSpec(
            uri="s3://models/model.onnx",
            sha256=hashlib.sha256(b"weights").hexdigest(),
        )
        store = ArtifactStore(tmp_path / "cache")

        with ThreadPoolExecutor(max_workers=2) as pool:
            resolved = list(pool.map(lambda _: store.resolve(spec), range(2)))

        assert resolved[0] == resolved[1]
        assert resolved[0].read_bytes() == b"weights"
        assert len(set(staging_paths)) == 2
        assert not list((tmp_path / "cache").rglob("*.part"))


class TestCatalogModelRegistry:
    @pytest.fixture
    def registry(self, tmp_path: Path) -> CatalogModelRegistry:
        catalog = ModelCatalog.from_file(write_catalog(tmp_path, CATALOG_YAML))
        return CatalogModelRegistry(catalog, ArtifactStore(tmp_path / "cache"))

    def test_returns_the_default_model_when_none_is_asked_for(
        self, registry: CatalogModelRegistry
    ) -> None:
        assert registry.get().info.name == "fake"

    def test_builds_the_backend_the_catalog_declares(self, registry: CatalogModelRegistry) -> None:
        assert registry.get("rfcn").info.framework == "tensorflow-serving"

    def test_caches_detectors(self, registry: CatalogModelRegistry) -> None:
        # Rebuilding a session per request would dominate the latency budget.
        assert registry.get("fake") is registry.get("fake")

    def test_unknown_models_are_reported_with_the_available_names(
        self, registry: CatalogModelRegistry
    ) -> None:
        with pytest.raises(ModelNotFoundError, match="unknown model 'yolo'"):
            registry.get("yolo")

    def test_lists_models_without_loading_them(self, registry: CatalogModelRegistry) -> None:
        listed = registry.available()

        assert {info.name for info in listed} == {"fake", "rfcn"}
        assert all(info.metadata["loaded"] == "false" for info in listed)

    def test_marks_loaded_models(self, registry: CatalogModelRegistry) -> None:
        registry.get("fake")

        loaded = {info.name: info.metadata["loaded"] for info in registry.available()}
        assert loaded == {"fake": "true", "rfcn": "false"}


class TestStaticModelRegistry:
    def test_resolves_by_name(self) -> None:
        registry = StaticModelRegistry({"a": FakeObjectDetector(name="a")})

        assert registry.get("a").info.name == "a"

    def test_defaults_to_the_first_detector(self) -> None:
        registry = StaticModelRegistry({"a": FakeObjectDetector(name="a")})

        assert registry.get().info.name == "a"

    def test_rejects_an_empty_registry(self) -> None:
        with pytest.raises(ValueError, match="at least one detector"):
            StaticModelRegistry({})
