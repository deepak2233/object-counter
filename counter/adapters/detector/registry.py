"""Resolves model names to detectors, one catalog entry at a time."""

from __future__ import annotations

import logging
import threading

from counter.adapters.detector.artifacts import ArtifactStore
from counter.adapters.detector.catalog import ModelCatalog, ModelSpec
from counter.adapters.detector.fake import FakeObjectDetector
from counter.adapters.detector.labels import load_labels
from counter.adapters.detector.onnx_runtime import OnnxObjectDetector
from counter.adapters.detector.tensorflow_serving import TensorFlowServingDetector
from counter.adapters.detector.torchscript import TorchScriptObjectDetector
from counter.domain.errors import ModelNotFoundError
from counter.domain.models import ModelInfo
from counter.domain.ports import ObjectDetector, ObjectDetectorRegistry

logger = logging.getLogger(__name__)


class CatalogModelRegistry(ObjectDetectorRegistry):
    """Build and cache detectors from catalog entries."""

    def __init__(
        self,
        catalog: ModelCatalog,
        artifacts: ArtifactStore | None = None,
        *,
        tfs_base_url: str = "http://localhost:8501",
        tfs_timeout: float = 30.0,
        max_image_side: int = 1024,
    ) -> None:
        self._catalog = catalog
        self._artifacts = artifacts or ArtifactStore()
        self._tfs_base_url = tfs_base_url
        self._tfs_timeout = tfs_timeout
        self._max_image_side = max_image_side
        self._detectors: dict[str, ObjectDetector] = {}
        self._lock = threading.Lock()

    @property
    def default_model(self) -> str:
        return self._catalog.default_model

    def get(self, model_name: str | None = None) -> ObjectDetector:
        name = model_name or self._catalog.default_model

        cached = self._detectors.get(name)
        if cached is not None:
            return cached

        spec = self._catalog.spec_for(name)
        if spec is None:
            raise ModelNotFoundError(
                f"unknown model '{name}'; available: "
                f"{sorted(entry.name for entry in self._catalog.models)}"
            )

        with self._lock:
            # Another request may have populated the cache before the lock.
            cached = self._detectors.get(name)
            if cached is None:
                cached = self._build(spec)
                self._detectors[name] = cached
            return cached

    def available(self) -> list[ModelInfo]:
        """Catalog contents, without loading anything."""
        return [
            ModelInfo(
                name=spec.name,
                framework=spec.framework,
                version=spec.version,
                metadata={"loaded": str(spec.name in self._detectors).lower()},
            )
            for spec in self._catalog.models
        ]

    def warm(self) -> None:
        """Load every catalog model."""
        for spec in self._catalog.models:
            try:
                self.get(spec.name)
            except Exception:
                logger.exception("could not warm model", extra={"model": spec.name})

    def close(self) -> None:
        with self._lock:
            detectors = list(self._detectors.values())
            self._detectors.clear()
        for detector in detectors:
            detector.close()

    def _build(self, spec: ModelSpec) -> ObjectDetector:
        logger.info("building detector", extra={"model": spec.name, "framework": spec.framework})

        if spec.framework == "fake":
            return FakeObjectDetector(name=spec.name)

        if spec.framework == "tensorflow-serving":
            return TensorFlowServingDetector(
                base_url=spec.endpoint or self._tfs_base_url,
                model_name=spec.name,
                remote_name=spec.remote_name,
                labels=load_labels(spec.labels) if spec.labels else None,
                version=spec.version,
                timeout=self._tfs_timeout,
                max_image_side=self._max_image_side,
            )

        if spec.framework == "onnx":
            assert spec.artifact is not None  # guaranteed by ModelSpec validation
            return OnnxObjectDetector(
                model_path=self._artifacts.resolve(spec.artifact),
                labels=load_labels(spec.labels) if spec.labels else {},
                name=spec.name,
                version=spec.version,
                input_size=spec.input_size,
                score_floor=spec.score_floor,
                iou_threshold=spec.iou_threshold,
            )

        assert spec.artifact is not None  # guaranteed by ModelSpec validation
        return TorchScriptObjectDetector(
            model_path=self._artifacts.resolve(spec.artifact),
            labels=load_labels(spec.labels) if spec.labels else {},
            name=spec.name,
            version=spec.version,
            input_size=spec.input_size,
            input_contract=spec.input_contract,
            score_floor=spec.score_floor,
            iou_threshold=spec.iou_threshold,
            device=spec.device,
        )


class StaticModelRegistry(ObjectDetectorRegistry):
    """Registry over already-built detectors."""

    def __init__(
        self, detectors: dict[str, ObjectDetector], default_model: str | None = None
    ) -> None:
        if not detectors:
            raise ValueError("StaticModelRegistry needs at least one detector")
        self._detectors = detectors
        self._default = default_model or next(iter(detectors))

    @property
    def default_model(self) -> str:
        return self._default

    def get(self, model_name: str | None = None) -> ObjectDetector:
        name = model_name or self._default
        try:
            return self._detectors[name]
        except KeyError as exc:
            raise ModelNotFoundError(
                f"unknown model '{name}'; available: {sorted(self._detectors)}"
            ) from exc

    def available(self) -> list[ModelInfo]:
        return [detector.info for detector in self._detectors.values()]

    def close(self) -> None:
        for detector in self._detectors.values():
            detector.close()
