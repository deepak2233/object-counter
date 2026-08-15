"""ONNX Runtime adapter: in-process inference for exported detectors.

Second framework behind the same port, so the choice between "call TF Serving"
and "run the model in this process" is a catalog entry, not a code change.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from counter.adapters.detector.images import decode, to_uint8_array
from counter.adapters.detector.onnx_ops import decode_yolo_output, letterbox, to_nchw_float
from counter.domain.errors import DetectorUnavailableError, ModelLoadError
from counter.domain.models import Image, ModelInfo, Prediction
from counter.domain.ports import ObjectDetector

logger = logging.getLogger(__name__)

DEFAULT_INPUT_SIZE = 640
DEFAULT_SCORE_FLOOR = 0.05
DEFAULT_IOU_THRESHOLD = 0.45


class InferenceSession(Protocol):
    """The slice of onnxruntime.InferenceSession this adapter needs.

    Typing against a Protocol rather than the concrete class means the adapter
    can be tested with a stub session: the ~200 MB onnxruntime wheel is not a
    prerequisite for running the test suite.
    """

    def run(
        self, output_names: Sequence[str] | None, input_feed: dict[str, Any]
    ) -> list[np.ndarray]: ...

    def get_inputs(self) -> list[Any]: ...


class OnnxObjectDetector(ObjectDetector):
    """Runs a YOLO-style ONNX detector locally."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels: Mapping[int, str] | None = None,
        *,
        name: str = "onnx",
        version: str = "1",
        input_size: int = DEFAULT_INPUT_SIZE,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        session: InferenceSession | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        self._name = name
        self._version = version
        self._labels = labels or {}
        self._input_size = input_size
        self._score_floor = score_floor
        self._iou_threshold = iou_threshold
        self._session = session or self._build_session(model_path, providers)
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _build_session(
        model_path: str | Path | None, providers: Sequence[str] | None
    ) -> InferenceSession:
        if model_path is None:
            raise ModelLoadError("onnx detector needs either a model_path or a session")

        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"onnx model not found: {path}")

        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise ModelLoadError(
                "onnxruntime is not installed; install the 'onnx' extra to serve ONNX models"
            ) from exc

        available = list(providers or onnxruntime.get_available_providers())
        logger.info("loading onnx model", extra={"path": str(path), "providers": available})
        try:
            return onnxruntime.InferenceSession(str(path), providers=available)
        except Exception as exc:  # pragma: no cover - onnxruntime raises bare Exception
            raise ModelLoadError(f"could not load onnx model {path}: {exc}") from exc

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name,
            framework="onnx",
            version=self._version,
            labels=len(self._labels),
            metadata={"input_size": str(self._input_size)},
        )

    def predict(self, image: Image) -> list[Prediction]:
        pixels = to_uint8_array(decode(image))
        letterboxed, transform = letterbox(pixels, self._input_size)
        model_input = to_nchw_float(letterboxed)

        try:
            outputs = self._session.run(None, {self._input_name: model_input})
        except Exception as exc:
            raise DetectorUnavailableError(
                f"onnx inference failed for {self._name}: {exc}"
            ) from exc

        if not outputs:
            raise DetectorUnavailableError(f"onnx model {self._name} returned no output tensor")

        try:
            return decode_yolo_output(
                outputs[0],
                transform,
                self._labels,
                score_threshold=self._score_floor,
                iou_threshold=self._iou_threshold,
            )
        except ValueError as exc:
            raise DetectorUnavailableError(
                f"onnx model {self._name} returned an unsupported output layout: {exc}"
            ) from exc
