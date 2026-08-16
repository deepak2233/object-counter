"""TorchScript detector adapter."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from counter.adapters.detector.images import decode, to_uint8_array
from counter.adapters.detector.labels import class_name_for
from counter.adapters.detector.onnx_ops import (
    LetterboxTransform,
    decode_yolo_output,
    letterbox,
    to_nchw_float,
)
from counter.domain.errors import DetectorUnavailableError, ModelLoadError
from counter.domain.models import Box, Image, ModelInfo, Prediction
from counter.domain.ports import ObjectDetector

logger = logging.getLogger(__name__)

DEFAULT_INPUT_SIZE = 640


class TorchScriptObjectDetector(ObjectDetector):
    def __init__(
        self,
        model_path: str | Path | None = None,
        labels: Mapping[int, str] | None = None,
        *,
        name: str = "torchscript",
        version: str = "1",
        input_size: int = DEFAULT_INPUT_SIZE,
        input_contract: str = "yolov8",
        score_floor: float = 0.05,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        module: Any | None = None,
    ) -> None:
        self._name = name
        self._version = version
        self._labels = labels or {}
        self._input_size = input_size
        self._input_contract = input_contract
        self._score_floor = score_floor
        self._iou_threshold = iou_threshold
        self._device = device
        self._module = module if module is not None else self._load_module(model_path, device)

    @staticmethod
    def _load_module(model_path: str | Path | None, device: str) -> Any:
        if model_path is None:
            raise ModelLoadError("torchscript detector needs either a model_path or a module")

        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"torchscript model not found: {path}")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise ModelLoadError(
                "torch is not installed; install the 'torch' extra to serve TorchScript models"
            ) from exc

        logger.info("loading torchscript model", extra={"path": str(path), "device": device})
        try:
            module = torch.jit.load(str(path), map_location=device)
            module.eval()
        except Exception as exc:  # pragma: no cover - torch raises bare Exception
            raise ModelLoadError(f"could not load torchscript model {path}: {exc}") from exc
        return module

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name,
            framework="torchscript",
            version=self._version,
            labels=len(self._labels),
            metadata={"device": self._device, "input_size": str(self._input_size)},
        )

    def predict(self, image: Image) -> list[Prediction]:
        pixels = to_uint8_array(decode(image))
        if self._input_contract == "torchvision":
            height, width = pixels.shape[:2]
            transform = LetterboxTransform(1.0, 0.0, 0.0, width, height)
            model_input: Any = [self._to_tensor(_to_chw_float(pixels))]
        else:
            letterboxed, transform = letterbox(pixels, self._input_size)
            model_input = self._to_tensor(to_nchw_float(letterboxed))

        try:
            output = self._run_module(model_input)
        except Exception as exc:
            raise DetectorUnavailableError(
                f"torchscript inference failed for {self._name}: {exc}"
            ) from exc

        try:
            return self._to_predictions(output, transform)
        except DetectorUnavailableError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DetectorUnavailableError(
                f"torchscript model {self._name} returned invalid detections: {exc}"
            ) from exc

    def _run_module(self, model_input: Any) -> Any:
        try:
            import torch
        except ImportError:
            return self._module(model_input)
        with torch.inference_mode():
            return self._module(model_input)

    def _to_tensor(self, array: np.ndarray) -> Any:
        try:
            import torch
        except ImportError:  # pragma: no cover - only when a module was injected
            return array
        return torch.from_numpy(array).to(self._device)

    def _to_predictions(self, output: Any, transform: LetterboxTransform) -> list[Prediction]:
        detections = output[0] if isinstance(output, (list, tuple)) and output else output

        if isinstance(detections, Mapping):
            return self._from_torchvision_dict(detections, transform)

        try:
            return decode_yolo_output(
                _as_numpy(detections),
                transform,
                self._labels,
                score_threshold=self._score_floor,
                iou_threshold=self._iou_threshold,
            )
        except ValueError as exc:
            raise DetectorUnavailableError(
                f"torchscript model {self._name} returned an unsupported output layout: {exc}"
            ) from exc

    def _from_torchvision_dict(
        self, detections: Mapping[str, Any], transform: LetterboxTransform
    ) -> list[Prediction]:
        boxes = _as_numpy(detections["boxes"]).reshape(-1, 4)
        scores = _as_numpy(detections["scores"]).reshape(-1)
        labels = _as_numpy(detections["labels"]).reshape(-1)

        predictions: list[Prediction] = []
        for index in range(min(len(boxes), len(scores), len(labels))):
            xmin, ymin, xmax, ymax = (float(value) for value in boxes[index])
            score = float(scores[index])
            class_id = float(labels[index])
            if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax, score, class_id)):
                raise ValueError("detection contains a non-finite value")
            if not 0.0 <= score <= 1.0:
                raise ValueError("detection score is outside 0..1")
            if not class_id.is_integer():
                raise ValueError("class id is not an integer")
            if xmin > xmax or ymin > ymax:
                raise ValueError("detection box is inverted")
            predictions.append(
                Prediction(
                    class_name=class_name_for(self._labels, int(class_id)),
                    score=score,
                    box=_normalise(xmin, ymin, xmax, ymax, transform),
                )
            )
        return predictions


def _normalise(
    xmin: float, ymin: float, xmax: float, ymax: float, transform: LetterboxTransform
) -> Box:
    scale = transform.scale or 1.0
    return Box(
        xmin=_clip((xmin - transform.pad_x) / scale / transform.original_width),
        ymin=_clip((ymin - transform.pad_y) / scale / transform.original_height),
        xmax=_clip((xmax - transform.pad_x) / scale / transform.original_width),
        ymax=_clip((ymax - transform.pad_y) / scale / transform.original_height),
    )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _as_numpy(value: Any) -> np.ndarray:
    """Accept a torch tensor without importing torch."""
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu().numpy()
    return np.asarray(value)


def _to_chw_float(image: np.ndarray) -> np.ndarray:
    normalised = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))
