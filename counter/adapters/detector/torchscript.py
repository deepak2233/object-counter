"""TorchScript adapter: third framework, same port.

Handles the two output conventions worth supporting:
  * torchvision detection models -> [{"boxes": [[x1,y1,x2,y2], ...], "labels":
    [...], "scores": [...]}], boxes in input-pixel coordinates;
  * exported single-stage detectors -> one raw tensor, decoded like the ONNX path.
"""

from __future__ import annotations

import logging
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
        device: str = "cpu",
        module: Any | None = None,
    ) -> None:
        self._name = name
        self._version = version
        self._labels = labels or {}
        self._input_size = input_size
        self._device = device
        self._module = module or self._load_module(model_path, device)

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
        letterboxed, transform = letterbox(pixels, self._input_size)
        model_input = to_nchw_float(letterboxed)

        try:
            output = self._module(self._to_tensor(model_input))
        except Exception as exc:
            raise DetectorUnavailableError(
                f"torchscript inference failed for {self._name}: {exc}"
            ) from exc

        return self._to_predictions(output, transform)

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
            return decode_yolo_output(_as_numpy(detections), transform, self._labels)
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
        for index in range(len(scores)):
            xmin, ymin, xmax, ymax = (float(value) for value in boxes[index])
            predictions.append(
                Prediction(
                    class_name=class_name_for(self._labels, int(labels[index])),
                    score=float(scores[index]),
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
