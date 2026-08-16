"""YOLO-style preprocessing and postprocessing."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from counter.adapters.detector.labels import class_name_for
from counter.domain.models import Box, Prediction


class LetterboxTransform:
    """Image scale and padding used for coordinate restoration."""

    __slots__ = ("original_height", "original_width", "pad_x", "pad_y", "scale")

    def __init__(
        self, scale: float, pad_x: float, pad_y: float, original_width: int, original_height: int
    ) -> None:
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.original_width = original_width
        self.original_height = original_height


def letterbox(
    image: np.ndarray, size: int, pad_value: int = 114
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to `size` x `size`."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))

    resized = _resize_nearest(image, new_width, new_height)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - new_width) / 2
    pad_y = (size - new_height) / 2
    top, left = int(pad_y), int(pad_x)
    canvas[top : top + new_height, left : left + new_width] = resized

    return canvas, LetterboxTransform(scale, pad_x, pad_y, width, height)


def to_nchw_float(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (1, 3, H, W) float32 scaled to 0..1."""
    normalised = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(normalised.transpose(2, 0, 1)[np.newaxis, ...])


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Centre-form boxes to corner-form. Operates on the last axis."""
    centre_x, centre_y, width, height = (boxes[..., i] for i in range(4))
    return np.stack(
        [
            centre_x - width / 2,
            centre_y - height / 2,
            centre_x + width / 2,
            centre_y + height / 2,
        ],
        axis=-1,
    )


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    class_ids: np.ndarray | None = None,
) -> list[int]:
    """Greedy class-aware NMS over corner-form boxes."""
    if boxes.size == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    kept: list[int] = []
    while order.size > 0:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        intersection = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)

        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        same_class = (
            np.ones(rest.shape, dtype=bool)
            if class_ids is None
            else class_ids[rest] == class_ids[current]
        )
        order = rest[(~same_class) | (iou <= iou_threshold)]

    return kept


def decode_yolo_output(
    raw_output: np.ndarray,
    transform: LetterboxTransform,
    labels: Mapping[int, str],
    *,
    score_threshold: float = 0.05,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> list[Prediction]:
    """Decode either common YOLOv8 output orientation."""
    try:
        predictions_tensor = np.asarray(raw_output, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector output is not numeric") from exc
    if predictions_tensor.ndim == 3:
        predictions_tensor = predictions_tensor[0]
    if predictions_tensor.ndim != 2:
        raise ValueError(f"unexpected detector output shape {np.asarray(raw_output).shape}")

    if _is_channels_first(predictions_tensor, len(labels)):
        predictions_tensor = predictions_tensor.T

    boxes_xywh = predictions_tensor[:, :4]
    class_scores = predictions_tensor[:, 4:]
    if class_scores.size == 0:
        return []
    if not np.isfinite(predictions_tensor).all():
        raise ValueError("detector output contains a non-finite value")
    if np.any(boxes_xywh[:, 2:] < 0):
        raise ValueError("detector output contains a negative box size")
    if np.any((class_scores < 0) | (class_scores > 1)):
        raise ValueError("detector output contains a score outside 0..1")

    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    above_floor = scores >= score_threshold
    boxes_xywh, class_ids, scores = (
        boxes_xywh[above_floor],
        class_ids[above_floor],
        scores[above_floor],
    )
    if scores.size == 0:
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    kept = non_max_suppression(boxes_xyxy, scores, iou_threshold, class_ids)[:max_detections]

    return [
        Prediction(
            class_name=class_name_for(labels, int(class_ids[index])),
            score=float(scores[index]),
            box=_to_normalised_box(boxes_xyxy[index], transform),
        )
        for index in kept
    ]


def _is_channels_first(tensor: np.ndarray, num_classes: int) -> bool:
    """Return whether the tensor uses channels-first output."""
    rows, columns = tensor.shape
    if num_classes:
        expected = 4 + num_classes
        if columns == expected:
            return False
        if rows == expected:
            return True
    return rows < columns


def _to_normalised_box(box_xyxy: np.ndarray, transform: LetterboxTransform) -> Box:
    """Undo the letterbox, then normalise against the original image size."""
    xmin, ymin, xmax, ymax = (float(value) for value in box_xyxy[:4])
    scale = transform.scale or 1.0

    xmin = (xmin - transform.pad_x) / scale / transform.original_width
    xmax = (xmax - transform.pad_x) / scale / transform.original_width
    ymin = (ymin - transform.pad_y) / scale / transform.original_height
    ymax = (ymax - transform.pad_y) / scale / transform.original_height

    return Box(
        xmin=_clip_unit(xmin),
        ymin=_clip_unit(ymin),
        xmax=_clip_unit(xmax),
        ymax=_clip_unit(ymax),
    )


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize with numpy only.

    Bilinear would be marginally more accurate, but pulling in a resize
    dependency for the pre-processing path is not worth it; adapters that need
    exact parity with a training pipeline should override this with the same
    resize the training code used.
    """
    source_height, source_width = image.shape[:2]
    row_index = (np.arange(height) * (source_height / height)).astype(np.int64)
    column_index = (np.arange(width) * (source_width / width)).astype(np.int64)
    return image[row_index[:, None], column_index[None, :]]
