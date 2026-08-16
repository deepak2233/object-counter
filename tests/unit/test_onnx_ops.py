"""Pre/post-processing maths. This is where detector bugs actually live."""

from __future__ import annotations

import numpy as np
import pytest

from counter.adapters.detector.onnx_ops import (
    decode_yolo_output,
    letterbox,
    non_max_suppression,
    to_nchw_float,
    xywh_to_xyxy,
)

pytestmark = pytest.mark.unit

LABELS = {0: "cat", 1: "dog"}


class TestLetterbox:
    def test_pads_to_a_square_without_distorting(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        padded, transform = letterbox(image, 640)

        assert padded.shape == (640, 640, 3)
        assert transform.scale == pytest.approx(3.2)
        assert (transform.pad_x, transform.pad_y) == (0.0, 160.0)

    def test_records_the_original_size(self) -> None:
        _, transform = letterbox(np.zeros((100, 200, 3), dtype=np.uint8), 640)

        assert (transform.original_width, transform.original_height) == (200, 100)

    def test_fills_the_padding_with_the_pad_value(self) -> None:
        padded, _ = letterbox(np.zeros((100, 200, 3), dtype=np.uint8), 640, pad_value=114)

        assert padded[0, 0, 0] == 114
        assert padded[320, 320, 0] == 0


class TestTensorLayout:
    def test_converts_to_normalised_nchw(self) -> None:
        tensor = to_nchw_float(np.full((4, 4, 3), 255, dtype=np.uint8))

        assert tensor.shape == (1, 3, 4, 4)
        assert tensor.dtype == np.float32
        assert tensor.max() == pytest.approx(1.0)

    def test_xywh_to_xyxy(self) -> None:
        converted = xywh_to_xyxy(np.array([[10.0, 20.0, 4.0, 6.0]]))

        assert converted.tolist() == [[8.0, 17.0, 12.0, 23.0]]


class TestNonMaxSuppression:
    def test_keeps_the_highest_scoring_of_two_overlapping_boxes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        scores = np.array([0.6, 0.9])

        assert non_max_suppression(boxes, scores, iou_threshold=0.5) == [1]

    def test_keeps_boxes_that_do_not_overlap(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
        scores = np.array([0.6, 0.9])

        assert sorted(non_max_suppression(boxes, scores, iou_threshold=0.5)) == [0, 1]

    def test_keeps_overlapping_boxes_from_different_classes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        scores = np.array([0.9, 0.8])
        classes = np.array([0, 1])

        assert non_max_suppression(boxes, scores, 0.5, classes) == [0, 1]

    def test_handles_an_empty_input(self) -> None:
        assert non_max_suppression(np.zeros((0, 4)), np.zeros(0), 0.5) == []

    def test_handles_zero_area_boxes_without_dividing_by_zero(self) -> None:
        boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], dtype=float)

        assert non_max_suppression(boxes, np.array([0.9, 0.8]), 0.5) == [0, 1]


class TestDecodeYoloOutput:
    @staticmethod
    def raw_output(rows: list[list[float]]) -> np.ndarray:
        """(1, 4 + num_classes, num_boxes), the YOLOv8 export layout."""
        return np.array([np.array(rows, dtype=np.float32).T])

    def test_maps_boxes_back_to_the_original_image(self) -> None:
        _, transform = letterbox(np.zeros((100, 200, 3), dtype=np.uint8), 640)
        # A box covering the whole original image inside the letterboxed frame.
        raw = self.raw_output([[320.0, 320.0, 640.0, 320.0, 0.9, 0.1]])

        predictions = decode_yolo_output(raw, transform, LABELS)

        assert len(predictions) == 1
        box = predictions[0].box
        assert (box.xmin, box.ymin, box.xmax, box.ymax) == pytest.approx((0.0, 0.0, 1.0, 1.0))

    def test_picks_the_highest_scoring_class(self) -> None:
        _, transform = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)
        raw = self.raw_output([[100.0, 100.0, 20.0, 20.0, 0.2, 0.8]])

        predictions = decode_yolo_output(raw, transform, LABELS)

        assert (predictions[0].class_name, predictions[0].score) == ("dog", pytest.approx(0.8))

    def test_accepts_the_transposed_layout(self) -> None:
        _, transform = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)
        raw = np.array([[[100.0, 100.0, 20.0, 20.0, 0.9, 0.1]] * 10], dtype=np.float32)

        assert decode_yolo_output(raw, transform, LABELS)

    def test_suppresses_duplicate_detections_of_the_same_object(self) -> None:
        _, transform = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)
        raw = self.raw_output(
            [
                [100.0, 100.0, 40.0, 40.0, 0.9, 0.0],
                [102.0, 102.0, 40.0, 40.0, 0.7, 0.0],
            ]
        )

        assert len(decode_yolo_output(raw, transform, LABELS)) == 1

    def test_drops_boxes_below_the_backend_floor(self) -> None:
        _, transform = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)
        raw = self.raw_output([[100.0, 100.0, 20.0, 20.0, 0.01, 0.0]])

        assert decode_yolo_output(raw, transform, LABELS, score_threshold=0.05) == []

    def test_clips_boxes_that_extend_past_the_image(self) -> None:
        _, transform = letterbox(np.zeros((100, 200, 3), dtype=np.uint8), 640)
        raw = self.raw_output([[320.0, 320.0, 5000.0, 5000.0, 0.9, 0.0]])

        box = decode_yolo_output(raw, transform, LABELS)[0].box
        assert (box.xmin, box.ymin, box.xmax, box.ymax) == (0.0, 0.0, 1.0, 1.0)

    def test_rejects_an_output_it_does_not_understand(self) -> None:
        _, transform = letterbox(np.zeros((64, 64, 3), dtype=np.uint8), 640)

        with pytest.raises(ValueError, match="unexpected detector output shape"):
            decode_yolo_output(np.zeros((2, 3, 4, 5)), transform, LABELS)

    @pytest.mark.parametrize(
        "raw",
        [
            [[100.0, 100.0, -20.0, 20.0, 0.9, 0.1]],
            [[100.0, 100.0, 20.0, 20.0, 1.1, 0.1]],
            [[100.0, 100.0, 20.0, 20.0, float("nan"), 0.1]],
        ],
    )
    def test_rejects_invalid_detector_values(self, raw: list[list[float]]) -> None:
        _, transform = letterbox(np.zeros((640, 640, 3), dtype=np.uint8), 640)

        with pytest.raises(ValueError):
            decode_yolo_output(self.raw_output(raw), transform, LABELS)
