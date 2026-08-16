"""ONNX and TorchScript adapters, driven through injected sessions/modules.

Both wheels are hundreds of megabytes and neither is needed to prove that the
adapter feeds the model the right tensor and reads its output correctly. The
Protocol seam is what makes that possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from counter.adapters.detector.onnx_runtime import OnnxObjectDetector
from counter.adapters.detector.torchscript import TorchScriptObjectDetector
from counter.domain.errors import DetectorUnavailableError, ModelLoadError
from counter.domain.models import Image

pytestmark = pytest.mark.unit

LABELS = {0: "cat", 1: "dog"}


class StubInput:
    name = "images"


class StubSession:
    """Stands in for onnxruntime.InferenceSession."""

    def __init__(self, output: np.ndarray | None = None, raises: Exception | None = None) -> None:
        self._output = output
        self._raises = raises
        self.last_input: dict[str, Any] | None = None

    def get_inputs(self) -> list[StubInput]:
        return [StubInput()]

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> list[np.ndarray]:
        if self._raises is not None:
            raise self._raises
        self.last_input = input_feed
        assert self._output is not None
        return [self._output]


def yolo_output(rows: list[list[float]]) -> np.ndarray:
    return np.array([np.array(rows, dtype=np.float32).T])


class TestOnnxObjectDetector:
    def test_feeds_a_normalised_nchw_tensor_named_after_the_model_input(
        self, image_bytes: bytes
    ) -> None:
        session = StubSession(yolo_output([[320.0, 320.0, 64.0, 64.0, 0.9, 0.1]]))
        detector = OnnxObjectDetector(labels=LABELS, session=session, input_size=640)

        detector.predict(Image(content=image_bytes))

        assert session.last_input is not None
        tensor = session.last_input["images"]
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0

    def test_returns_domain_predictions(self, image_bytes: bytes) -> None:
        session = StubSession(yolo_output([[320.0, 320.0, 64.0, 64.0, 0.1, 0.95]]))
        detector = OnnxObjectDetector(labels=LABELS, session=session)

        predictions = detector.predict(Image(content=image_bytes))

        assert [(p.class_name, round(p.score, 2)) for p in predictions] == [("dog", 0.95)]

    def test_reports_its_identity(self) -> None:
        detector = OnnxObjectDetector(
            labels=LABELS, session=StubSession(yolo_output([])), name="yolov8n", version="1.2.0"
        )

        assert (detector.info.name, detector.info.framework, detector.info.version) == (
            "yolov8n",
            "onnx",
            "1.2.0",
        )

    def test_inference_failure_becomes_detector_unavailable(self, image_bytes: bytes) -> None:
        detector = OnnxObjectDetector(
            labels=LABELS, session=StubSession(raises=RuntimeError("CUDA OOM"))
        )

        with pytest.raises(DetectorUnavailableError, match="inference failed"):
            detector.predict(Image(content=image_bytes))

    def test_an_unreadable_output_becomes_detector_unavailable(self, image_bytes: bytes) -> None:
        detector = OnnxObjectDetector(labels=LABELS, session=StubSession(np.zeros((2, 3, 4, 5))))

        with pytest.raises(DetectorUnavailableError, match="unsupported output layout"):
            detector.predict(Image(content=image_bytes))

    def test_a_missing_artifact_is_reported_at_load_time(self, tmp_path: Any) -> None:
        with pytest.raises(ModelLoadError, match="onnx model not found"):
            OnnxObjectDetector(model_path=tmp_path / "absent.onnx", labels=LABELS)

    def test_needs_either_a_path_or_a_session(self) -> None:
        with pytest.raises(ModelLoadError, match="model_path or a session"):
            OnnxObjectDetector(labels=LABELS)


class TestTorchScriptObjectDetector:
    def test_reads_the_torchvision_output_convention(self, image_bytes: bytes) -> None:
        # torchvision detectors return dicts of pixel-space boxes.
        def module(_: Any) -> list[dict[str, Any]]:
            return [
                {
                    "boxes": np.array([[0.0, 160.0, 640.0, 480.0]], dtype=np.float32),
                    "scores": np.array([0.87], dtype=np.float32),
                    "labels": np.array([1], dtype=np.int64),
                }
            ]

        detector = TorchScriptObjectDetector(labels=LABELS, module=module)
        predictions = detector.predict(Image(content=image_bytes))

        assert predictions[0].class_name == "dog"
        assert predictions[0].score == pytest.approx(0.87)

    def test_reads_a_raw_tensor_output(self, image_bytes: bytes) -> None:
        raw = yolo_output([[320.0, 320.0, 64.0, 64.0, 0.9, 0.1]])
        detector = TorchScriptObjectDetector(labels=LABELS, module=lambda _: raw)

        assert [p.class_name for p in detector.predict(Image(content=image_bytes))] == ["cat"]

    def test_uses_the_torchvision_input_contract(self, image_bytes: bytes) -> None:
        received: list[Any] = []

        def module(value: Any) -> list[dict[str, Any]]:
            received.append(value)
            return [{"boxes": [], "scores": [], "labels": []}]

        detector = TorchScriptObjectDetector(
            labels=LABELS,
            module=module,
            input_contract="torchvision",
        )
        detector.predict(Image(content=image_bytes))

        assert isinstance(received[0], list)
        assert received[0][0].shape[0] == 3

    def test_inference_failure_becomes_detector_unavailable(self, image_bytes: bytes) -> None:
        def module(_: Any) -> Any:
            raise RuntimeError("shape mismatch")

        detector = TorchScriptObjectDetector(labels=LABELS, module=module)

        with pytest.raises(DetectorUnavailableError, match="inference failed"):
            detector.predict(Image(content=image_bytes))

    def test_invalid_torchvision_output_becomes_detector_unavailable(
        self, image_bytes: bytes
    ) -> None:
        def module(_: Any) -> list[dict[str, Any]]:
            return [{"boxes": [[0, 0, 1, 1]], "scores": [2.0], "labels": [0]}]

        detector = TorchScriptObjectDetector(
            labels=LABELS,
            module=module,
            input_contract="torchvision",
        )

        with pytest.raises(DetectorUnavailableError, match="invalid detections"):
            detector.predict(Image(content=image_bytes))

    def test_reports_its_identity(self) -> None:
        detector = TorchScriptObjectDetector(labels=LABELS, module=lambda _: [], name="fasterrcnn")

        assert (detector.info.name, detector.info.framework) == ("fasterrcnn", "torchscript")
