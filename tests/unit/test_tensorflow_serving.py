"""The TF Serving adapter, driven through a stubbed transport.

No container, no model, no network: httpx's MockTransport gives the adapter a
real client whose responses the test writes, which is what makes the failure
paths (timeouts, 500s, truncated bodies) testable at all.
"""

from __future__ import annotations

import httpx
import pytest

from counter.adapters.detector.tensorflow_serving import (
    TensorFlowServingDetector,
    parse_predictions,
)
from counter.domain.errors import DetectorUnavailableError
from counter.domain.models import Image

pytestmark = pytest.mark.unit

LABELS = {1: "person", 17: "cat"}

TFS_BODY = {
    "predictions": [
        {
            "num_detections": 2.0,
            "detection_boxes": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            "detection_scores": [0.98, 0.42],
            "detection_classes": [17.0, 1.0],
        }
    ]
}


def detector_with(handler: object, **kwargs: object) -> TensorFlowServingDetector:
    client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return TensorFlowServingDetector(
        base_url="http://tfs:8501",
        model_name="rfcn",
        labels=LABELS,
        client=client,
        **kwargs,  # type: ignore[arg-type]
    )


class TestParsePredictions:
    def test_maps_boxes_scores_and_classes(self) -> None:
        predictions = parse_predictions(TFS_BODY, LABELS)

        assert [(p.class_name, p.score) for p in predictions] == [("cat", 0.98), ("person", 0.42)]
        # TFS emits [ymin, xmin, ymax, xmax]; the domain box is x-first.
        assert predictions[0].box.xmin == 0.2
        assert predictions[0].box.ymin == 0.1

    def test_returns_every_detection_including_low_scores(self) -> None:
        # Thresholding is the domain's job; an adapter that filters would make
        # the caller's threshold unenforceable.
        assert len(parse_predictions(TFS_BODY, LABELS)) == 2

    def test_survives_a_truncated_body(self) -> None:
        body = {
            "predictions": [
                {
                    "num_detections": 5.0,
                    "detection_boxes": [[0.1, 0.2, 0.3, 0.4]],
                    "detection_scores": [0.9],
                    "detection_classes": [17.0],
                }
            ]
        }

        assert len(parse_predictions(body, LABELS)) == 1

    @pytest.mark.parametrize(
        "body",
        [{}, {"predictions": []}, {"predictions": [{"num_detections": "many"}]}],
    )
    def test_rejects_a_body_it_cannot_read(self, body: dict) -> None:
        with pytest.raises(DetectorUnavailableError):
            parse_predictions(body, LABELS)


class TestTensorFlowServingDetector:
    def test_calls_the_predict_endpoint_of_the_named_model(self, image_bytes: bytes) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json=TFS_BODY)

        predictions = detector_with(handler).predict(Image(content=image_bytes))

        assert seen["url"] == "http://tfs:8501/v1/models/rfcn:predict"
        assert [prediction.class_name for prediction in predictions] == ["cat", "person"]

    def test_downscales_the_image_before_serialising_it(self, image_bytes: bytes) -> None:
        payloads: list[list] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            payloads.append(json.loads(request.content)["instances"][0])
            return httpx.Response(200, json=TFS_BODY)

        detector_with(handler, max_image_side=64).predict(Image(content=image_bytes))

        height = len(payloads[0])
        width = len(payloads[0][0])
        assert max(height, width) == 64

    def test_maps_a_server_error_to_detector_unavailable(self, image_bytes: bytes) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="model not loaded")

        with pytest.raises(DetectorUnavailableError, match="HTTP 500"):
            detector_with(handler).predict(Image(content=image_bytes))

    def test_maps_a_timeout_to_detector_unavailable(self, image_bytes: bytes) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        with pytest.raises(DetectorUnavailableError, match="unreachable"):
            detector_with(handler).predict(Image(content=image_bytes))

    def test_maps_a_non_json_body_to_detector_unavailable(self, image_bytes: bytes) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>proxy error</html>")

        with pytest.raises(DetectorUnavailableError, match="non-JSON"):
            detector_with(handler).predict(Image(content=image_bytes))

    def test_reports_its_identity(self) -> None:
        info = detector_with(lambda _request: httpx.Response(200, json=TFS_BODY)).info

        assert (info.name, info.framework) == ("rfcn", "tensorflow-serving")
