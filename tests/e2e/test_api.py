"""End-to-end: real HTTP stack, real routing, real error handling.

Only the model is a fake, and that is deliberate — asserting on the output of a
real detector would test the weights, not the service.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


def post_image(
    client: TestClient, path: str, image_bytes: bytes, **data: object
) -> pytest.ExceptionInfo | object:
    return client.post(
        path,
        files={"file": ("cat.jpg", image_bytes, "image/jpeg")},
        data={key: str(value) for key, value in data.items()},
    )


class TestObjectDetect:
    def test_returns_the_predictions_above_the_threshold(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        response = post_image(client, "/object-detect", image_bytes, threshold=0.7)

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert [prediction["class_name"] for prediction in body["predictions"]] == ["cat", "cat"]
        assert body["predictions"][0]["box"].keys() == {"xmin", "ymin", "xmax", "ymax"}

    def test_threshold_changes_the_result(self, client: TestClient, image_bytes: bytes) -> None:
        low = post_image(client, "/object-detect", image_bytes, threshold=0.1).json()
        high = post_image(client, "/object-detect", image_bytes, threshold=0.99).json()

        assert low["count"] == 4
        assert high["count"] == 0

    def test_threshold_is_optional(self, client: TestClient, image_bytes: bytes) -> None:
        response = post_image(client, "/object-detect", image_bytes)

        assert response.status_code == 200
        assert response.json()["threshold"] == 0.5

    def test_reports_which_model_answered(self, client: TestClient, image_bytes: bytes) -> None:
        response = post_image(client, "/object-detect", image_bytes, model_name="other")

        assert response.json()["model"] == "other"

    def test_predictions_are_serialisable_json(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        prediction = post_image(client, "/object-detect", image_bytes).json()["predictions"][0]

        assert isinstance(prediction["score"], float)
        assert all(0.0 <= value <= 1.0 for value in prediction["box"].values())


class TestObjectCount:
    def test_counts_by_class(self, client: TestClient, image_bytes: bytes) -> None:
        response = post_image(client, "/object-count", image_bytes, threshold=0.5)

        assert response.status_code == 200
        body = response.json()
        assert body["current_objects"] == [
            {"object_class": "cat", "count": 2},
            {"object_class": "dog", "count": 1},
        ]
        assert body["current_total"] == 3

    def test_totals_accumulate_across_requests(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        post_image(client, "/object-count", image_bytes, threshold=0.5)
        body = post_image(client, "/object-count", image_bytes, threshold=0.5).json()

        assert body["total_objects"] == [
            {"object_class": "cat", "count": 4},
            {"object_class": "dog", "count": 2},
        ]
        assert body["accumulated_total"] == 6

    def test_keeps_the_original_response_field_names(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        # Clients of the original service must not have to change.
        body = post_image(client, "/object-count", image_bytes, threshold=0.5).json()

        assert {"current_objects", "total_objects"} <= body.keys()

    def test_counting_agrees_with_detecting(self, client: TestClient, image_bytes: bytes) -> None:
        detected = post_image(client, "/object-detect", image_bytes, threshold=0.6).json()
        counted = post_image(client, "/object-count", image_bytes, threshold=0.6).json()

        assert counted["current_total"] == detected["count"]


class TestErrors:
    @pytest.mark.parametrize("threshold", [-0.5, 1.5, 90])
    def test_a_threshold_outside_zero_to_one_is_rejected(
        self, client: TestClient, image_bytes: bytes, threshold: float
    ) -> None:
        response = post_image(client, "/object-count", image_bytes, threshold=threshold)

        assert response.status_code == 422
        assert response.json()["error"]["type"] == "invalid_threshold"

    def test_a_non_numeric_threshold_is_rejected(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        response = post_image(client, "/object-count", image_bytes, threshold="high")

        assert response.status_code == 422
        assert response.json()["error"]["type"] == "validation_error"

    def test_a_missing_file_is_rejected(self, client: TestClient) -> None:
        response = client.post("/object-count", data={"threshold": "0.5"})

        assert response.status_code == 422
        assert response.json()["error"]["message"].startswith("file:")

    def test_a_payload_that_is_not_an_image_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/object-count", files={"file": ("notes.txt", b"just some text", "text/plain")}
        )

        assert response.status_code == 415
        assert response.json()["error"]["type"] == "invalid_image"

    def test_an_oversized_upload_is_rejected(self, client: TestClient) -> None:
        oversized = b"\xff" * (1024 * 1024 + 1)

        response = client.post(
            "/object-count", files={"file": ("big.jpg", oversized, "image/jpeg")}
        )

        assert response.status_code == 413
        assert response.json()["error"]["type"] == "payload_too_large"

    def test_an_unknown_model_is_a_404(self, client: TestClient, image_bytes: bytes) -> None:
        response = post_image(client, "/object-detect", image_bytes, model_name="nope")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "model_not_found"

    def test_every_error_carries_the_request_id(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        response = post_image(client, "/object-count", image_bytes, threshold=7)

        assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


class TestOperations:
    def test_liveness_does_not_depend_on_anything(self, client: TestClient) -> None:
        assert client.get("/healthz").json() == {"status": "ok", "checks": {}}

    def test_readiness_checks_the_store_and_the_default_model(self, client: TestClient) -> None:
        body = client.get("/readyz").json()

        assert body["status"] == "ready"
        assert body["checks"] == {"repository": "ok", "detector": "ok"}

    def test_lists_the_servable_models(self, client: TestClient) -> None:
        body = client.get("/models").json()

        assert body["default_model"] == "fake"
        assert {model["name"] for model in body["models"]} == {"fake", "other"}

    def test_publishes_an_openapi_document(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        assert {"/object-detect", "/object-count", "/models"} <= paths.keys()

    def test_echoes_a_caller_supplied_request_id(
        self, client: TestClient, image_bytes: bytes
    ) -> None:
        response = client.post(
            "/object-detect",
            files={"file": ("cat.jpg", image_bytes, "image/jpeg")},
            headers={"X-Request-ID": "trace-me"},
        )

        assert response.headers["X-Request-ID"] == "trace-me"
