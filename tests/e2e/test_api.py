"""HTTP application tests with injected detectors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from counter.bootstrap import Services
from counter.domain.errors import DetectorUnavailableError

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
        assert (body["model"], body["model_version"]) == ("fake", "1")
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
        assert body["model_version"] == "1"

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
        body = post_image(client, "/object-count", image_bytes, threshold=0.5).json()

        assert {"current_objects", "total_objects"} <= body.keys()

    def test_counting_agrees_with_detecting(self, client: TestClient, image_bytes: bytes) -> None:
        detected = post_image(client, "/object-detect", image_bytes, threshold=0.6).json()
        counted = post_image(client, "/object-count", image_bytes, threshold=0.6).json()

        assert counted["current_total"] == detected["count"]

    def test_totals_are_isolated_by_model(self, client: TestClient, image_bytes: bytes) -> None:
        post_image(client, "/object-count", image_bytes, model_name="fake")
        body = post_image(client, "/object-count", image_bytes, model_name="other").json()

        assert body["total_objects"] == [{"object_class": "cat", "count": 1}]


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
        assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]

    def test_multipart_overhead_does_not_reduce_the_file_limit(self, client: TestClient) -> None:
        response = client.post(
            "/object-count",
            files={"file": ("invalid.jpg", b"x" * (1024 * 1024), "image/jpeg")},
        )

        assert response.status_code == 415

    def test_an_unknown_model_is_a_404(self, client: TestClient, image_bytes: bytes) -> None:
        response = post_image(client, "/object-detect", image_bytes, model_name="nope")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "model_not_found"

    def test_detector_failures_do_not_expose_backend_details(
        self,
        client: TestClient,
        services: Services,
        image_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(_: object) -> None:
            raise DetectorUnavailableError("http://secret-model:8501 returned credentials")

        monkeypatch.setattr(services.registry.get(), "predict", fail)
        response = post_image(client, "/object-detect", image_bytes)

        assert response.status_code == 503
        assert response.json()["error"]["message"] == "detector is unavailable"

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

    def test_readiness_fails_when_the_detector_is_unavailable(
        self,
        client: TestClient,
        services: Services,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def unavailable() -> None:
            raise DetectorUnavailableError("http://internal-model:8501 failed")

        monkeypatch.setattr(services.registry.get(), "health_check", unavailable)
        response = client.get("/readyz")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "checks": {"repository": "ok", "detector": "error"},
        }

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

    def test_replaces_an_invalid_request_id(self, client: TestClient, image_bytes: bytes) -> None:
        supplied = "x" * 100
        response = client.post(
            "/object-detect",
            files={"file": ("cat.jpg", image_bytes, "image/jpeg")},
            headers={"X-Request-ID": supplied},
        )

        assert response.headers["X-Request-ID"] != supplied
        assert len(response.headers["X-Request-ID"]) == 32
