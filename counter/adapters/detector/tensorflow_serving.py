"""TensorFlow Serving REST adapter (the backend the original service used)."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import httpx

from counter.adapters.detector.images import decode, resize_longest_side, to_uint8_array
from counter.adapters.detector.labels import class_name_for, load_labels
from counter.domain.errors import DetectorUnavailableError
from counter.domain.models import Box, Image, ModelInfo, Prediction
from counter.domain.ports import ObjectDetector

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_IMAGE_SIDE = 1024


class TensorFlowServingDetector(ObjectDetector):
    """Object detector backed by the TensorFlow Serving REST API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        remote_name: str | None = None,
        labels: Mapping[int, str] | None = None,
        *,
        version: str = "1",
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    ) -> None:
        self._model_name = model_name
        self._remote_name = remote_name or model_name
        self._version = version
        self._labels = labels if labels is not None else load_labels()
        self._max_image_side = max_image_side
        model_url = f"{base_url.rstrip('/')}/v1/models/{self._remote_name}/versions/{version}"
        self._url = f"{model_url}:predict"
        self._status_url = model_url
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(
                timeout=timeout,
                transport=httpx.HTTPTransport(retries=2),
            )
        )

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            framework="tensorflow-serving",
            version=self._version,
            labels=len(self._labels),
            metadata={"url": self._url},
        )

    def predict(self, image: Image) -> list[Prediction]:
        pil_image = resize_longest_side(decode(image), self._max_image_side)
        instance = to_uint8_array(pil_image).tolist()

        try:
            response = self._client.post(self._url, json={"instances": [instance]})
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise DetectorUnavailableError(
                f"{self._model_name} returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise DetectorUnavailableError(f"{self._model_name} is unreachable: {exc}") from exc
        except ValueError as exc:
            raise DetectorUnavailableError(f"{self._model_name} returned a non-JSON body") from exc

        return parse_predictions(body, self._labels, self._model_name)

    def health_check(self) -> None:
        try:
            response = self._client.get(self._status_url)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DetectorUnavailableError(f"{self._model_name} is not ready") from exc

        statuses = body.get("model_version_status", []) if isinstance(body, dict) else []
        if not any(
            isinstance(item, Mapping)
            and str(item.get("version")) == self._version
            and item.get("state") == "AVAILABLE"
            for item in statuses
        ):
            raise DetectorUnavailableError(f"{self._model_name} is not ready")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def parse_predictions(
    body: Any,
    labels: Mapping[int, str],
    model_name: str = "tfs",
) -> list[Prediction]:
    """Translate a TFS response into domain predictions."""
    try:
        raw = body["predictions"][0]
    except (TypeError, KeyError, IndexError) as exc:
        raise DetectorUnavailableError(f"{model_name} response has no predictions field") from exc

    try:
        num_detections = int(float(raw["num_detections"]))
        boxes = raw["detection_boxes"]
        scores = raw["detection_scores"]
        classes = raw["detection_classes"]
    except (TypeError, KeyError, ValueError) as exc:
        raise DetectorUnavailableError(f"{model_name} response is malformed: {exc}") from exc

    try:
        if num_detections < 0:
            raise ValueError("num_detections is negative")
        usable = min(num_detections, len(boxes), len(scores), len(classes))
    except (TypeError, ValueError) as exc:
        raise DetectorUnavailableError(f"{model_name} response is malformed: {exc}") from exc
    if usable < num_detections:
        logger.warning(
            "truncated detection arrays",
            extra={"model": model_name, "declared": num_detections, "usable": usable},
        )

    try:
        predictions: list[Prediction] = []
        for index in range(usable):
            ymin, xmin, ymax, xmax = (float(value) for value in boxes[index][:4])
            score = float(scores[index])
            class_id = float(classes[index])
            values = (ymin, xmin, ymax, xmax, score, class_id)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("detection contains a non-finite value")
            if not 0.0 <= score <= 1.0:
                raise ValueError("detection score is outside 0..1")
            if not class_id.is_integer():
                raise ValueError("class id is not an integer")
            if ymin > ymax or xmin > xmax:
                raise ValueError("detection box is inverted")

            predictions.append(
                Prediction(
                    class_name=class_name_for(labels, int(class_id)),
                    score=score,
                    box=Box(
                        xmin=_clip(xmin),
                        ymin=_clip(ymin),
                        xmax=_clip(xmax),
                        ymax=_clip(ymax),
                    ),
                )
            )
    except (TypeError, ValueError, IndexError) as exc:
        raise DetectorUnavailableError(f"{model_name} response is malformed: {exc}") from exc
    return predictions


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
