"""TensorFlow Serving REST adapter (the backend the original service used)."""

from __future__ import annotations

import logging
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
    """Calls the TFS REST predict API for an object-detection SavedModel.

    Differences from the original adapter, all of them things that bite in
    production rather than in a demo:
      * a connect/read timeout, so a hung TFS cannot pin every worker forever;
      * transport-level retries for connection errors;
      * non-2xx and malformed bodies raise DetectorUnavailableError instead of
        a KeyError from `response.json()['predictions']`;
      * the image is downscaled before serialisation, because the payload is
        JSON-encoded pixel integers: a 4000x3000 photo is ~36M numbers, roughly
        90 MB of JSON per request;
      * float class ids resolve correctly against the int-keyed label map.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        labels: Mapping[int, str] | None = None,
        *,
        version: str = "1",
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    ) -> None:
        self._model_name = model_name
        self._version = version
        self._labels = labels if labels is not None else load_labels()
        self._max_image_side = max_image_side
        self._url = f"{base_url.rstrip('/')}/v1/models/{model_name}:predict"
        self._client = client or httpx.Client(
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=2),
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


def parse_predictions(
    body: Any,
    labels: Mapping[int, str],
    model_name: str = "tfs",
) -> list[Prediction]:
    """Translate a TFS predict response into domain predictions.

    A module-level function so the wire format can be tested without a client,
    a socket, or a running TensorFlow Serving.
    """
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

    # Trust the shortest of the parallel arrays rather than num_detections alone:
    # a truncated body would otherwise raise IndexError deep inside the loop.
    usable = min(num_detections, len(boxes), len(scores), len(classes))
    if usable < num_detections:
        logger.warning(
            "truncated detection arrays",
            extra={"model": model_name, "declared": num_detections, "usable": usable},
        )

    predictions: list[Prediction] = []
    for index in range(usable):
        ymin, xmin, ymax, xmax = boxes[index][:4]
        predictions.append(
            Prediction(
                class_name=class_name_for(labels, classes[index]),
                score=float(scores[index]),
                box=Box(xmin=float(xmin), ymin=float(ymin), xmax=float(xmax), ymax=float(ymax)),
            )
        )
    return predictions
