"""Dependency-free detector for development and tests."""

from __future__ import annotations

from collections.abc import Sequence

from counter.adapters.detector.images import decode
from counter.domain.models import Box, Image, ModelInfo, Prediction
from counter.domain.ports import ObjectDetector

DEFAULT_PREDICTIONS: tuple[Prediction, ...] = (
    Prediction("cat", 0.999190748, Box(0.367288858, 0.278333426, 0.735821366, 0.6988855)),
    Prediction("cat", 0.752194285, Box(0.101288858, 0.118333426, 0.335821366, 0.4988855)),
    Prediction("dog", 0.612194285, Box(0.501288858, 0.518333426, 0.935821366, 0.8988855)),
    Prediction("person", 0.312194285, Box(0.010288858, 0.010333426, 0.135821366, 0.1988855)),
)


class FakeObjectDetector(ObjectDetector):
    """Return a fixed set of predictions."""

    def __init__(
        self,
        predictions: Sequence[Prediction] = DEFAULT_PREDICTIONS,
        name: str = "fake",
        validate_image: bool = True,
    ) -> None:
        self._predictions = list(predictions)
        self._name = name
        self._validate_image = validate_image

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._name,
            framework="fake",
            version="1",
            labels=len({prediction.class_name for prediction in self._predictions}),
        )

    def predict(self, image: Image) -> list[Prediction]:
        if self._validate_image:
            decode(image)
        return list(self._predictions)
