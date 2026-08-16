"""Interfaces implemented by adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from counter.domain.models import Image, ModelInfo, ObjectCount, Prediction


class ObjectDetector(ABC):
    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Name, framework and version of the model behind this detector."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image) -> list[Prediction]:
        """Return predictions before the caller threshold is applied."""
        raise NotImplementedError

    def health_check(self) -> None:
        return

    def close(self) -> None:
        return


class ObjectDetectorRegistry(ABC):
    @abstractmethod
    def get(self, model_name: str | None = None) -> ObjectDetector:
        """Return the named detector, or the configured default when None.

        Raises:
            ModelNotFoundError: the name is not in the catalog.
            ModelLoadError: the name is known but its artifact will not load.
        """
        raise NotImplementedError

    @abstractmethod
    def available(self) -> list[ModelInfo]:
        """Return configured models without forcing them to load."""
        raise NotImplementedError

    def close(self) -> None:
        return


class ObjectCountRepo(ABC):
    @abstractmethod
    def read_values(
        self,
        model_name: str,
        model_version: str,
        object_classes: Sequence[str] | None = None,
    ) -> list[ObjectCount]:
        raise NotImplementedError

    @abstractmethod
    def update_values(
        self,
        model_name: str,
        model_version: str,
        new_values: Sequence[ObjectCount],
    ) -> None:
        raise NotImplementedError

    def health_check(self) -> None:
        return
