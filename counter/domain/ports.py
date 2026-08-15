"""Ports: the interfaces the domain owns and the adapters implement.

The dependency rule is one-directional. Domain code may import from
`counter.domain`; it may never import from `counter.adapters` or
`counter.entrypoints`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from counter.domain.models import Image, ModelInfo, ObjectCount, Prediction


class ObjectDetector(ABC):
    """One trained model, ready to answer predictions."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Name, framework and version of the model behind this detector."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image) -> list[Prediction]:
        """Return every prediction the model produced, unfiltered.

        Thresholding is a domain decision, so detectors must not apply the
        caller's threshold themselves. A backend-internal floor (for example the
        score cut-off inside NMS) is an implementation detail and is documented
        per adapter.
        """
        raise NotImplementedError


class ObjectDetectorRegistry(ABC):
    """Resolves a model name to a detector.

    This is the seam that makes "several internally trained models" and "several
    deep learning frameworks" the same problem: both are just different entries
    in a catalog, resolved by name.
    """

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
        """Every model the service can serve, whether or not it is loaded yet."""
        raise NotImplementedError


class ObjectCountRepo(ABC):
    """Running totals of detected objects, by class."""

    @abstractmethod
    def read_values(self, object_classes: Sequence[str] | None = None) -> list[ObjectCount]:
        """Return stored counts, optionally restricted to the given classes.

        Classes that were never counted come back with a count of 0 rather than
        being omitted or returned as None, so callers never have to guard.
        """
        raise NotImplementedError

    @abstractmethod
    def update_values(self, new_values: Sequence[ObjectCount]) -> None:
        """Add the given counts to the stored totals.

        This is an increment, not a write: two concurrent callers counting one
        cat each must leave the store at two, never one. Implementations must
        make the whole batch atomic.
        """
        raise NotImplementedError

    def health_check(self) -> None:
        """Raise RepositoryError when the backing store is not usable.

        Default is a no-op for stores that cannot fail independently (memory).
        """
        return
