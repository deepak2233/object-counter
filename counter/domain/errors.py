"""Domain errors.

The domain raises these; the entrypoint layer is the only place that knows how
they map onto HTTP status codes (see counter/entrypoints/api/errors.py). Adapters
translate infrastructure failures (httpx, SQLAlchemy, onnxruntime) into these so
that no driver-specific exception ever reaches a caller.
"""

from __future__ import annotations


class ObjectCounterError(Exception):
    """Base class for every error this service raises deliberately."""


class InvalidThresholdError(ObjectCounterError):
    """Threshold outside the closed interval [0, 1]."""


class InvalidImageError(ObjectCounterError):
    """Payload is empty, too large, or not a decodable image."""


class ModelNotFoundError(ObjectCounterError):
    """Requested model is not in the catalog."""


class ModelLoadError(ObjectCounterError):
    """Model is in the catalog but its artifact could not be loaded."""


class DetectorUnavailableError(ObjectCounterError):
    """The inference backend is unreachable, timed out, or answered garbage."""


class RepositoryError(ObjectCounterError):
    """Persistence is unreachable or rejected the write."""
