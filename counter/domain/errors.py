"""Errors shared across the application boundary."""

from __future__ import annotations


class ObjectCounterError(Exception):
    """Base class for every error this service raises deliberately."""


class InvalidThresholdError(ObjectCounterError):
    """Threshold outside the closed interval [0, 1]."""


class InvalidImageError(ObjectCounterError):
    """Payload is empty or not a decodable image."""


class PayloadTooLargeError(ObjectCounterError):
    """The uploaded image exceeds the configured limit."""


class ModelNotFoundError(ObjectCounterError):
    """Requested model is not in the catalog."""


class ModelLoadError(ObjectCounterError):
    """Model is in the catalog but its artifact could not be loaded."""


class DetectorUnavailableError(ObjectCounterError):
    """The inference backend is unreachable, timed out, or answered garbage."""


class RepositoryError(ObjectCounterError):
    """Persistence is unreachable or rejected the write."""
