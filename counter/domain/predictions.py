"""Pure functions over predictions. No I/O, no state, trivially testable."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from counter.domain.errors import InvalidThresholdError
from counter.domain.models import ObjectCount, Prediction

MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0


def validate_threshold(threshold: float) -> float:
    """Return the threshold as a float, or raise if it is not a usable score.

    Confidence scores from a detector are probabilities in [0, 1], so a
    threshold outside that interval is a caller mistake, not an empty result:
    0.9 and 90 mean very different things and only one of them is a threshold.
    """
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise InvalidThresholdError(f"threshold must be a number, got {threshold!r}") from exc

    if not MIN_THRESHOLD <= value <= MAX_THRESHOLD:
        raise InvalidThresholdError(
            f"threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}, got {value}"
        )
    return value


def over_threshold(predictions: Iterable[Prediction], threshold: float) -> list[Prediction]:
    """Keep predictions whose score is at or above the threshold.

    Returns a list, not a generator: the caller counts them, logs them and
    serialises them, and a one-shot iterator makes that a bug waiting to happen.
    """
    return [prediction for prediction in predictions if prediction.score >= threshold]


def count(predictions: Iterable[Prediction]) -> list[ObjectCount]:
    """Group predictions by class name.

    Ordered by class name so that responses, logs and test assertions are
    reproducible; callers should not have to sort before comparing.
    """
    occurrences = Counter(prediction.class_name for prediction in predictions)
    return [ObjectCount(object_class, n) for object_class, n in sorted(occurrences.items())]


def total(counts: Iterable[ObjectCount]) -> int:
    """Accumulated number of objects across classes."""
    return sum(object_count.count for object_count in counts)
