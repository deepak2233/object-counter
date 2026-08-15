from __future__ import annotations

import pytest

from counter.domain.errors import InvalidThresholdError
from counter.domain.models import Box, ObjectCount, Prediction
from counter.domain.predictions import count, over_threshold, total, validate_threshold

BOX = Box(0.0, 0.0, 1.0, 1.0)


def prediction(class_name: str, score: float = 1.0) -> Prediction:
    return Prediction(class_name=class_name, score=score, box=BOX)


pytestmark = pytest.mark.unit


class TestOverThreshold:
    def test_keeps_predictions_at_or_above_the_threshold(self) -> None:
        kept = over_threshold([prediction("dog", 0.9), prediction("cat", 0.89)], 0.9)
        assert kept == [prediction("dog", 0.9)]

    def test_threshold_is_inclusive(self) -> None:
        assert over_threshold([prediction("cat", 0.5)], 0.5) == [prediction("cat", 0.5)]

    def test_zero_threshold_keeps_everything(self) -> None:
        predictions = [prediction("cat", 0.01), prediction("dog", 0.0)]
        assert over_threshold(predictions, 0.0) == predictions

    def test_returns_a_reusable_sequence(self) -> None:
        # A generator would be empty on the second pass, and the caller counts,
        # logs and serialises the same result.
        kept = over_threshold([prediction("cat", 0.9)], 0.5)
        assert list(kept) == list(kept)


class TestCount:
    def test_groups_by_class(self) -> None:
        counts = count([prediction("cat"), prediction("cat"), prediction("dog")])
        assert counts == [ObjectCount("cat", 2), ObjectCount("dog", 1)]

    def test_is_ordered_by_class_name(self) -> None:
        counts = count([prediction("racket"), prediction("cup"), prediction("cat")])
        assert [item.object_class for item in counts] == ["cat", "cup", "racket"]

    def test_empty_input_counts_nothing(self) -> None:
        assert count([]) == []

    def test_total_accumulates_across_classes(self) -> None:
        assert total(count([prediction("cat"), prediction("cat"), prediction("dog")])) == 3


class TestValidateThreshold:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, "0.75"])
    def test_accepts_scores_in_range(self, value: object) -> None:
        assert validate_threshold(value) == float(value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [-0.1, 1.1, 90, float("nan")])
    def test_rejects_scores_out_of_range(self, value: float) -> None:
        with pytest.raises(InvalidThresholdError):
            validate_threshold(value)

    @pytest.mark.parametrize("value", ["high", None, object()])
    def test_rejects_non_numbers(self, value: object) -> None:
        with pytest.raises(InvalidThresholdError):
            validate_threshold(value)  # type: ignore[arg-type]
