from __future__ import annotations

import pytest

from counter.adapters.detector.fake import FakeObjectDetector
from counter.adapters.repo.memory import InMemoryObjectCountRepo
from counter.domain.actions import CountDetectedObjects, DetectObjects
from counter.domain.errors import InvalidThresholdError
from counter.domain.models import Image, ObjectCount, Prediction

pytestmark = pytest.mark.unit


@pytest.fixture
def image() -> Image:
    return Image(content=b"not-decoded-by-the-fake", filename="test.jpg")


@pytest.fixture
def detector(predictions: list[Prediction]) -> FakeObjectDetector:
    return FakeObjectDetector(predictions, name="fake", validate_image=False)


class TestDetectObjects:
    def test_returns_predictions_above_the_threshold(
        self, detector: FakeObjectDetector, image: Image
    ) -> None:
        result = DetectObjects(detector).execute(image, 0.7)

        assert [prediction.class_name for prediction in result.predictions] == ["cat", "cat"]
        assert result.model == "fake"
        assert result.model_version == "1"
        assert result.threshold == 0.7

    def test_reports_the_threshold_it_applied(
        self, detector: FakeObjectDetector, image: Image
    ) -> None:
        assert DetectObjects(detector).execute(image, "0.5").threshold == 0.5  # type: ignore[arg-type]

    def test_rejects_an_impossible_threshold_before_calling_the_model(
        self, detector: FakeObjectDetector, image: Image
    ) -> None:
        with pytest.raises(InvalidThresholdError):
            DetectObjects(detector).execute(image, 42)


class TestCountDetectedObjects:
    def test_counts_by_class(self, detector: FakeObjectDetector, image: Image) -> None:
        response = CountDetectedObjects(detector, InMemoryObjectCountRepo()).execute(image, 0.5)

        assert response.current_objects == [ObjectCount("cat", 2), ObjectCount("dog", 1)]

    def test_totals_accumulate_across_images(
        self, detector: FakeObjectDetector, image: Image
    ) -> None:
        action = CountDetectedObjects(detector, InMemoryObjectCountRepo())

        action.execute(image, 0.5)
        response = action.execute(image, 0.5)

        assert response.current_objects == [ObjectCount("cat", 2), ObjectCount("dog", 1)]
        assert response.total_objects == [ObjectCount("cat", 4), ObjectCount("dog", 2)]

    def test_totals_include_the_image_just_counted(
        self, detector: FakeObjectDetector, image: Image
    ) -> None:
        response = CountDetectedObjects(detector, InMemoryObjectCountRepo()).execute(image, 0.5)

        assert response.total_objects == response.current_objects

    def test_counting_and_detecting_agree(self, detector: FakeObjectDetector, image: Image) -> None:
        # The two endpoints must never disagree about what was in the image.
        detected = DetectObjects(detector).execute(image, 0.55)
        counted = CountDetectedObjects(detector, InMemoryObjectCountRepo()).execute(image, 0.55)

        assert sum(item.count for item in counted.current_objects) == len(detected.predictions)

    def test_an_image_with_nothing_over_the_threshold_writes_nothing(
        self, detector: FakeObjectDetector, image: Image
    ) -> None:
        repo = InMemoryObjectCountRepo()

        response = CountDetectedObjects(detector, repo).execute(image, 1.0)

        assert response.current_objects == []
        assert repo.read_values("fake", "1") == []


def test_object_count_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        ObjectCount("cat", -1)
