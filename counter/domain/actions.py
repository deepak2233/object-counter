"""Use cases. One class per thing the service does."""

from __future__ import annotations

import logging

from counter.domain.models import CountResponse, DetectionResult, Image
from counter.domain.ports import ObjectCountRepo, ObjectDetector
from counter.domain.predictions import count, over_threshold, total, validate_threshold

logger = logging.getLogger(__name__)


class DetectObjects:
    """Return the predictions above the threshold. Reads nothing, writes nothing."""

    def __init__(self, object_detector: ObjectDetector) -> None:
        self._object_detector = object_detector

    def execute(self, image: Image, threshold: float) -> DetectionResult:
        threshold = validate_threshold(threshold)
        raw_predictions = self._object_detector.predict(image)
        predictions = over_threshold(raw_predictions, threshold)

        logger.info(
            "detection completed",
            extra={
                "model": self._object_detector.info.name,
                "threshold": threshold,
                "predictions_raw": len(raw_predictions),
                "predictions_kept": len(predictions),
            },
        )
        return DetectionResult(
            model=self._object_detector.info.name,
            threshold=threshold,
            predictions=predictions,
        )


class CountDetectedObjects:
    """Count the detected objects by class and accumulate the running totals.

    Built on top of DetectObjects rather than calling the detector itself, so the
    two endpoints can never drift: whatever /object-detect shows is exactly what
    /object-count counted.
    """

    def __init__(self, object_detector: ObjectDetector, object_count_repo: ObjectCountRepo) -> None:
        self._detect_objects = DetectObjects(object_detector)
        self._object_count_repo = object_count_repo

    def execute(self, image: Image, threshold: float) -> CountResponse:
        detection = self._detect_objects.execute(image, threshold)
        current_objects = count(detection.predictions)

        # Increment first, then read back, so the totals returned to this caller
        # already include this image.
        self._object_count_repo.update_values(current_objects)
        total_objects = self._object_count_repo.read_values()

        logger.info(
            "count completed",
            extra={
                "model": detection.model,
                "threshold": detection.threshold,
                "current_total": total(current_objects),
                "classes": len(current_objects),
            },
        )
        return CountResponse(
            model=detection.model,
            threshold=detection.threshold,
            current_objects=current_objects,
            total_objects=total_objects,
        )
