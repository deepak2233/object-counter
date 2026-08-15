"""Domain value objects.

Everything here is immutable and dependency-free: no PIL, no HTTP client, no ORM.
That is what lets the domain be unit tested without a single service running.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Box:
    """A bounding box in *normalised* coordinates (0..1), origin top-left.

    Normalised coordinates keep the domain independent of the resolution the
    model happened to run at, so a prediction stays comparable across backends.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True, slots=True)
class Prediction:
    class_name: str
    score: float
    box: Box


@dataclass(frozen=True, slots=True)
class ObjectCount:
    object_class: str
    count: int


@dataclass(frozen=True, slots=True)
class CountResponse:
    """Counts for the submitted image plus the running totals held by the repo."""

    model: str
    threshold: float
    current_objects: list[ObjectCount]
    total_objects: list[ObjectCount]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Payload of the detection endpoint: the predictions themselves."""

    model: str
    threshold: float
    predictions: list[Prediction]


@dataclass(frozen=True, slots=True)
class Image:
    """An image as it crosses the domain boundary.

    Bytes rather than a file handle on purpose. A `BinaryIO` is stateful: the
    first reader exhausts it and the second one silently sees an empty stream,
    which is exactly the class of bug that hides behind "it works in the test".
    """

    content: bytes
    filename: str | None = None
    content_type: str | None = None

    def __len__(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What the service can tell a caller about a servable model."""

    name: str
    framework: str
    version: str = "unknown"
    labels: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
