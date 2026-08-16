"""Domain value objects."""

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

    def __post_init__(self) -> None:
        if not self.object_class:
            raise ValueError("object_class must not be empty")
        if self.count < 0:
            raise ValueError("count must not be negative")


@dataclass(frozen=True, slots=True)
class CountResponse:
    model: str
    model_version: str
    threshold: float
    current_objects: list[ObjectCount]
    total_objects: list[ObjectCount]


@dataclass(frozen=True, slots=True)
class DetectionResult:
    model: str
    model_version: str
    threshold: float
    predictions: list[Prediction]


@dataclass(frozen=True, slots=True)
class Image:
    content: bytes
    filename: str | None = None
    content_type: str | None = None

    def __len__(self) -> int:
        return len(self.content)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    framework: str
    version: str = "unknown"
    labels: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
