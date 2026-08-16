"""HTTP request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from counter.domain.models import (
    Box,
    CountResponse,
    DetectionResult,
    ModelInfo,
    ObjectCount,
    Prediction,
)


class BoxSchema(BaseModel):
    xmin: float = Field(description="Left edge, normalised to 0..1")
    ymin: float = Field(description="Top edge, normalised to 0..1")
    xmax: float = Field(description="Right edge, normalised to 0..1")
    ymax: float = Field(description="Bottom edge, normalised to 0..1")

    @classmethod
    def of(cls, box: Box) -> BoxSchema:
        return cls(xmin=box.xmin, ymin=box.ymin, xmax=box.xmax, ymax=box.ymax)


class PredictionSchema(BaseModel):
    class_name: str
    score: float
    box: BoxSchema

    @classmethod
    def of(cls, prediction: Prediction) -> PredictionSchema:
        return cls(
            class_name=prediction.class_name,
            score=prediction.score,
            box=BoxSchema.of(prediction.box),
        )


class ObjectCountSchema(BaseModel):
    object_class: str
    count: int

    @classmethod
    def of(cls, object_count: ObjectCount) -> ObjectCountSchema:
        return cls(object_class=object_count.object_class, count=object_count.count)


class DetectResponseSchema(BaseModel):
    """Response of POST /object-detect."""

    model_config = {"protected_namespaces": ()}

    model: str = Field(description="Model that produced these predictions")
    model_version: str
    threshold: float
    count: int = Field(description="Number of predictions above the threshold")
    predictions: list[PredictionSchema]

    @classmethod
    def of(cls, result: DetectionResult) -> DetectResponseSchema:
        return cls(
            model=result.model,
            model_version=result.model_version,
            threshold=result.threshold,
            count=len(result.predictions),
            predictions=[PredictionSchema.of(prediction) for prediction in result.predictions],
        )


class CountResponseSchema(BaseModel):
    """Response of POST /object-count."""

    model_config = {"protected_namespaces": ()}

    model: str
    model_version: str
    threshold: float
    current_objects: list[ObjectCountSchema]
    current_total: int = Field(description="Objects counted in this image")
    total_objects: list[ObjectCountSchema]
    accumulated_total: int = Field(description="Objects counted since the store was created")

    @classmethod
    def of(cls, response: CountResponse) -> CountResponseSchema:
        return cls(
            model=response.model,
            model_version=response.model_version,
            threshold=response.threshold,
            current_objects=[ObjectCountSchema.of(item) for item in response.current_objects],
            current_total=sum(item.count for item in response.current_objects),
            total_objects=[ObjectCountSchema.of(item) for item in response.total_objects],
            accumulated_total=sum(item.count for item in response.total_objects),
        )


class ModelInfoSchema(BaseModel):
    name: str
    framework: str
    version: str
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def of(cls, info: ModelInfo) -> ModelInfoSchema:
        return cls(
            name=info.name,
            framework=info.framework,
            version=info.version,
            metadata=info.metadata,
        )


class ModelsResponseSchema(BaseModel):
    default_model: str
    models: list[ModelInfoSchema]


class HealthSchema(BaseModel):
    status: str
    checks: dict[str, str] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    type: str
    message: str
    request_id: str


class ErrorSchema(BaseModel):
    error: ErrorDetail
