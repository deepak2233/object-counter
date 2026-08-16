"""HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from counter.bootstrap import Services
from counter.domain.errors import ObjectCounterError, PayloadTooLargeError
from counter.domain.models import CountResponse, DetectionResult, Image
from counter.domain.predictions import validate_threshold
from counter.entrypoints.api.schemas import (
    CountResponseSchema,
    DetectResponseSchema,
    ErrorSchema,
    HealthSchema,
    ModelInfoSchema,
    ModelsResponseSchema,
)

router = APIRouter()
UPLOAD_CHUNK_BYTES = 1024 * 1024

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorSchema, "description": "Unknown model"},
    413: {"model": ErrorSchema, "description": "Payload is too large"},
    415: {"model": ErrorSchema, "description": "Payload is not a decodable image"},
    422: {"model": ErrorSchema, "description": "Threshold or form data is invalid"},
    503: {"model": ErrorSchema, "description": "Detector or store unavailable"},
}

ImageFile = Annotated[UploadFile, File(description="Image to run detection on")]
Threshold = Annotated[float | None, Form(description="Confidence cut-off, 0..1")]
ModelName = Annotated[str | None, Form(description="Model to use; defaults to the catalog default")]


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


@router.post(
    "/object-detect",
    response_model=DetectResponseSchema,
    responses=ERROR_RESPONSES,
    summary="Detect objects and return the predictions",
    tags=["detection"],
)
async def object_detect(
    file: ImageFile,
    threshold: Threshold = None,
    model_name: ModelName = None,
    services: Services = Depends(get_services),
) -> DetectResponseSchema:
    applied_threshold = _threshold_or_default(threshold, services)
    image = await _read_image(file, services)
    result = await run_in_threadpool(
        _detect,
        services,
        image,
        applied_threshold,
        model_name,
    )
    return DetectResponseSchema.of(result)


@router.post(
    "/object-count",
    response_model=CountResponseSchema,
    responses=ERROR_RESPONSES,
    summary="Count detected objects by class and accumulate the totals",
    tags=["detection"],
)
async def object_count(
    file: ImageFile,
    threshold: Threshold = None,
    model_name: ModelName = None,
    services: Services = Depends(get_services),
) -> CountResponseSchema:
    applied_threshold = _threshold_or_default(threshold, services)
    image = await _read_image(file, services)
    response = await run_in_threadpool(
        _count,
        services,
        image,
        applied_threshold,
        model_name,
    )
    return CountResponseSchema.of(response)


@router.get(
    "/models",
    response_model=ModelsResponseSchema,
    summary="List the models this service can serve",
    tags=["models"],
)
async def list_models(services: Services = Depends(get_services)) -> ModelsResponseSchema:
    default_model = getattr(services.registry, "default_model", "")
    return ModelsResponseSchema(
        default_model=default_model,
        models=[ModelInfoSchema.of(info) for info in services.registry.available()],
    )


@router.get("/healthz", response_model=HealthSchema, summary="Liveness", tags=["ops"])
async def healthz() -> HealthSchema:
    """Report process liveness."""
    return HealthSchema(status="ok")


@router.get("/readyz", response_model=HealthSchema, summary="Readiness", tags=["ops"])
async def readyz(response: Response, services: Services = Depends(get_services)) -> HealthSchema:
    """Check the repository and default detector."""
    checks: dict[str, str] = {}

    for name, probe in (
        ("repository", services.repo.health_check),
        ("detector", lambda: _check_default_detector(services)),
    ):
        try:
            await run_in_threadpool(probe)
            checks[name] = "ok"
        except ObjectCounterError:
            checks[name] = "error"

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthSchema(status="ready" if ready else "not_ready", checks=checks)


async def _read_image(file: UploadFile, services: Services) -> Image:
    limit = services.settings.max_image_bytes
    content = bytearray()
    while chunk := await file.read(min(UPLOAD_CHUNK_BYTES, limit + 1 - len(content))):
        content.extend(chunk)
        if len(content) > limit:
            raise PayloadTooLargeError(f"image exceeds the {limit} byte limit")
    return Image(
        content=bytes(content),
        filename=file.filename,
        content_type=file.content_type,
    )


def _threshold_or_default(threshold: float | None, services: Services) -> float:
    value = services.settings.default_threshold if threshold is None else threshold
    return validate_threshold(value)


def _detect(
    services: Services,
    image: Image,
    threshold: float,
    model_name: str | None,
) -> DetectionResult:
    return services.detect_action(model_name).execute(image, threshold)


def _count(
    services: Services,
    image: Image,
    threshold: float,
    model_name: str | None,
) -> CountResponse:
    return services.count_action(model_name).execute(image, threshold)


def _check_default_detector(services: Services) -> None:
    services.registry.get(None).health_check()
