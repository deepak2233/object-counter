"""HTTP routes.

Endpoints are `async def` but push the blocking work (decode, inference, SQL)
onto the threadpool. Running inference directly in the event loop would stall
every other in-flight request for the duration of a forward pass.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from counter.bootstrap import Services
from counter.domain.errors import InvalidImageError, ObjectCounterError
from counter.domain.models import Image
from counter.entrypoints.api.schemas import (
    CountResponseSchema,
    DetectResponseSchema,
    ErrorSchema,
    HealthSchema,
    ModelInfoSchema,
    ModelsResponseSchema,
)

router = APIRouter()

ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorSchema, "description": "Unknown model"},
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
    """Assignment task 1: same input as /object-count, but the predictions themselves."""
    image = await _read_image(file, services)
    action = services.detect_action(model_name)
    result = await run_in_threadpool(
        action.execute, image, _threshold_or_default(threshold, services)
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
    image = await _read_image(file, services)
    action = services.count_action(model_name)
    response = await run_in_threadpool(
        action.execute, image, _threshold_or_default(threshold, services)
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
    """Is the process up. Deliberately checks nothing else: a liveness probe that
    depends on the database restarts healthy pods during a database incident."""
    return HealthSchema(status="ok")


@router.get("/readyz", response_model=HealthSchema, summary="Readiness", tags=["ops"])
async def readyz(response: Response, services: Services = Depends(get_services)) -> HealthSchema:
    """Can this instance serve traffic: store reachable, default model resolvable."""
    checks: dict[str, str] = {}

    for name, probe in (
        ("repository", services.repo.health_check),
        ("detector", lambda: services.registry.get(None)),
    ):
        try:
            await run_in_threadpool(probe)
            checks[name] = "ok"
        except ObjectCounterError as exc:
            checks[name] = f"error: {exc}"

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthSchema(status="ready" if ready else "not_ready", checks=checks)


async def _read_image(file: UploadFile, services: Services) -> Image:
    content = await file.read()
    limit = services.settings.max_image_bytes
    if len(content) > limit:
        raise InvalidImageError(f"image is {len(content)} bytes, limit is {limit} bytes")
    return Image(
        content=content,
        filename=file.filename,
        content_type=file.content_type,
    )


def _threshold_or_default(threshold: float | None, services: Services) -> float:
    return services.settings.default_threshold if threshold is None else threshold
