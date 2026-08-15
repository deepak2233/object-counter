"""Domain error -> HTTP status. The only module that knows about both."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from counter.domain.errors import (
    DetectorUnavailableError,
    InvalidImageError,
    InvalidThresholdError,
    ModelLoadError,
    ModelNotFoundError,
    ObjectCounterError,
    RepositoryError,
)
from counter.observability.logging import request_id_var

logger = logging.getLogger(__name__)

# Starlette renamed 413/422 between releases (REQUEST_ENTITY_TOO_LARGE ->
# CONTENT_TOO_LARGE, UNPROCESSABLE_ENTITY -> UNPROCESSABLE_CONTENT). Numbers do
# not get deprecated, so the two that moved are spelled out here.
HTTP_413_PAYLOAD_TOO_LARGE = 413
HTTP_422_UNPROCESSABLE_CONTENT = 422

STATUS_BY_ERROR: dict[type[ObjectCounterError], int] = {
    InvalidThresholdError: HTTP_422_UNPROCESSABLE_CONTENT,
    InvalidImageError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    ModelNotFoundError: status.HTTP_404_NOT_FOUND,
    ModelLoadError: status.HTTP_503_SERVICE_UNAVAILABLE,
    DetectorUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    RepositoryError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def error_response(error_type: str, message: str, status_code: int) -> JSONResponse:
    """One error envelope for the whole API, so clients parse one shape."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
                "request_id": request_id_var.get(),
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ObjectCounterError)
    async def _domain_error(_: Request, exc: ObjectCounterError) -> JSONResponse:
        status_code = STATUS_BY_ERROR.get(type(exc), status.HTTP_400_BAD_REQUEST)
        logger.warning(
            "request rejected",
            extra={"error_type": type(exc).__name__, "status": status_code, "detail": str(exc)},
        )
        return error_response(_snake_case(type(exc).__name__), str(exc), status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            "validation_error",
            "; ".join(
                f"{'.'.join(str(part) for part in item['loc'][1:])}: {item['msg']}"
                for item in exc.errors()
            )
            or "request is not valid",
            HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Log the detail, return none of it: stack traces and driver messages
        # are for the operator, not for the caller.
        logger.exception("unhandled error", extra={"error_type": type(exc).__name__})
        return error_response(
            "internal_error",
            "the service failed to process this request",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _snake_case(name: str) -> str:
    trimmed = name.removesuffix("Error")
    return "".join(f"_{char.lower()}" if char.isupper() else char for char in trimmed).lstrip("_")
