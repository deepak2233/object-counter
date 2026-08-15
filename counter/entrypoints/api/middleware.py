"""Request correlation, access logging and an upload size guard."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from counter.entrypoints.api.errors import HTTP_413_PAYLOAD_TOO_LARGE, error_response
from counter.observability.logging import request_id_var

logger = logging.getLogger("counter.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Tags every request with an id, then logs one structured access line.

    The id is echoed back in the response header and included in every log line
    and error body, so a caller reporting "my request failed" hands you the key
    to the exact trace.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return response
        finally:
            request_id_var.reset(token)


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized uploads on the declared Content-Length.

    Cheap first line of defence: it costs nothing and stops the obvious case
    before the body is buffered. The real enforcement is in the route, which
    measures the bytes it actually received — Content-Length can lie, and
    chunked uploads do not send one at all.
    """

    def __init__(self, app: object, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max_bytes:
            return error_response(
                "payload_too_large",
                f"request body is {declared} bytes, limit is {self._max_bytes} bytes",
                HTTP_413_PAYLOAD_TOO_LARGE,
            )
        return await call_next(request)
