"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from counter.bootstrap import Services, build_services
from counter.config import Settings
from counter.entrypoints.api.errors import register_exception_handlers
from counter.entrypoints.api.middleware import MaxBodySizeMiddleware, RequestContextMiddleware
from counter.entrypoints.api.routes import router
from counter.observability.logging import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
Detects objects in an image and counts the ones above a confidence threshold,
grouped by class.

* `POST /object-detect` returns the predictions.
* `POST /object-count` returns counts by class for this image plus the running totals.
* `GET /models` lists the servable models.
"""


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the app.

    `services` is injectable so tests can run the real HTTP stack against a fake
    detector and a throwaway database without monkeypatching module globals.
    """
    settings = settings or (services.settings if services else Settings())
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or build_services(settings)
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(
        title="Object Counter",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        # Docs stay on in every profile: an API nobody can read is an API nobody
        # can integrate against. Turn them off here if the service is public.
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_image_bytes)
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(router)
    return app


def main() -> None:
    """`python -m counter.entrypoints.api.app` — the local dev server."""
    import uvicorn

    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # logging is already configured; let ours win
    )


if __name__ == "__main__":
    main()
