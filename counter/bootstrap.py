"""Composition root: the one place that knows which adapter is plugged in.

Wiring lives here so that every other module can be read without asking "which
implementation is this at runtime?", and so that a test can swap one adapter
without touching the app, the routes or the domain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import Engine

from counter.adapters.detector.artifacts import ArtifactStore
from counter.adapters.detector.catalog import ModelCatalog
from counter.adapters.detector.registry import CatalogModelRegistry
from counter.adapters.repo.memory import InMemoryObjectCountRepo
from counter.adapters.repo.sql import SqlObjectCountRepo, build_engine
from counter.config import Settings
from counter.domain.actions import CountDetectedObjects, DetectObjects
from counter.domain.ports import ObjectCountRepo, ObjectDetectorRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    """Everything the entrypoints need, already wired."""

    settings: Settings
    registry: ObjectDetectorRegistry
    repo: ObjectCountRepo
    engine: Engine | None = None

    def detect_action(self, model_name: str | None = None) -> DetectObjects:
        return DetectObjects(self.registry.get(model_name))

    def count_action(self, model_name: str | None = None) -> CountDetectedObjects:
        return CountDetectedObjects(self.registry.get(model_name), self.repo)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings()
    repo, engine = build_repo(settings)
    registry = build_registry(settings)

    if settings.preload_models and hasattr(registry, "warm"):
        registry.warm()

    logger.info(
        "services ready",
        extra={
            "env": settings.env,
            "persistence": settings.persistence,
            "catalog": str(settings.model_catalog),
            "models": [info.name for info in registry.available()],
        },
    )
    return Services(settings=settings, registry=registry, repo=repo, engine=engine)


def build_repo(settings: Settings) -> tuple[ObjectCountRepo, Engine | None]:
    """Pick the persistence adapter. Explicit branches, no dynamic lookup."""
    if settings.persistence == "sql":
        engine = build_engine(
            settings.database_url, echo=settings.db_echo, pool_size=settings.db_pool_size
        )
        return SqlObjectCountRepo(engine), engine

    if settings.is_production:
        # Counts that vanish on restart are not "totals", and with more than one
        # replica they are not even consistent between requests.
        logger.warning("in-memory counts in a production profile: totals will not survive restarts")
    return InMemoryObjectCountRepo(), None


def build_registry(settings: Settings) -> ObjectDetectorRegistry:
    assert settings.model_catalog is not None  # filled in by Settings validation
    catalog = ModelCatalog.from_file(settings.model_catalog)

    if settings.default_model and settings.default_model != catalog.default_model:
        catalog = ModelCatalog.model_validate(
            {**catalog.model_dump(), "default_model": settings.default_model}
        )

    return CatalogModelRegistry(
        catalog,
        ArtifactStore(settings.artifact_cache_dir),
        tfs_base_url=settings.tfs_base_url,
        tfs_timeout=settings.tfs_timeout_seconds,
        max_image_side=settings.max_image_side,
    )
