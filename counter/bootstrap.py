"""Application composition root."""

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
from counter.domain.errors import ModelLoadError
from counter.domain.ports import ObjectCountRepo, ObjectDetectorRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    """Application services."""

    settings: Settings
    registry: ObjectDetectorRegistry
    repo: ObjectCountRepo
    engine: Engine | None = None

    def detect_action(self, model_name: str | None = None) -> DetectObjects:
        return DetectObjects(self.registry.get(model_name))

    def count_action(self, model_name: str | None = None) -> CountDetectedObjects:
        return CountDetectedObjects(self.registry.get(model_name), self.repo)

    def close(self) -> None:
        try:
            self.registry.close()
        finally:
            if self.engine is not None:
                self.engine.dispose()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings()
    registry = build_registry(settings)
    repo, engine = build_repo(settings)

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
    """Build the configured repository."""
    if settings.persistence == "sql":
        engine = build_engine(
            settings.database_url, echo=settings.db_echo, pool_size=settings.db_pool_size
        )
        return SqlObjectCountRepo(engine), engine

    return InMemoryObjectCountRepo(), None


def build_registry(settings: Settings) -> ObjectDetectorRegistry:
    assert settings.model_catalog is not None  # filled in by Settings validation
    catalog = ModelCatalog.from_file(settings.model_catalog)
    if settings.is_production and any(spec.framework == "fake" for spec in catalog.models):
        raise ModelLoadError("fake detectors are not allowed in the prod profile")

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
