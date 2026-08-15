"""Shared fixtures.

Integration tests run against PostgreSQL when TEST_DATABASE_URL points at one,
and against a temporary SQLite file otherwise. That keeps `make test` runnable
on a laptop with nothing installed while CI exercises the real target dialect —
see docs/TESTING.md for what each level does and does not prove.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from PIL import Image as PilImage
from sqlalchemy import Engine

from counter.adapters.detector.fake import FakeObjectDetector
from counter.adapters.detector.registry import StaticModelRegistry
from counter.adapters.repo.memory import InMemoryObjectCountRepo
from counter.adapters.repo.sql import SqlObjectCountRepo, build_engine
from counter.bootstrap import Services
from counter.config import Settings
from counter.domain.models import Box, Prediction
from counter.entrypoints.api.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "resources" / "images"


@pytest.fixture(scope="session")
def image_bytes() -> bytes:
    return (IMAGES_DIR / "cat.jpg").read_bytes()


@pytest.fixture(scope="session")
def png_with_alpha(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """An RGBA PNG: the format the original decoder crashed on."""
    path = tmp_path_factory.mktemp("images") / "alpha.png"
    PilImage.new("RGBA", (32, 24), (255, 0, 0, 128)).save(path)
    return path.read_bytes()


@pytest.fixture(scope="session")
def grayscale_jpeg(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("images") / "gray.jpg"
    PilImage.new("L", (32, 24), 128).save(path)
    return path.read_bytes()


@pytest.fixture
def predictions() -> list[Prediction]:
    box = Box(0.1, 0.1, 0.2, 0.2)
    return [
        Prediction("cat", 0.95, box),
        Prediction("cat", 0.80, box),
        Prediction("dog", 0.60, box),
        Prediction("racket", 0.20, box),
    ]


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        log_level="WARNING",
        log_format="text",
        persistence="memory",
        artifact_cache_dir=tmp_path / "models",
        max_image_bytes=1024 * 1024,
    )


@pytest.fixture
def services(test_settings: Settings, predictions: list[Prediction]) -> Services:
    registry = StaticModelRegistry(
        {
            "fake": FakeObjectDetector(predictions, name="fake"),
            "other": FakeObjectDetector(predictions[:1], name="other"),
        },
        default_model="fake",
    )
    return Services(settings=test_settings, registry=registry, repo=InMemoryObjectCountRepo())


@pytest.fixture
def client(services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(services=services)) as test_client:
        yield test_client


# --- database ------------------------------------------------------------


def database_url_for(tmp_path: Path) -> str:
    return os.environ.get("TEST_DATABASE_URL") or f"sqlite+pysqlite:///{tmp_path / 'counter.db'}"


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    return database_url_for(tmp_path)


@pytest.fixture
def migrated_engine(database_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """A database whose schema was built by the real migrations.

    Not `metadata.create_all()`: that would test a schema nobody deploys and
    would let a broken migration reach production green.
    """
    monkeypatch.setenv("COUNTER_DATABASE_URL", database_url)
    config = alembic_config()

    command.upgrade(config, "head")
    engine = build_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()
        command.downgrade(config, "base")


@pytest.fixture
def sql_repo(migrated_engine: Engine) -> SqlObjectCountRepo:
    return SqlObjectCountRepo(migrated_engine)


def alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return config


def running_on_postgres() -> bool:
    return "postgresql" in os.environ.get("TEST_DATABASE_URL", "")
