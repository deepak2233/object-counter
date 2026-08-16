"""Migrations have to run forwards and backwards on the target database."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import inspect, text

from counter.adapters.repo.sql import build_engine
from tests.conftest import alembic_config

pytestmark = pytest.mark.integration


def test_upgrade_creates_the_counts_table(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COUNTER_DATABASE_URL", database_url)
    config = alembic_config()
    engine = build_engine(database_url)

    try:
        command.upgrade(config, "head")

        inspector = inspect(engine)
        assert "object_counts" in inspector.get_table_names()
        columns = {column["name"] for column in inspector.get_columns("object_counts")}
        assert columns == {
            "model_name",
            "model_version",
            "object_class",
            "count",
            "updated_at",
        }
    finally:
        command.downgrade(config, "base")
        engine.dispose()


def test_downgrade_removes_the_counts_table(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COUNTER_DATABASE_URL", database_url)
    config = alembic_config()
    engine = build_engine(database_url)

    try:
        command.upgrade(config, "head")
        command.downgrade(config, "base")

        assert "object_counts" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_upgrade_preserves_existing_counts(
    database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COUNTER_DATABASE_URL", database_url)
    config = alembic_config()
    engine = build_engine(database_url)

    try:
        command.upgrade(config, "0001")
        with engine.begin() as connection:
            connection.execute(
                text("INSERT INTO object_counts (object_class, count) VALUES ('cat', 3)")
            )

        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT model_name, model_version, object_class, count FROM object_counts")
            ).one()

        assert tuple(row) == ("legacy", "unknown", "cat", 3)
    finally:
        command.downgrade(config, "base")
        engine.dispose()
