"""The API on top of the relational adapter: the production wiring.

Runs against PostgreSQL when TEST_DATABASE_URL is set, SQLite otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from counter.adapters.detector.fake import FakeObjectDetector
from counter.adapters.detector.registry import StaticModelRegistry
from counter.adapters.repo.sql import SqlObjectCountRepo, build_engine
from counter.bootstrap import Services
from counter.config import Settings
from counter.domain.models import Prediction
from counter.entrypoints.api.app import create_app

pytestmark = pytest.mark.e2e


@pytest.fixture
def sql_services(
    test_settings: Settings, migrated_engine: Engine, predictions: list[Prediction]
) -> Services:
    return Services(
        settings=test_settings,
        registry=StaticModelRegistry({"fake": FakeObjectDetector(predictions, name="fake")}),
        repo=SqlObjectCountRepo(migrated_engine),
    )


@pytest.fixture
def sql_client(sql_services: Services) -> Iterator[TestClient]:
    with TestClient(create_app(services=sql_services)) as client:
        yield client


def count_once(client: TestClient, image_bytes: bytes, threshold: float = 0.5) -> dict:
    response = client.post(
        "/object-count",
        files={"file": ("cat.jpg", image_bytes, "image/jpeg")},
        data={"threshold": str(threshold)},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_counts_are_written_to_the_database(
    sql_client: TestClient, migrated_engine: Engine, image_bytes: bytes
) -> None:
    count_once(sql_client, image_bytes)

    with migrated_engine.connect() as connection:
        rows = connection.execute(
            text("SELECT object_class, count FROM object_counts ORDER BY object_class")
        ).all()

    assert [tuple(row) for row in rows] == [("cat", 2), ("dog", 1)]


def test_totals_accumulate_across_requests(sql_client: TestClient, image_bytes: bytes) -> None:
    count_once(sql_client, image_bytes)
    body = count_once(sql_client, image_bytes)

    assert body["accumulated_total"] == 6


def test_totals_survive_a_restart(
    sql_services: Services, migrated_engine: Engine, image_bytes: bytes
) -> None:
    # The point of the relational adapter: a redeploy does not reset the product.
    with TestClient(create_app(services=sql_services)) as first_boot:
        count_once(first_boot, image_bytes)

    with TestClient(create_app(services=sql_services)) as second_boot:
        body = count_once(second_boot, image_bytes)

    assert body["accumulated_total"] == 6


def test_an_unreachable_store_is_a_503_not_a_500(
    test_settings: Settings, predictions: list[Prediction], image_bytes: bytes
) -> None:
    # No migrations run against this database, so every statement fails.
    broken = Services(
        settings=test_settings,
        registry=StaticModelRegistry({"fake": FakeObjectDetector(predictions, name="fake")}),
        repo=SqlObjectCountRepo(build_engine("sqlite+pysqlite:///:memory:")),
    )

    with TestClient(create_app(services=broken)) as client:
        response = client.post(
            "/object-count", files={"file": ("cat.jpg", image_bytes, "image/jpeg")}
        )
        readiness = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "repository"
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "not_ready"
