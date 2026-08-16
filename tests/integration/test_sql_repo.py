"""Relational repository integration tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import Engine, text

from counter.adapters.repo.sql import (
    SqlObjectCountRepo,
    _merge_duplicates,
    _upsert_statement,
    build_engine,
)
from counter.domain.errors import RepositoryError
from counter.domain.models import ObjectCount
from tests.conftest import running_on_postgres

pytestmark = pytest.mark.integration
MODEL = ("fake", "1")


class TestReadAndWrite:
    def test_stores_and_reads_counts(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 2), ObjectCount("dog", 1)])

        assert sql_repo.read_values(*MODEL) == [ObjectCount("cat", 2), ObjectCount("dog", 1)]

    def test_updates_accumulate(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 2)])
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 3)])

        assert sql_repo.read_values(*MODEL, ["cat"]) == [ObjectCount("cat", 5)]

    def test_reads_are_ordered_by_class(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("zebra", 1), ObjectCount("ant", 1)])

        assert [item.object_class for item in sql_repo.read_values(*MODEL)] == ["ant", "zebra"]

    def test_filters_by_class(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 1), ObjectCount("dog", 1)])

        assert sql_repo.read_values(*MODEL, ["dog"]) == [ObjectCount("dog", 1)]

    def test_unknown_classes_read_as_zero(self, sql_repo: SqlObjectCountRepo) -> None:
        assert sql_repo.read_values(*MODEL, ["giraffe"]) == [ObjectCount("giraffe", 0)]

    def test_an_empty_filter_reads_nothing(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 1)])

        assert sql_repo.read_values(*MODEL, []) == []

    def test_writing_nothing_is_a_no_op(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [])
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 0)])

        assert sql_repo.read_values(*MODEL) == []

    def test_a_batch_repeating_a_class_is_merged(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 1), ObjectCount("cat", 2)])

        assert sql_repo.read_values(*MODEL, ["cat"]) == [ObjectCount("cat", 3)]

    def test_timestamps_move_when_a_class_is_updated(
        self, sql_repo: SqlObjectCountRepo, migrated_engine: Engine
    ) -> None:
        sql_repo.update_values(*MODEL, [ObjectCount("cat", 1)])
        with migrated_engine.connect() as connection:
            first = connection.execute(text("SELECT updated_at FROM object_counts")).scalar_one()

        sql_repo.update_values(*MODEL, [ObjectCount("cat", 1)])
        with migrated_engine.connect() as connection:
            second = connection.execute(text("SELECT updated_at FROM object_counts")).scalar_one()

        assert second >= first

    def test_counts_are_isolated_by_model_version(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.update_values("shelf", "1", [ObjectCount("can", 2)])
        sql_repo.update_values("shelf", "2", [ObjectCount("can", 5)])

        assert sql_repo.read_values("shelf", "1") == [ObjectCount("can", 2)]
        assert sql_repo.read_values("shelf", "2") == [ObjectCount("can", 5)]


class TestConcurrency:
    @pytest.mark.skipif(
        not running_on_postgres(),
        reason="needs a server that allows concurrent writers; set TEST_DATABASE_URL",
    )
    def test_parallel_increments_do_not_lose_counts(self, sql_repo: SqlObjectCountRepo) -> None:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda _: sql_repo.update_values(*MODEL, [ObjectCount("cat", 1)]),
                    range(200),
                )
            )

        assert sql_repo.read_values(*MODEL, ["cat"]) == [ObjectCount("cat", 200)]


class TestFailureModes:
    def test_health_check_passes_on_a_live_database(self, sql_repo: SqlObjectCountRepo) -> None:
        sql_repo.health_check()

    def test_health_check_fails_when_the_database_is_unreachable(self) -> None:
        engine = build_engine("postgresql+psycopg://nobody:nobody@127.0.0.1:1/none")

        with pytest.raises(RepositoryError, match="not reachable"):
            SqlObjectCountRepo(engine).health_check()

    def test_health_check_fails_when_the_migrations_have_not_run(self) -> None:
        repo = SqlObjectCountRepo(build_engine("sqlite+pysqlite:///:memory:"))

        with pytest.raises(RepositoryError, match="not migrated"):
            repo.health_check()

    def test_a_missing_schema_surfaces_as_a_repository_error(self) -> None:
        repo = SqlObjectCountRepo(build_engine("sqlite+pysqlite:///:memory:"))

        with pytest.raises(RepositoryError, match="could not read"):
            repo.read_values(*MODEL)

    def test_a_failed_write_surfaces_as_a_repository_error(self) -> None:
        repo = SqlObjectCountRepo(build_engine("sqlite+pysqlite:///:memory:"))

        with pytest.raises(RepositoryError, match="could not persist"):
            repo.update_values(*MODEL, [ObjectCount("cat", 1)])

    def test_an_unknown_dialect_is_refused_loudly(self) -> None:
        with pytest.raises(RepositoryError, match="no atomic upsert"):
            _upsert_statement("oracle", [{"object_class": "cat", "count": 1}])


class TestMergeDuplicates:
    def test_sums_repeated_classes_and_drops_empty_counts(self) -> None:
        merged = _merge_duplicates(
            *MODEL, [ObjectCount("cat", 1), ObjectCount("cat", 2), ObjectCount("dog", 0)]
        )

        assert merged == [
            {
                "model_name": "fake",
                "model_version": "1",
                "object_class": "cat",
                "count": 3,
            }
        ]
