"""Relational adapter for ObjectCountRepo (assignment task 2).

Targets PostgreSQL in production. MySQL and SQLite are supported by the same
code path through dialect-specific upserts: SQLite makes the integration tests
runnable anywhere, and MySQL was the other option the assignment allowed.

The write is a single atomic `INSERT ... ON CONFLICT DO UPDATE SET count = count
+ excluded.count`. Reading a row, adding to it in Python and writing it back
would lose counts the moment two workers process images at the same time — and
"lost counts" in a counting service is the one bug the product cannot absorb.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.exc import SQLAlchemyError

from counter.domain.errors import RepositoryError
from counter.domain.models import ObjectCount
from counter.domain.ports import ObjectCountRepo

logger = logging.getLogger(__name__)

metadata = MetaData()

object_counts_table = Table(
    "object_counts",
    metadata,
    # The class name is the natural key: there is exactly one running total per
    # class, and the uniqueness constraint is what makes the upsert atomic.
    Column("object_class", String(128), primary_key=True),
    Column("count", BigInteger, nullable=False, server_default=text("0")),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)


def build_engine(database_url: str, *, echo: bool = False, pool_size: int = 5) -> Engine:
    """Create an engine with settings a long-running service needs.

    `pool_pre_ping` costs one round trip per checkout and removes the class of
    5xx you get after a database failover or an idle-connection reaper closes
    pooled sockets.
    """
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False},
        )

    return create_engine(
        database_url,
        echo=echo,
        future=True,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=pool_size * 2,
        pool_recycle=1800,
    )


class SqlObjectCountRepo(ObjectCountRepo):
    """ObjectCountRepo backed by a relational database."""

    def __init__(self, engine: Engine) -> None:
        # An Engine, not a URL: connection pooling belongs to the composition
        # root, and injecting it is what lets a test point the same adapter at
        # SQLite or at a throwaway Postgres schema.
        self._engine = engine

    def read_values(self, object_classes: Sequence[str] | None = None) -> list[ObjectCount]:
        # `count` is labelled: a Row exposes tuple.count(), so reading the
        # column as `row.count` is ambiguous to every reader and to the type
        # checker.
        statement = select(
            object_counts_table.c.object_class,
            object_counts_table.c.count.label("total"),
        ).order_by(object_counts_table.c.object_class)

        requested = list(object_classes) if object_classes is not None else None
        if requested is not None:
            if not requested:
                return []
            statement = statement.where(object_counts_table.c.object_class.in_(requested))

        try:
            with self._engine.connect() as connection:
                rows = connection.execute(statement).all()
        except SQLAlchemyError as exc:
            raise RepositoryError(f"could not read object counts: {exc}") from exc

        stored = {row.object_class: int(row.total) for row in rows}
        if requested is None:
            return [ObjectCount(name, count) for name, count in stored.items()]
        return [ObjectCount(name, stored.get(name, 0)) for name in requested]

    def update_values(self, new_values: Sequence[ObjectCount]) -> None:
        rows = _merge_duplicates(new_values)
        if not rows:
            return

        try:
            with self._engine.begin() as connection:
                connection.execute(_upsert_statement(self._engine.dialect.name, rows))
        except SQLAlchemyError as exc:
            raise RepositoryError(f"could not persist object counts: {exc}") from exc

    def health_check(self) -> None:
        """Reachable *and* migrated.

        `SELECT 1` would pass against a database where the migrations never ran,
        which is exactly the state where this instance must not take traffic, so
        the probe touches the table it actually needs.
        """
        try:
            with self._engine.connect() as connection:
                connection.execute(select(object_counts_table.c.object_class).limit(1))
        except SQLAlchemyError as exc:
            raise RepositoryError(f"database is not reachable or not migrated: {exc}") from exc


def _merge_duplicates(new_values: Sequence[ObjectCount]) -> list[dict[str, Any]]:
    """Collapse repeated classes and drop non-positive counts.

    A multi-row upsert that names the same key twice is a runtime error on
    PostgreSQL ("cannot affect row a second time"), so the batch is made unique
    before it reaches the database rather than trusting every caller.
    """
    totals: Counter[str] = Counter()
    for value in new_values:
        if value.count:
            totals[value.object_class] += value.count
    return [
        {"object_class": name, "count": count}
        for name, count in sorted(totals.items())
        if count > 0
    ]


def _upsert_statement(dialect: str, rows: list[dict[str, Any]]) -> Any:
    """Build the dialect's atomic increment-or-insert."""
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        postgres_statement = insert(object_counts_table).values(rows)
        return postgres_statement.on_conflict_do_update(
            index_elements=[object_counts_table.c.object_class],
            set_={
                "count": object_counts_table.c.count + postgres_statement.excluded.count,
                "updated_at": func.now(),
            },
        )

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        sqlite_statement = sqlite_insert(object_counts_table).values(rows)
        return sqlite_statement.on_conflict_do_update(
            index_elements=[object_counts_table.c.object_class],
            set_={
                "count": object_counts_table.c.count + sqlite_statement.excluded.count,
                "updated_at": func.now(),
            },
        )

    if dialect in {"mysql", "mariadb"}:
        from sqlalchemy.dialects.mysql import insert as mysql_insert

        mysql_statement = mysql_insert(object_counts_table).values(rows)
        return mysql_statement.on_duplicate_key_update(
            count=object_counts_table.c.count + mysql_statement.inserted.count,
            updated_at=func.now(),
        )

    raise RepositoryError(
        f"dialect '{dialect}' has no atomic upsert here; add one before deploying on it"
    )
