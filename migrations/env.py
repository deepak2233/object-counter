"""Alembic environment.

Schema changes ship as migrations, never as `metadata.create_all()` against a
live database: the same script has to be reviewable, replayable and reversible
in every environment.
"""

from __future__ import annotations

from alembic import context

from counter.adapters.repo.sql import build_engine, metadata
from counter.config import Settings

target_metadata = metadata


def database_url() -> str:
    override = context.get_x_argument(as_dictionary=True).get("db_url")
    return override or Settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = build_engine(database_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
