"""scope counts by model

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "object_counts_v2",
        sa.Column("model_name", sa.String(length=128), primary_key=True),
        sa.Column("model_version", sa.String(length=64), primary_key=True),
        sa.Column("object_class", sa.String(length=128), primary_key=True),
        sa.Column("count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("count >= 0", name="ck_object_counts_count_non_negative_v2"),
    )
    op.execute(
        """
        INSERT INTO object_counts_v2
            (model_name, model_version, object_class, count, updated_at)
        SELECT 'legacy', 'unknown', object_class, count, updated_at
        FROM object_counts
        """
    )
    op.drop_index("ix_object_counts_updated_at", table_name="object_counts")
    op.drop_table("object_counts")
    op.rename_table("object_counts_v2", "object_counts")
    op.create_index("ix_object_counts_updated_at", "object_counts", ["updated_at"])


def downgrade() -> None:
    op.create_table(
        "object_counts_v1",
        sa.Column("object_class", sa.String(length=128), primary_key=True),
        sa.Column("count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("count >= 0", name="ck_object_counts_count_non_negative_v1"),
    )
    op.execute(
        """
        INSERT INTO object_counts_v1 (object_class, count, updated_at)
        SELECT object_class, SUM(count), MAX(updated_at)
        FROM object_counts
        GROUP BY object_class
        """
    )
    op.drop_index("ix_object_counts_updated_at", table_name="object_counts")
    op.drop_table("object_counts")
    op.rename_table("object_counts_v1", "object_counts")
    op.create_index("ix_object_counts_updated_at", "object_counts", ["updated_at"])
