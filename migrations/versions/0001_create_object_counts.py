"""create object_counts

Revision ID: 0001
Revises:
Create Date: 2026-08-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "object_counts",
        # The class name is the primary key rather than a surrogate id: the
        # uniqueness constraint is what makes the counter's upsert atomic, and a
        # surrogate key would need a separate unique index to get it back.
        sa.Column("object_class", sa.String(length=128), primary_key=True),
        sa.Column("count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("count >= 0", name="ck_object_counts_count_non_negative"),
    )
    # Supports "which classes were seen recently" without a full scan.
    op.create_index("ix_object_counts_updated_at", "object_counts", ["updated_at"])


def downgrade() -> None:
    op.drop_index("ix_object_counts_updated_at", table_name="object_counts")
    op.drop_table("object_counts")
