"""add wines.crop column

Revision ID: 003_wines_crop
Revises: 002_sitemap_tables
Create Date: 2026-09-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003_wines_crop"
down_revision: Union[str, None] = "002_sitemap_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "wines",
        sa.Column("crop", sa.Integer(), nullable=True, server_default="0"),
    )
    op.create_index("ix_wines_crop", "wines", ["crop"])


def downgrade() -> None:
    op.drop_index("ix_wines_crop", table_name="wines")
    op.drop_column("wines", "crop")
