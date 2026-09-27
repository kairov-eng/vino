"""add wines.label column

Revision ID: 004_wines_label
Revises: 003_wines_crop
Create Date: 2026-09-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "004_wines_label"
down_revision: Union[str, None] = "003_wines_crop"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wines", sa.Column("label", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("wines", "label")
