"""search_photos.manual_wines_id — manual correspondence wine id

Revision ID: 010_manual_wines_id
Revises: 009_wines_label_ocr
Create Date: 2026-09-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "010_manual_wines_id"
down_revision: Union[str, None] = "009_wines_label_ocr"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "search_photos",
        sa.Column("manual_wines_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_search_photos_manual_wines_id",
        "search_photos",
        "wines",
        ["manual_wines_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_search_photos_manual_wines_id",
        "search_photos",
        ["manual_wines_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_search_photos_manual_wines_id", table_name="search_photos")
    op.drop_constraint(
        "fk_search_photos_manual_wines_id",
        "search_photos",
        type_="foreignkey",
    )
    op.drop_column("search_photos", "manual_wines_id")
