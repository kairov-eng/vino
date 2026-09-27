"""wines.hsv — stored HSV histogram of the catalog label crop

Revision ID: 014_wines_hsv
Revises: 013_search_photos_hsv
Create Date: 2026-09-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "014_wines_hsv"
down_revision: Union[str, None] = "013_search_photos_hsv"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("wines", sa.Column("hsv", sa.LargeBinary(), nullable=True))
    op.execute(
        """
        COMMENT ON COLUMN wines.hsv IS
        'L2-normalized HSV histogram H×S 30×32 float32 row-major of the catalog label crop'
        """
    )


def downgrade() -> None:
    op.drop_column("wines", "hsv")
