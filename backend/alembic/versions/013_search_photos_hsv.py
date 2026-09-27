"""search_photos.hsv — generated Bhattacharyya distance query vs winner

Revision ID: 013_search_photos_hsv
Revises: 012_search_photos_coef
Create Date: 2026-09-23
"""

from typing import Sequence, Union

from alembic import op


revision: str = "013_search_photos_hsv"
down_revision: Union[str, None] = "012_search_photos_coef"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_HSV_EXPR = (
    "CASE "
    "WHEN jsonb_typeof(status #> '{steps,hsv,winner}') = 'number' "
    "THEN (status #>> '{steps,hsv,winner}')::double precision "
    "ELSE NULL END"
)


def upgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE search_photos
        ADD COLUMN hsv double precision
        GENERATED ALWAYS AS ({_HSV_EXPR}) STORED
        """
    )
    op.execute(
        """
        COMMENT ON COLUMN search_photos.hsv IS
        'Расстояние Бхаттачарии между HSV-гистограммами искомой этикетки и победителя (status.steps.hsv.winner)'
        """
    )
    op.create_index("ix_search_photos_hsv", "search_photos", ["hsv"])


def downgrade() -> None:
    op.drop_index("ix_search_photos_hsv", table_name="search_photos")
    op.drop_column("search_photos", "hsv")
