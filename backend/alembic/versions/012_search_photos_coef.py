"""search_photos.coef — generated YOLO conf of the selected label

Revision ID: 012_search_photos_coef
Revises: 011_search_photos_history_denorm
Create Date: 2026-09-23
"""

from typing import Sequence, Union

from alembic import op


revision: str = "012_search_photos_coef"
down_revision: Union[str, None] = "011_search_photos_history_denorm"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COEF_EXPR = (
    "CASE "
    "WHEN jsonb_typeof(status #> '{steps,yolo,selected,conf}') = 'number' "
    "THEN (status #>> '{steps,yolo,selected,conf}')::double precision "
    "ELSE NULL END"
)


def upgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE search_photos
        ADD COLUMN coef double precision
        GENERATED ALWAYS AS ({_COEF_EXPR}) STORED
        """
    )
    op.execute(
        """
        COMMENT ON COLUMN search_photos.coef IS
        'YOLO confidence выбранной этикетки (status.steps.yolo.selected.conf)'
        """
    )
    op.create_index("ix_search_photos_coef", "search_photos", ["coef"])


def downgrade() -> None:
    op.drop_index("ix_search_photos_coef", table_name="search_photos")
    op.drop_column("search_photos", "coef")
