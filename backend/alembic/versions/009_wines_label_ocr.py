"""add wines.label_ocr JSONB (structured OCR from label text)

Revision ID: 009_wines_label_ocr
Revises: 008_embeddings_siglip2_cpu
Create Date: 2026-09-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "009_wines_label_ocr"
down_revision: Union[str, None] = "008_embeddings_siglip2_cpu"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "wines",
        sa.Column("label_ocr", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("wines", "label_ocr")
