"""search_photos.sha256 index (idempotent)

Revision ID: 007_search_photos_sha256_index
Revises: 006_label_embeddings
Create Date: 2026-09-18
"""

from typing import Sequence, Union

from alembic import op


revision: str = "007_search_photos_sha256_index"
down_revision: Union[str, None] = "006_label_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_search_photos_sha256 "
        "ON search_photos (sha256)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_search_photos_sha256")
