"""embeddings_siglip2.embedding_cpu (CPU server vector)

Revision ID: 008_embeddings_siglip2_cpu
Revises: 007_search_photos_sha256_index
Create Date: 2026-09-20
"""

from typing import Sequence, Union

from alembic import op


revision: str = "008_embeddings_siglip2_cpu"
down_revision: Union[str, None] = "007_search_photos_sha256_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE embeddings_siglip2
        ADD COLUMN IF NOT EXISTS embedding_cpu vector(768)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE embeddings_siglip2
        DROP COLUMN IF EXISTS embedding_cpu
        """
    )
