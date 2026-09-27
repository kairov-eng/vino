"""search_photo_embeddings — HNSW on embedding (cosine)

Revision ID: 016_search_photo_embeddings_hnsw
Revises: 015_xgb_exclude_photos
Create Date: 2026-09-27

Нужен для reuse_previous_searches: ANN по прошлым siglip2-векторам
без Seq Scan + sort по всем строкам.
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


revision: str = "016_search_photo_embeddings_hnsw"
down_revision: Union[str, None] = "015_xgb_exclude_photos"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Partial: reuse path filters embedding_type = 'siglip2'.
_INDEX = "ix_search_photo_embeddings_embedding_hnsw_siglip2"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn = op.get_bind()
    conn.execute(text("SAVEPOINT sp_hnsw_spe"))
    try:
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS {_INDEX}
                ON search_photo_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WHERE embedding_type = 'siglip2'
                """
            )
        )
        conn.execute(text("RELEASE SAVEPOINT sp_hnsw_spe"))
    except Exception:
        conn.execute(text("ROLLBACK TO SAVEPOINT sp_hnsw_spe"))
        raise


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
