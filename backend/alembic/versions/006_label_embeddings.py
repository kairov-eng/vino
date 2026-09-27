"""embeddings_siglip2 + embeddings_dinov3 (pgvector 768)

Revision ID: 006_label_embeddings
Revises: 005_search_photos
Create Date: 2026-09-18
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text


revision: str = "006_label_embeddings"
down_revision: Union[str, None] = "005_search_photos"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ensure_table(table: str) -> None:
    """CREATE TABLE IF NOT EXISTS; HNSW — через SAVEPOINT (таблица может быть чужой)."""
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            wines_id integer PRIMARY KEY
                REFERENCES wines(id) ON DELETE CASCADE,
            embedding vector(768) NOT NULL
        )
        """
    )
    conn = op.get_bind()
    conn.execute(text("SAVEPOINT sp_hnsw"))
    try:
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS ix_{table}_embedding_hnsw
                ON {table}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
        conn.execute(text("RELEASE SAVEPOINT sp_hnsw"))
    except Exception:
        conn.execute(text("ROLLBACK TO SAVEPOINT sp_hnsw"))


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _ensure_table("embeddings_siglip2")
    _ensure_table("embeddings_dinov3")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embeddings_dinov3")
    op.execute("DROP TABLE IF EXISTS embeddings_siglip2")
