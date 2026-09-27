"""search_photos + search_photo_embeddings

Revision ID: 005_search_photos
Revises: 004_wines_label
Create Date: 2026-09-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "005_search_photos"
down_revision: Union[str, None] = "004_wines_label"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "search_photos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(length=1024), nullable=False),
        sa.Column("filesize", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_search_photos_id", "search_photos", ["id"])
    op.create_index("ix_search_photos_created_at", "search_photos", ["created_at"])
    op.create_index("ix_search_photos_sha256", "search_photos", ["sha256"])

    op.execute(
        """
        CREATE TABLE search_photo_embeddings (
            id SERIAL PRIMARY KEY,
            search_photos_id INTEGER NOT NULL
                REFERENCES search_photos(id) ON DELETE CASCADE,
            embedding_type VARCHAR(64) NOT NULL,
            embedding vector(768) NOT NULL,
            CONSTRAINT ux_search_photo_embeddings_photo_type
                UNIQUE (search_photos_id, embedding_type)
        )
        """
    )
    op.create_index(
        "ix_search_photo_embeddings_search_photos_id",
        "search_photo_embeddings",
        ["search_photos_id"],
    )
    op.create_index(
        "ix_search_photo_embeddings_embedding_type",
        "search_photo_embeddings",
        ["embedding_type"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_search_photo_embeddings_embedding_type",
        table_name="search_photo_embeddings",
    )
    op.drop_index(
        "ix_search_photo_embeddings_search_photos_id",
        table_name="search_photo_embeddings",
    )
    op.drop_table("search_photo_embeddings")
    op.drop_index("ix_search_photos_sha256", table_name="search_photos")
    op.drop_index("ix_search_photos_created_at", table_name="search_photos")
    op.drop_index("ix_search_photos_id", table_name="search_photos")
    op.drop_table("search_photos")
