"""create wines_sitemap and wineries_sitemap tables

Revision ID: 002_sitemap_tables
Revises: 001_wineries_wines
Create Date: 2026-09-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002_sitemap_tables"
down_revision: Union[str, None] = "001_wineries_wines"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wines_sitemap",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("loc", sa.String(length=1024), nullable=False),
        sa.Column("slug", sa.String(length=512), nullable=False),
        sa.Column("lastmod", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image_loc", sa.String(length=2048), nullable=True),
        sa.Column("image_file", sa.String(length=1024), nullable=True),
        sa.Column("image_title", sa.String(length=1024), nullable=True),
        sa.Column("image_caption", sa.String(length=1024), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wines_sitemap_id", "wines_sitemap", ["id"])
    op.create_index("ix_wines_sitemap_loc", "wines_sitemap", ["loc"])
    op.create_index("ix_wines_sitemap_slug", "wines_sitemap", ["slug"])
    op.create_index("ix_wines_sitemap_lastmod", "wines_sitemap", ["lastmod"])
    op.create_index("ix_wines_sitemap_image_loc", "wines_sitemap", ["image_loc"])
    op.create_index("ix_wines_sitemap_image_file", "wines_sitemap", ["image_file"])
    op.create_index("ix_wines_sitemap_image_title", "wines_sitemap", ["image_title"])
    op.create_index("ix_wines_sitemap_image_caption", "wines_sitemap", ["image_caption"])

    op.create_table(
        "wineries_sitemap",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("loc", sa.String(length=1024), nullable=False),
        sa.Column("slug", sa.String(length=512), nullable=False),
        sa.Column("lastmod", sa.DateTime(timezone=True), nullable=True),
        sa.Column("image_loc", sa.String(length=2048), nullable=True),
        sa.Column("image_file", sa.String(length=1024), nullable=True),
        sa.Column("image_title", sa.String(length=1024), nullable=True),
        sa.Column("image_caption", sa.String(length=1024), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wineries_sitemap_id", "wineries_sitemap", ["id"])
    op.create_index("ix_wineries_sitemap_loc", "wineries_sitemap", ["loc"])
    op.create_index("ix_wineries_sitemap_slug", "wineries_sitemap", ["slug"])
    op.create_index("ix_wineries_sitemap_lastmod", "wineries_sitemap", ["lastmod"])
    op.create_index("ix_wineries_sitemap_image_loc", "wineries_sitemap", ["image_loc"])
    op.create_index("ix_wineries_sitemap_image_file", "wineries_sitemap", ["image_file"])
    op.create_index("ix_wineries_sitemap_image_title", "wineries_sitemap", ["image_title"])
    op.create_index(
        "ix_wineries_sitemap_image_caption", "wineries_sitemap", ["image_caption"]
    )


def downgrade() -> None:
    op.drop_index("ix_wineries_sitemap_image_caption", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_image_title", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_image_file", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_image_loc", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_lastmod", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_slug", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_loc", table_name="wineries_sitemap")
    op.drop_index("ix_wineries_sitemap_id", table_name="wineries_sitemap")
    op.drop_table("wineries_sitemap")

    op.drop_index("ix_wines_sitemap_image_caption", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_image_title", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_image_file", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_image_loc", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_lastmod", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_slug", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_loc", table_name="wines_sitemap")
    op.drop_index("ix_wines_sitemap_id", table_name="wines_sitemap")
    op.drop_table("wines_sitemap")
