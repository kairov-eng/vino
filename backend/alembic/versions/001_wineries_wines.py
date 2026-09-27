"""create wineries and wines tables

Revision ID: 001_wineries_wines
Revises:
Create Date: 2026-03-16
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001_wineries_wines"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "wineries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("url", sa.String(length=1024), nullable=False),
        sa.Column("photo", sa.String(length=2048), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wineries_id", "wineries", ["id"])
    op.create_index("ix_wineries_name", "wineries", ["name"])
    op.create_index("ix_wineries_url", "wineries", ["url"])
    op.create_index("ix_wineries_photo", "wineries", ["photo"])

    op.create_table(
        "wines",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("wineries_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=True),
        sa.Column("color", sa.String(length=256), nullable=True),
        sa.Column("region", sa.String(length=256), nullable=True),
        sa.Column("grape_variety", sa.String(length=512), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("winery", sa.String(length=512), nullable=True),
        sa.Column("slug", sa.String(length=512), nullable=True),
        sa.Column("photo_name", sa.String(length=1024), nullable=True),
        sa.ForeignKeyConstraint(["wineries_id"], ["wineries.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_wines_id", "wines", ["id"])
    op.create_index("ix_wines_wineries_id", "wines", ["wineries_id"])
    op.create_index("ix_wines_name", "wines", ["name"])
    op.create_index("ix_wines_category", "wines", ["category"])
    op.create_index("ix_wines_color", "wines", ["color"])
    op.create_index("ix_wines_region", "wines", ["region"])
    op.create_index("ix_wines_grape_variety", "wines", ["grape_variety"])
    op.create_index("ix_wines_description", "wines", ["description"], postgresql_using="hash")
    op.create_index("ix_wines_winery", "wines", ["winery"])
    op.create_index("ix_wines_slug", "wines", ["slug"])
    op.create_index("ix_wines_photo_name", "wines", ["photo_name"])


def downgrade() -> None:
    op.drop_index("ix_wines_photo_name", table_name="wines")
    op.drop_index("ix_wines_slug", table_name="wines")
    op.drop_index("ix_wines_winery", table_name="wines")
    op.drop_index("ix_wines_description", table_name="wines", postgresql_using="hash")
    op.drop_index("ix_wines_grape_variety", table_name="wines")
    op.drop_index("ix_wines_region", table_name="wines")
    op.drop_index("ix_wines_color", table_name="wines")
    op.drop_index("ix_wines_category", table_name="wines")
    op.drop_index("ix_wines_name", table_name="wines")
    op.drop_index("ix_wines_wineries_id", table_name="wines")
    op.drop_index("ix_wines_id", table_name="wines")
    op.drop_table("wines")

    op.drop_index("ix_wineries_photo", table_name="wineries")
    op.drop_index("ix_wineries_url", table_name="wineries")
    op.drop_index("ix_wineries_name", table_name="wineries")
    op.drop_index("ix_wineries_id", table_name="wineries")
    op.drop_table("wineries")
