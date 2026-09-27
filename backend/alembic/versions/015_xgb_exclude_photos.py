"""xgb_exclude_photos — файлы, поиски по которым не брать в train XGB

Revision ID: 015_xgb_exclude_photos
Revises: 014_wines_hsv
Create Date: 2026-09-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "015_xgb_exclude_photos"
down_revision: Union[str, None] = "014_wines_hsv"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "xgb_exclude_photos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("folder", sa.String(length=2048), nullable=False),
        sa.Column("filename", sa.String(length=1024), nullable=False),
        sa.Column("filesize", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "filename",
            "sha256",
            name="ux_xgb_exclude_photos_filename_sha256",
        ),
    )
    op.create_index(
        "ix_xgb_exclude_photos_sha256",
        "xgb_exclude_photos",
        ["sha256"],
        unique=False,
    )
    op.create_index(
        "ix_xgb_exclude_photos_folder",
        "xgb_exclude_photos",
        ["folder"],
        unique=False,
    )
    op.execute(
        """
        COMMENT ON TABLE xgb_exclude_photos IS
        'Файлы из eval/holdout-папок: sha256 как у search_photos; '
        'при prepare XGB (--exclude-listed-photos) сканы с тем же hash не в train'
        """
    )


def downgrade() -> None:
    op.drop_index("ix_xgb_exclude_photos_folder", table_name="xgb_exclude_photos")
    op.drop_index("ix_xgb_exclude_photos_sha256", table_name="xgb_exclude_photos")
    op.drop_table("xgb_exclude_photos")
