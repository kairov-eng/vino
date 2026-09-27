"""search_photos denormalized history columns for fast filters/sort

Revision ID: 011_search_photos_history_denorm
Revises: 010_manual_wines_id
Create Date: 2026-09-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "011_search_photos_history_denorm"
down_revision: Union[str, None] = "010_manual_wines_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "search_photos",
        sa.Column("hist_wine_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "search_photos",
        sa.Column("hist_wine_confidence", sa.Float(), nullable=True),
    )
    op.add_column(
        "search_photos",
        sa.Column("hist_wine_slug", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "search_photos",
        sa.Column("hist_total_ms", sa.Float(), nullable=True),
    )
    op.add_column(
        "search_photos",
        sa.Column(
            "hist_fp",
            sa.SmallInteger(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "search_photos",
        sa.Column(
            "hist_fn",
            sa.SmallInteger(),
            nullable=False,
            server_default="0",
        ),
    )
    op.create_index("ix_search_photos_hist_wine_id", "search_photos", ["hist_wine_id"])
    op.create_index("ix_search_photos_hist_wine_slug", "search_photos", ["hist_wine_slug"])
    op.create_index("ix_search_photos_hist_total_ms", "search_photos", ["hist_total_ms"])
    op.create_index("ix_search_photos_hist_fp", "search_photos", ["hist_fp"])
    op.create_index("ix_search_photos_hist_fn", "search_photos", ["hist_fn"])
    op.create_index(
        "ix_search_photos_hist_wine_confidence",
        "search_photos",
        ["hist_wine_confidence"],
    )

    # Backfill from status JSON using the same resolve rules as the API.
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id, status FROM search_photos")).mappings().all()
    # Import after columns exist; path is backend root when alembic runs.
    from app.scan_history_denorm import history_extract_fields

    update_sql = sa.text(
        """
        UPDATE search_photos SET
            hist_wine_id = :hist_wine_id,
            hist_wine_confidence = :hist_wine_confidence,
            hist_wine_slug = :hist_wine_slug,
            hist_total_ms = :hist_total_ms,
            hist_fp = :hist_fp,
            hist_fn = :hist_fn
        WHERE id = :id
        """
    )
    # Fill missing slugs from wines table in a second pass via JOIN where possible.
    wine_slugs = {
        int(r[0]): str(r[1])
        for r in conn.execute(
            sa.text("SELECT id, slug FROM wines WHERE slug IS NOT NULL")
        ).all()
        if r[1]
    }

    batch: list[dict] = []
    for row in rows:
        st = row["status"] if isinstance(row["status"], dict) else {}
        fields = history_extract_fields(st)
        mid = fields["hist_wine_id"]
        if mid is not None and not fields["hist_wine_slug"]:
            fields["hist_wine_slug"] = wine_slugs.get(int(mid))
        batch.append({"id": int(row["id"]), **fields})
        if len(batch) >= 200:
            for item in batch:
                conn.execute(update_sql, item)
            batch.clear()
    if batch:
        for item in batch:
            conn.execute(update_sql, item)


def downgrade() -> None:
    op.drop_index("ix_search_photos_hist_wine_confidence", table_name="search_photos")
    op.drop_index("ix_search_photos_hist_fn", table_name="search_photos")
    op.drop_index("ix_search_photos_hist_fp", table_name="search_photos")
    op.drop_index("ix_search_photos_hist_total_ms", table_name="search_photos")
    op.drop_index("ix_search_photos_hist_wine_slug", table_name="search_photos")
    op.drop_index("ix_search_photos_hist_wine_id", table_name="search_photos")
    op.drop_column("search_photos", "hist_fn")
    op.drop_column("search_photos", "hist_fp")
    op.drop_column("search_photos", "hist_total_ms")
    op.drop_column("search_photos", "hist_wine_slug")
    op.drop_column("search_photos", "hist_wine_confidence")
    op.drop_column("search_photos", "hist_wine_id")
