"""Cosine ANN over catalog embeddings via pgvector."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

# User said embeddings_sdinov3; actual table in DB is embeddings_dinov3.
CATALOG_TABLES = {
    "siglip2": "embeddings_siglip2",
    "dinov3": "embeddings_dinov3",
}

# Transaction-local GUC: query vector stays in Postgres (no round-trip to app).
_QVEC_GUC = "vino.query_embedding"


def search_top_wines_by_cosine(
    db: Session,
    *,
    search_photos_id: int,
    embedding_type: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Top-N wines by cosine similarity vs search_photo_embeddings row.

    Two SQL ops on the same connection (no vector shipped to Python):
      1) load query embedding into a transaction-local GUC
      2) ANN ORDER BY embedding <=> current_setting(...)::vector LIMIT N
         so pgvector can use the HNSW index.
    """
    table = CATALOG_TABLES.get(embedding_type)
    if not table:
        raise ValueError(f"unknown embedding_type: {embedding_type}")

    # 1) Vector → session variable (SET LOCAL via set_config(..., is_local=true))
    loaded = db.execute(
        text(
            f"""
            SELECT set_config(
                '{_QVEC_GUC}',
                embedding::text,
                true
            ) AS ok
            FROM search_photo_embeddings
            WHERE search_photos_id = :sid
              AND embedding_type = :etype
            LIMIT 1
            """
        ),
        {"sid": search_photos_id, "etype": embedding_type},
    ).scalar()
    if not loaded:
        return []

    # 2) Top-N using that variable (HNSW-friendly: constant vector in ORDER BY)
    # <=> = cosine distance; similarity = 1 - distance
    sql = text(
        f"""
        SELECT
            e.wines_id AS id,
            (1 - (e.embedding <=> current_setting('{_QVEC_GUC}')::vector))::float8
                AS cosine_similarity
        FROM {table} e
        ORDER BY e.embedding <=> current_setting('{_QVEC_GUC}')::vector
        LIMIT :lim
        """
    )
    rows = db.execute(sql, {"lim": limit}).mappings().all()
    return [
        {
            "id": int(r["id"]),
            "cosine_similarity": round(float(r["cosine_similarity"]), 6),
        }
        for r in rows
    ]
