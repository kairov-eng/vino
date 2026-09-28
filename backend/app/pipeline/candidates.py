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

# Min cosine between current query embedding and a past search_photo embedding
# to reuse that search's winner wine + Google Vision OCR.
_REUSE_MIN_COSINE = 0.5


def search_top_wines_by_cosine(
    db: Session,
    *,
    search_photos_id: int,
    embedding_type: str,
    limit: int = 40,
    reuse_previous_searches: bool = False,
) -> dict[str, Any]:
    """Top-N wines by cosine similarity vs search_photo_embeddings row.

    Returns ``{"hits": [...], "reuse": None|dict}``.

    Each hit has ``search_photos_id``:
      - ``0`` — from catalog ANN;
      - ``>0`` — injected from that past search_photos row.

    When ``reuse_previous_searches`` and ``embedding_type==siglip2``:
      1) find nearest past search_photo (siglip2) that has a winner
         (manual_wines_id or matched/hist wine) and cosine ≥ 0.5;
      2) take that search's winner wine_id + Google Vision OCR text;
      3) return catalog top-``limit`` (full N) **without excluding** that wine,
         plus the winner as an **extra** hit (``search_photos_id=prev_sid``).
         Same bottle may appear twice: catalog + past search. Total = N or N+1.

    Otherwise classic catalog ANN LIMIT ``limit`` (all ``search_photos_id=0``).
    Catalog ANN is always full ``limit`` — never N−1 — whether reuse is on or off.
    """
    table = CATALOG_TABLES.get(embedding_type)
    if not table:
        raise ValueError(f"unknown embedding_type: {embedding_type}")

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
        return {"hits": [], "reuse": None}

    reuse_on = bool(reuse_previous_searches) and embedding_type == "siglip2"
    if not reuse_on:
        sql = text(
            f"""
            SELECT
                e.wines_id AS id,
                (1 - (e.embedding <=> current_setting('{_QVEC_GUC}')::vector))::float8
                    AS cosine_similarity,
                0 AS search_photos_id,
                false AS from_previous_search
            FROM {table} e
            ORDER BY e.embedding <=> current_setting('{_QVEC_GUC}')::vector
            LIMIT :lim
            """
        )
        rows = db.execute(sql, {"lim": int(limit)}).mappings().all()
        return {
            "hits": [_hit_row(r) for r in rows],
            "reuse": None,
        }

    # SigLIP2 + reuse: full catalog top-N, plus previous-search winner (N+1).
    sql = text(
        f"""
        WITH q AS (
            SELECT current_setting('{_QVEC_GUC}')::vector AS v
        ),
        prev AS (
            SELECT
                spe.search_photos_id AS prev_sid,
                (1 - (spe.embedding <=> (SELECT v FROM q)))::float8 AS search_cosine,
                COALESCE(
                    sp.manual_wines_id,
                    NULLIF(TRIM(sp.status ->> 'matched_wine_id'), '')::int,
                    sp.hist_wine_id
                ) AS wine_id,
                NULLIF(
                    TRIM(sp.status #>> '{{steps,ocr_google_vision,text}}'),
                    ''
                ) AS gv_ocr
            FROM search_photo_embeddings spe
            JOIN search_photos sp ON sp.id = spe.search_photos_id
            WHERE spe.embedding_type = 'siglip2'
              AND spe.search_photos_id <> :sid
              AND COALESCE(
                    sp.manual_wines_id,
                    NULLIF(TRIM(sp.status ->> 'matched_wine_id'), '')::int,
                    sp.hist_wine_id
                  ) IS NOT NULL
            ORDER BY spe.embedding <=> (SELECT v FROM q)
            LIMIT 1
        ),
        prev_hit AS (
            SELECT *
            FROM prev
            WHERE search_cosine >= :min_cos
              AND wine_id IS NOT NULL
        ),
        top_catalog AS (
            SELECT
                e.wines_id AS id,
                (1 - (e.embedding <=> (SELECT v FROM q)))::float8
                    AS cosine_similarity,
                0 AS search_photos_id,
                false AS from_previous_search,
                NULL::int AS prev_sid,
                NULL::float8 AS search_cosine,
                NULL::text AS gv_ocr
            FROM {table} e
            ORDER BY e.embedding <=> (SELECT v FROM q)
            LIMIT :lim
        ),
        prev_as_cand AS (
            SELECT
                p.wine_id AS id,
                COALESCE(
                    (
                        SELECT
                            (1 - (e.embedding <=> (SELECT v FROM q)))::float8
                        FROM {table} e
                        WHERE e.wines_id = p.wine_id
                    ),
                    p.search_cosine
                ) AS cosine_similarity,
                p.prev_sid AS search_photos_id,
                true AS from_previous_search,
                p.prev_sid,
                p.search_cosine,
                p.gv_ocr
            FROM prev_hit p
        )
        SELECT * FROM top_catalog
        UNION ALL
        SELECT * FROM prev_as_cand
        """
    )
    rows = db.execute(
        sql,
        {
            "sid": search_photos_id,
            "lim": int(limit),
            "min_cos": float(_REUSE_MIN_COSINE),
        },
    ).mappings().all()

    hits: list[dict[str, Any]] = []
    reuse: dict[str, Any] | None = None
    for r in rows:
        hits.append(_hit_row(r))
        if r.get("from_previous_search"):
            reuse = {
                "ok": True,
                "search_photos_id": int(r["prev_sid"])
                if r.get("prev_sid") is not None
                else None,
                "wine_id": int(r["id"]),
                "cosine": round(float(r["search_cosine"]), 6)
                if r.get("search_cosine") is not None
                else None,
                "google_vision_ocr": r.get("gv_ocr") or None,
                "min_cosine": float(_REUSE_MIN_COSINE),
            }

    return {"hits": hits, "reuse": reuse}


def _hit_row(r: Any) -> dict[str, Any]:
    try:
        sid = int(r.get("search_photos_id") or 0)
    except (TypeError, ValueError):
        sid = 0
    out: dict[str, Any] = {
        "id": int(r["id"]),
        "cosine_similarity": round(float(r["cosine_similarity"]), 6),
        "search_photos_id": sid,
    }
    if sid > 0 or r.get("from_previous_search"):
        out["from_previous_search"] = True
        gv = str(r.get("gv_ocr") or "").strip()
        if gv:
            # OCR этикетки из прошлого поиска — для UI и XGB compare.
            out["previous_search_ocr"] = gv
    return out
