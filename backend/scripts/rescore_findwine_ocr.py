"""Refresh ocr_wine_id + matched_wine for an existing scan using new FinalScore."""
from __future__ import annotations

import sys

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.config import DATABASE_URL
from app.db.models import SearchPhoto, Wine
from app.pipeline.candidates import search_top_wines_by_cosine
from app.pipeline.exclusive_lexicon import evaluate_exclusive_lexicon
from app.pipeline.ocr_match import evaluate_ocr_wine_id_help


def main(photo_id: int) -> None:
    eng = create_engine(DATABASE_URL)
    with Session(eng) as db:
        row = db.get(SearchPhoto, photo_id)
        if row is None:
            raise SystemExit(f"search_photos id={photo_id} not found")
        st = dict(row.status or {})
        ocr = (st.get("steps") or {}).get("ocr") or {}
        ids = list(st.get("candidates_siglip2_ids") or [])
        for wid in st.get("candidates_dinov3_ids") or []:
            if int(wid) not in ids:
                ids.append(int(wid))
        cos: dict[int, float] = {}
        # prefer live cosine from embeddings
        try:
            hits = search_top_wines_by_cosine(
                db,
                search_photos_id=photo_id,
                embedding_type="siglip2",
                limit=max(20, len(ids) or 20),
            )
            for h in hits:
                cos[int(h["id"])] = float(h["cosine_similarity"])
                if int(h["id"]) not in ids:
                    ids.append(int(h["id"]))
        except Exception as exc:  # noqa: BLE001
            print("cosine reload failed:", exc)
            for k, v in (st.get("candidates_cosine") or {}).items():
                cos[int(k)] = float(v)

        query_text = (
            str(ocr.get("text") or "").strip()
            or str(ocr.get("text_aggregated") or "").strip()
            or str(ocr.get("text_norm") or "").strip()
        )
        wine_snaps: list[dict] = []
        if ids:
            rows = list(db.scalars(select(Wine).where(Wine.id.in_(ids))).all())
            by_id = {int(w.id): w for w in rows}
            for wid in ids:
                w = by_id.get(int(wid))
                if w is None:
                    continue
                wine_snaps.append(
                    {
                        "id": int(w.id),
                        "name": w.name,
                        "label": w.label,
                        "category": w.category,
                        "color": w.color,
                        "grape_variety": w.grape_variety,
                        "winery": w.winery,
                    }
                )
        exclusive = evaluate_exclusive_lexicon(query_text, wine_snaps)
        exclusive_reject_ids = {int(x) for x in (exclusive.get("rejected_ids") or [])}

        help_ = evaluate_ocr_wine_id_help(
            db,
            ocr_step=ocr,
            visual_candidate_ids=ids,
            cosine_by_id=cos,
            top_k=max(20, len(ids)),
            exclusive_reject_ids=exclusive_reject_ids,
        )
        steps = dict(st.get("steps") or {})
        steps["exclusive_lexicon"] = exclusive
        steps["ocr_wine_id"] = help_
        st["steps"] = steps
        st["candidates_cosine"] = {str(k): round(v, 6) for k, v in cos.items()}
        st["algorithm_version"] = __import__(
            "app.pipeline.version", fromlist=["ALGORITHM_VERSION"]
        ).ALGORITHM_VERSION
        st["algorithm_updated_at"] = __import__(
            "app.pipeline.version", fromlist=["ALGORITHM_UPDATED_AT"]
        ).ALGORITHM_UPDATED_AT
        st["algorithm_notes"] = __import__(
            "app.pipeline.version", fromlist=["ALGORITHM_NOTES"]
        ).ALGORITHM_NOTES

        bf = help_.get("best_final") or {}
        if bf.get("id") is not None:
            st["matched_wine_id"] = int(bf["id"])
            st["matched_wine_confidence"] = bf.get("final_score")
            st["matched_wine"] = {
                "id": int(bf["id"]),
                "confidence": bf.get("final_score"),
                "source": "final_score",
                "final_score": bf.get("final_score"),
                "text_score": bf.get("score"),
                "text_score_01": bf.get("score_01"),
                "cosine_similarity": bf.get("cosine"),
                "text_from": bf.get("text_from"),
                "name": bf.get("name"),
                "winery": bf.get("winery"),
            }
        else:
            st["matched_wine_id"] = None
            st["matched_wine_confidence"] = None
            veto = help_.get("soft_veto")
            st["matched_wine"] = {
                "id": None,
                "confidence": None,
                "source": "ocr_rejected_all",
                "reason": (
                    (veto or {}).get("reason")
                    if isinstance(veto, dict)
                    else "OCR rejected all / Soft TF-IDF veto"
                ),
                "soft_veto": veto,
            }
        row.status = st
        db.add(row)
        db.commit()
        print("updated", photo_id, "matched", st.get("matched_wine_id"), st.get("matched_wine"))
        print("final top:")
        for r in (help_.get("final_ranked") or [])[:5]:
            print(
                f"  id={r['id']} final={r['final_score']} "
                f"text={r['score']} cos={r.get('cosine')} name={r.get('name')}"
            )


if __name__ == "__main__":
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 131
    main(pid)
