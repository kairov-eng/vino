"""Query OCR longer than catalog label; successful match without False Positive."""
from __future__ import annotations

import math
from collections import Counter

from app.database import SessionLocal
from app.db.models import SearchPhoto, Wine
from app.pipeline.xgb_match import collect_query_ocr_text


def n_lines(text: str) -> int:
    return len([ln.strip() for ln in str(text or "").splitlines() if ln.strip()])


def pct(ys: list[float], p: float) -> float:
    ys = sorted(ys)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def main() -> None:
    db = SessionLocal()
    try:
        rows = (
            db.query(SearchPhoto)
            .filter(SearchPhoto.id > 800)
            .filter(
                (SearchPhoto.hist_wine_id.isnot(None))
                | (SearchPhoto.manual_wines_id.isnot(None))
            )
            .order_by(SearchPhoto.id.asc())
            .all()
        )
        wine_ids: set[int] = set()
        for r in rows:
            if r.manual_wines_id:
                wine_ids.add(int(r.manual_wines_id))
            if r.hist_wine_id:
                wine_ids.add(int(r.hist_wine_id))
        wines = {
            int(w.id): w
            for w in db.query(Wine).filter(Wine.id.in_(wine_ids)).all()
        } if wine_ids else {}

        out: list[tuple] = []
        skipped_fp = 0
        for r in rows:
            gt = (
                int(r.manual_wines_id)
                if r.manual_wines_id
                else (int(r.hist_wine_id) if r.hist_wine_id else None)
            )
            if gt is None:
                continue
            w = wines.get(gt)
            if w is None:
                continue
            status = r.status if isinstance(r.status, dict) else {}
            q = collect_query_ocr_text(status)
            ql, cl = n_lines(q), n_lines(str(w.label or ""))
            if ql <= cl:
                continue
            if int(r.hist_fp or 0) != 0:
                skipped_fp += 1
                continue
            ratio = (ql / cl) if cl > 0 else float("inf")
            src = "manual" if r.manual_wines_id else "matched"
            out.append((int(r.id), gt, src, ql, cl, ql - cl, ratio))
    finally:
        db.close()

    print(f"query_more_no_fp={len(out)} skipped_fp={skipped_fp}")
    n_m = sum(1 for x in out if x[2] == "matched")
    n_man = sum(1 for x in out if x[2] == "manual")
    print(f"matched={n_m} manual={n_man}")
    finite = [x[6] for x in out if not math.isinf(x[6])]
    ex = [x[5] for x in out]
    print(
        f"ratio median={pct(finite, 0.5):.2f} p90={pct(finite, 0.9):.2f} "
        f"max={max(finite):.2f} mean={sum(finite)/len(finite):.2f}"
    )
    print(
        f">1.5x={sum(1 for v in finite if v > 1.5)} "
        f">2x={sum(1 for v in finite if v > 2)} "
        f">3x={sum(1 for v in finite if v > 3)}"
    )
    se = sorted(ex)
    print(
        f"excess median={se[len(se)//2]} p90={se[int((len(se)-1)*0.9)]} "
        f"max={max(ex)} mean={sum(ex)/len(ex):.2f}"
    )
    be: Counter[str] = Counter()
    for e in ex:
        if e == 1:
            be["+1"] += 1
        elif e == 2:
            be["+2"] += 1
        elif e == 3:
            be["+3"] += 1
        elif e <= 5:
            be["+4..5"] += 1
        elif e <= 8:
            be["+6..8"] += 1
        else:
            be["+9+"] += 1
    print("excess_buckets", dict(be))
    print("LIST")
    for id_, wid, src, ql, cl, exc, rr in out:
        print(f"{id_}\t{wid}\t{src}\t{ql}\t{cl}\t+{exc}\t{rr:.2f}")
    print("GT3")
    for id_, wid, src, ql, cl, exc, rr in out:
        if rr > 3:
            print(f"{id_}\t{ql}/{cl}\t{rr:.2f}")


if __name__ == "__main__":
    main()
