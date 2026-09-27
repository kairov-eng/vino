"""Successful scans where query OCR has MORE lines than catalog label."""
from __future__ import annotations

import math
from collections import Counter

from app.database import SessionLocal
from app.db.models import SearchPhoto, Wine
from app.pipeline.xgb_match import collect_query_ocr_text


def n_lines(text: str) -> int:
    return len([ln.strip() for ln in str(text or "").splitlines() if ln.strip()])


def pct(ys: list[float], p: float) -> float:
    if not ys:
        return float("nan")
    ys = sorted(ys)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def bucket_r(r: float) -> str:
    if math.isinf(r):
        return "catalog_0_lines"
    if r <= 1.25:
        return "<=1.25x"
    if r <= 1.5:
        return "<=1.5x"
    if r <= 2:
        return "<=2x"
    if r <= 3:
        return "<=3x"
    if r <= 5:
        return "<=5x"
    return ">5x"


def bucket_e(e: int) -> str:
    if e == 1:
        return "+1"
    if e == 2:
        return "+2"
    if e == 3:
        return "+3"
    if e <= 5:
        return "+4..5"
    if e <= 8:
        return "+6..8"
    return "+9+"


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
            c = str(w.label or "")
            ql, cl = n_lines(q), n_lines(c)
            if ql <= cl:
                continue
            ratio = (ql / cl) if cl > 0 else float("inf")
            excess = ql - cl
            src = "manual" if r.manual_wines_id else "matched"
            out.append((int(r.id), gt, src, ql, cl, excess, ratio))
    finally:
        db.close()

    print(f"total_gt_searches={len(rows)}")
    print(f"query_more_lines={len(out)}")
    print(
        "matched=%d manual=%d"
        % (
            sum(1 for x in out if x[2] == "matched"),
            sum(1 for x in out if x[2] == "manual"),
        )
    )

    br = Counter(bucket_r(x[6]) for x in out)
    be = Counter(bucket_e(x[5]) for x in out)
    print("RATIO_BUCKETS (OCR/catalog)")
    for k in [
        "<=1.25x",
        "<=1.5x",
        "<=2x",
        "<=3x",
        "<=5x",
        ">5x",
        "catalog_0_lines",
    ]:
        if br.get(k):
            print(f"  {k}: {br[k]}")
    print("EXCESS_BUCKETS (OCR - catalog lines)")
    for k in ["+1", "+2", "+3", "+4..5", "+6..8", "+9+"]:
        if be.get(k):
            print(f"  {k}: {be[k]}")

    finite = [x[6] for x in out if not math.isinf(x[6])]
    excesses = [x[5] for x in out]
    if finite:
        print(
            "ratio: min=%.2f median=%.2f p75=%.2f p90=%.2f max=%.2f mean=%.2f"
            % (
                min(finite),
                pct(finite, 0.5),
                pct(finite, 0.75),
                pct(finite, 0.9),
                max(finite),
                sum(finite) / len(finite),
            )
        )
        print(
            "  >1.5x: %d  >2x: %d  >3x: %d"
            % (
                sum(1 for v in finite if v > 1.5),
                sum(1 for v in finite if v > 2),
                sum(1 for v in finite if v > 3),
            )
        )
    if excesses:
        se = sorted(excesses)
        print(
            "excess: min=%d median=%d p90=%d max=%d mean=%.2f"
            % (
                min(excesses),
                se[len(se) // 2],
                se[int((len(se) - 1) * 0.9)],
                max(excesses),
                sum(excesses) / len(excesses),
            )
        )

    print("LIST")
    for id_, wid, src, ql, cl, ex, rr in out:
        rs = "inf" if math.isinf(rr) else f"{rr:.2f}"
        print(f"{id_}\t{wid}\t{src}\t{ql}\t{cl}\t+{ex}\t{rs}x")


if __name__ == "__main__":
    main()
