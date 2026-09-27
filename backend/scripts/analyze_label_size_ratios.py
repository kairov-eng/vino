"""Analyze OCR vs catalog label size ratios for search_photos id>800 with a found/manual wine."""
from __future__ import annotations

import math
import statistics
from collections import Counter

from app.database import SessionLocal
from app.db.models import SearchPhoto, Wine
from app.pipeline.xgb_match import collect_query_ocr_text
from app.pipeline.xgb_text_features import normalize_text, tokens


def n_lines(text: str) -> int:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return len(lines)


def n_chars(text: str) -> int:
    # длина без учёта leading/trailing, но с внутренними пробелами/переносами
    return len(str(text or "").strip())


def n_words(text: str) -> int:
    return len(tokens(normalize_text(text)))


def ratio(a: int, b: int) -> float | None:
    """max/min; None если обе стороны 0."""
    if a <= 0 and b <= 0:
        return None
    # одна сторона 0 → бесконечный разрыв; для статистики берём большой sentinel
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a, b) / min(a, b)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def bucket(r: float | None) -> str:
    if r is None:
        return "both_empty"
    if math.isinf(r):
        return "one_empty"
    if r <= 1.25:
        return "<=1.25x"
    if r <= 1.5:
        return "<=1.5x"
    if r <= 2.0:
        return "<=2x"
    if r <= 3.0:
        return "<=3x"
    if r <= 5.0:
        return "<=5x"
    return ">5x"


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
        wine_ids = set()
        for r in rows:
            if r.manual_wines_id:
                wine_ids.add(int(r.manual_wines_id))
            if r.hist_wine_id:
                wine_ids.add(int(r.hist_wine_id))
        wines = {
            int(w.id): w
            for w in db.query(Wine).filter(Wine.id.in_(wine_ids)).all()
        } if wine_ids else {}

        samples: list[dict] = []
        skipped_no_ocr = 0
        skipped_no_wine = 0
        for r in rows:
            # ground truth: manual override wins
            gt_id = int(r.manual_wines_id) if r.manual_wines_id else (
                int(r.hist_wine_id) if r.hist_wine_id else None
            )
            if gt_id is None:
                skipped_no_wine += 1
                continue
            w = wines.get(gt_id)
            if w is None:
                skipped_no_wine += 1
                continue
            status = r.status if isinstance(r.status, dict) else {}
            q = collect_query_ocr_text(status)
            c = str(w.label or "")
            if not str(q or "").strip() and not str(c or "").strip():
                skipped_no_ocr += 1
                continue
            ql, cl = n_lines(q), n_lines(c)
            qc, cc = n_chars(q), n_chars(c)
            qw, cw = n_words(q), n_words(c)
            src = "manual" if r.manual_wines_id else "matched"
            samples.append(
                {
                    "id": int(r.id),
                    "wine_id": gt_id,
                    "src": src,
                    "q_lines": ql,
                    "c_lines": cl,
                    "r_lines": ratio(ql, cl),
                    "q_chars": qc,
                    "c_chars": cc,
                    "r_chars": ratio(qc, cc),
                    "q_words": qw,
                    "c_words": cw,
                    "r_words": ratio(qw, cw),
                }
            )
    finally:
        db.close()

    print(f"searches id>800 with matched|manual: {len(rows)}")
    print(f"usable samples: {len(samples)}  skipped_no_wine={skipped_no_wine} skipped_empty_both={skipped_no_ocr}")
    n_manual = sum(1 for s in samples if s["src"] == "manual")
    n_matched = sum(1 for s in samples if s["src"] == "matched")
    print(f"  src: matched={n_matched} manual={n_manual}")

    for key, title in (
        ("r_lines", "lines"),
        ("r_chars", "chars"),
        ("r_words", "words"),
    ):
        vals = [s[key] for s in samples if s[key] is not None]
        finite = [v for v in vals if not math.isinf(v)]
        inf_n = sum(1 for v in vals if math.isinf(v))
        print()
        print(f"=== {title} ===")
        print(f"n={len(vals)}  finite={len(finite)}  one_side_empty={inf_n}")
        if finite:
            print(
                "  min={:.2f}  p25={:.2f}  median={:.2f}  p75={:.2f}  p90={:.2f}  p95={:.2f}  max={:.2f}  mean={:.2f}".format(
                    min(finite),
                    pct(finite, 0.25),
                    pct(finite, 0.50),
                    pct(finite, 0.75),
                    pct(finite, 0.90),
                    pct(finite, 0.95),
                    max(finite),
                    statistics.mean(finite),
                )
            )
            over2 = sum(1 for v in finite if v > 2.0)
            over3 = sum(1 for v in finite if v > 3.0)
            print(
                f"  ratio > 2x: {over2}/{len(finite)} ({100*over2/len(finite):.1f}%)"
                f"  |  >3x: {over3}/{len(finite)} ({100*over3/len(finite):.1f}%)"
            )
        buckets = Counter(bucket(s[key]) for s in samples)
        order = ["<=1.25x", "<=1.5x", "<=2x", "<=3x", "<=5x", ">5x", "one_empty", "both_empty"]
        for b in order:
            if buckets.get(b):
                print(f"  {b}: {buckets[b]}")

    # top offenders for lines
    print()
    print("=== top-15 by line ratio (finite) ===")
    ranked = sorted(
        [s for s in samples if s["r_lines"] is not None and not math.isinf(s["r_lines"])],
        key=lambda s: s["r_lines"],
        reverse=True,
    )[:15]
    for s in ranked:
        rc = s["r_chars"]
        rw = s["r_words"]
        rc_s = "inf" if rc == float("inf") else f"{rc:.2f}"
        rw_s = "inf" if rw == float("inf") else f"{rw:.2f}"
        print(
            f"  #{s['id']} wine={s['wine_id']} {s['src']}: "
            f"lines {s['q_lines']}/{s['c_lines']} = {s['r_lines']:.2f}x  "
            f"chars {s['q_chars']}/{s['c_chars']} = {rc_s}x  "
            f"words {s['q_words']}/{s['c_words']} = {rw_s}x"
        )

    print()
    print("=== false rejects if lines >2x rule ===")
    finite_lines = [s for s in samples if s["r_lines"] is not None and not math.isinf(s["r_lines"])]
    hit = [s for s in finite_lines if s["r_lines"] > 2.0]
    print(f"  false rejects among GT: {len(hit)}/{len(finite_lines)} ({100*len(hit)/max(1,len(finite_lines)):.1f}%)")
    for s in hit[:40]:
        print(
            f"  #{s['id']} wine={s['wine_id']} {s['src']}: "
            f"{s['q_lines']} vs {s['c_lines']} lines ({s['r_lines']:.2f}x)"
        )

    # JSON dump for canvas
    import json
    from pathlib import Path

    def _ser(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isinf(v):
            return None
        return v

    summary = {}
    for key, title in (
        ("r_lines", "lines"),
        ("r_chars", "chars"),
        ("r_words", "words"),
    ):
        finite = [s[key] for s in samples if s[key] is not None and not math.isinf(s[key])]
        buckets = Counter(bucket(s[key]) for s in samples)
        summary[title] = {
            "n_finite": len(finite),
            "one_empty": sum(1 for s in samples if s[key] is not None and math.isinf(s[key])),
            "min": round(min(finite), 3) if finite else None,
            "p25": round(pct(finite, 0.25), 3) if finite else None,
            "median": round(pct(finite, 0.5), 3) if finite else None,
            "p75": round(pct(finite, 0.75), 3) if finite else None,
            "p90": round(pct(finite, 0.9), 3) if finite else None,
            "p95": round(pct(finite, 0.95), 3) if finite else None,
            "max": round(max(finite), 3) if finite else None,
            "mean": round(statistics.mean(finite), 3) if finite else None,
            "gt2": sum(1 for v in finite if v > 2.0),
            "gt2_pct": round(100 * sum(1 for v in finite if v > 2.0) / len(finite), 1) if finite else 0,
            "gt3": sum(1 for v in finite if v > 3.0),
            "buckets": dict(buckets),
        }

    out = {
        "n_searches": len(rows),
        "n_samples": len(samples),
        "n_matched": n_matched,
        "n_manual": n_manual,
        "summary": summary,
        "line_gt2": [
            {
                "id": s["id"],
                "wine_id": s["wine_id"],
                "src": s["src"],
                "q_lines": s["q_lines"],
                "c_lines": s["c_lines"],
                "r_lines": round(s["r_lines"], 3),
                "r_chars": _ser(round(s["r_chars"], 3) if s["r_chars"] != float("inf") else None),
                "r_words": _ser(round(s["r_words"], 3) if s["r_words"] != float("inf") else None),
            }
            for s in hit
        ],
        "top_lines": [
            {
                "id": s["id"],
                "wine_id": s["wine_id"],
                "src": s["src"],
                "q_lines": s["q_lines"],
                "c_lines": s["c_lines"],
                "r_lines": round(s["r_lines"], 3),
                "r_chars": round(s["r_chars"], 3) if s["r_chars"] != float("inf") else None,
                "r_words": round(s["r_words"], 3) if s["r_words"] != float("inf") else None,
            }
            for s in ranked
        ],
    }
    out_path = Path(__file__).resolve().parent / "analyze_label_size_ratios.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
