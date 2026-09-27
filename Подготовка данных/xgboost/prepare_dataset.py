"""Подготовка датасета OCR Text Matcher из истории поиска (`search_photos`).

Один пайплайн для обоих подходов (ТЗ п.26):

    search_photos.status  →  raw/scans.jsonl + raw/wines.jsonl
                          →  processed/{train,validation,test}.jsonl   (пары, Cross-Encoder)
                          →  features/{train,validation,test}.parquet  (признаки, XGBoost)

Источник OCR запроса (только сырые строки, без LLM-категоризации):
    variants.google_vision.lines → gemini.lines → openai.lines  (приоритет настраивается)

Кандидаты: только SigLIP2 top-N (`steps.candidates.siglip2.items` → cosine_similarity).

Разметка (по ТЗ пользователя):
    eval.false_positive=1  → найденное вино = label 0 (hard negative), остальные кандидаты
                             пропускаются (истинное вино неизвестно)
    eval.false_negative=1  → истинное вино неизвестно → скан пропускается
    без флагов, matched_wine.source ∈ --sources → найденное вино = 1, остальные SigLIP-кандидаты = 0
    без matched_wine (или source вне --sources) → скан пропускается

Split: по группам (id найденного вина), а не по строкам — иначе утечка (ТЗ п.7).

Запуск:
    cd C:\\dev\\Vino2026\\models\\ocr_matcher
    python prepare_dataset.py
    python prepare_dataset.py --sources final_score,final_score2,siglip2   # добавить cosine-fallback как positives
    python prepare_dataset.py --all-ocr-engines                            # пары для каждого доступного OCR-движка
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    WineRef,
    group_features,
    lines_to_text,
    ocr_cross_encoder_text,
)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE  # outputs under this folder
REPO_ROOT = MODELS_DIR.parent
DEFAULT_ENV = HERE.parents[1] / "backend" / ".env"  # vino-svoe/backend/.env
DEFAULT_DATA_DIR = MODELS_DIR / "data"

TRUSTED_SOURCES_DEFAULT = "final_score,final_score2"
OCR_ENGINES_DEFAULT = "google_vision,gemini,openai"


# ----------------------------------------------------------------------------
# Модели данных
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    wine_id: int
    cosine: float | None
    rank: int


@dataclass
class ScanRecord:
    scan_id: int
    sha256: str
    created_at: str | None
    algorithm_version: str | None
    ocr_engine: str
    ocr_lines: list[str]
    ocr_text: str
    ocr_variants: dict[str, str] = field(default_factory=dict)  # engine -> text (все доступные)
    matched_wine_id: int | None = None
    matched_source: str | None = None
    false_positive: int = 0
    false_negative: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    label_policy: str = ""  # tp_tn | fp_hard
    group_id: int | None = None
    split: str | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [asdict(c) for c in self.candidates]
        return d


# ----------------------------------------------------------------------------
# Извлечение из status JSON
# ----------------------------------------------------------------------------
def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def extract_ocr_variant(variant: dict[str, Any] | None) -> list[str]:
    """Сырые строки OCR одного движка: lines[].text → text.splitlines()."""
    if not isinstance(variant, dict) or not variant.get("ok"):
        return []
    lines: list[str] = []
    raw_lines = variant.get("lines")
    if isinstance(raw_lines, list):
        for ln in raw_lines:
            if isinstance(ln, dict):
                t = str(ln.get("text") or "").strip()
            else:
                t = str(ln or "").strip()
            if t:
                lines.append(t)
    if not lines:
        txt = str(variant.get("text") or "")
        lines = [x.strip() for x in txt.splitlines() if x.strip()]
    return lines


def extract_all_ocr(status: dict[str, Any], engines: list[str]) -> dict[str, list[str]]:
    variants = ((status.get("steps") or {}).get("ocr") or {}).get("variants") or {}
    out: dict[str, list[str]] = {}
    for eng in engines:
        lines = extract_ocr_variant(variants.get(eng))
        if lines:
            out[eng] = lines
    return out


def extract_siglip_candidates(status: dict[str, Any]) -> list[Candidate]:
    """Только SigLIP2: steps.candidates.siglip2.items (cosine чистый), fallback — ids + candidates_cosine."""
    cands: list[Candidate] = []
    step = ((status.get("steps") or {}).get("candidates") or {}).get("siglip2") or {}
    items = step.get("items") or []
    seen: set[int] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        wid = _as_int(it.get("id"))
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        cands.append(Candidate(wine_id=wid, cosine=_as_float(it.get("cosine_similarity")), rank=len(cands) + 1))
    if cands:
        return cands
    cos_map = status.get("candidates_cosine") or {}
    for wid_raw in status.get("candidates_siglip2_ids") or []:
        wid = _as_int(wid_raw)
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        cos = _as_float(cos_map.get(str(wid)))
        cands.append(Candidate(wine_id=wid, cosine=cos, rank=len(cands) + 1))
    return cands


def extract_eval_flags(status: dict[str, Any]) -> tuple[int, int]:
    raw = status.get("eval")
    fp = fn = 0
    if isinstance(raw, dict):
        fp = 1 if (_as_int(raw.get("false_positive")) or 0) else 0
        fn = 1 if (_as_int(raw.get("false_negative")) or 0) else 0
    if fp and fn:
        fn = 0
    return fp, fn


def build_scan_record(
    scan_id: int,
    sha256: str,
    created_at: datetime | None,
    status: dict[str, Any],
    *,
    engines: list[str],
    trusted_sources: set[str],
    fn_policy: str,
    fp_policy: str,
    min_ocr_chars: int,
) -> tuple[ScanRecord | None, str]:
    """Возвращает (record, reason). record=None → скан пропущен, reason — почему."""
    all_ocr = extract_all_ocr(status, engines)
    if not all_ocr:
        return None, "no_ocr"
    engine = next(e for e in engines if e in all_ocr)
    ocr_lines = all_ocr[engine]
    ocr_text = lines_to_text(ocr_lines)
    if len(ocr_text) < min_ocr_chars:
        return None, "ocr_too_short"

    cands = extract_siglip_candidates(status)
    if not cands:
        return None, "no_siglip_candidates"

    matched_id = _as_int(status.get("matched_wine_id"))
    matched = status.get("matched_wine") or {}
    source = str(matched.get("source") or "").strip().lower() or None
    fp, fn = extract_eval_flags(status)

    if fn:
        if fn_policy == "skip":
            return None, "false_negative_skipped"
        # fn_policy == positive: считаем показанное вино верным
        if matched_id is None:
            return None, "false_negative_no_matched"
        policy = "tp_tn"
    elif fp:
        if matched_id is None:
            return None, "false_positive_no_matched"
        if fp_policy == "skip":
            return None, "false_positive_skipped"
        policy = "fp_hard"
    else:
        if matched_id is None:
            return None, "no_matched_wine"
        if "all" not in trusted_sources and (source or "") not in trusted_sources:
            return None, f"source_not_trusted:{source}"
        policy = "tp_tn"

    if matched_id not in {c.wine_id for c in cands}:
        return None, "matched_not_in_siglip_candidates"

    rec = ScanRecord(
        scan_id=scan_id,
        sha256=sha256,
        created_at=created_at.isoformat() if created_at else None,
        algorithm_version=status.get("algorithm_version"),
        ocr_engine=engine,
        ocr_lines=ocr_lines,
        ocr_text=ocr_text,
        ocr_variants={e: lines_to_text(v) for e, v in all_ocr.items()},
        matched_wine_id=matched_id,
        matched_source=source,
        false_positive=fp,
        false_negative=fn,
        candidates=cands,
        label_policy=policy,
        group_id=matched_id,
    )
    return rec, "ok"


# ----------------------------------------------------------------------------
# БД
# ----------------------------------------------------------------------------
def resolve_database_url(cli_url: str | None, env_file: Path) -> str:
    if cli_url:
        return cli_url
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        if os.getenv("DATABASE_URL"):
            return os.environ["DATABASE_URL"]
    raise SystemExit(
        f"DATABASE_URL не найден: передайте --database-url или укажите его в {env_file}"
    )


def load_scans(engine, limit: int | None) -> list[tuple[int, str, datetime | None, dict]]:
    from sqlalchemy import text

    sql = "select id, sha256, created_at, status from search_photos order by id"
    if limit:
        sql += f" limit {int(limit)}"
    with engine.connect() as c:
        rows = c.execute(text(sql)).fetchall()
    return [(int(r[0]), str(r[1] or ""), r[2], r[3] or {}) for r in rows]


def load_wines(engine, wine_ids: set[int]) -> dict[int, WineRef]:
    from sqlalchemy import text

    if not wine_ids:
        return {}
    out: dict[int, WineRef] = {}
    ids = sorted(wine_ids)

    def _year(value: object) -> int | None:
        try:
            y = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return y if 1950 <= y <= 2100 else None

    with engine.connect() as c:
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            rows = c.execute(
                text(
                    "select id, label, name, winery, category, color, grape_variety, "
                    "region, slug, label_ocr "
                    "from wines where id = any(:ids)"
                ),
                {"ids": chunk},
            ).fetchall()
            for r in rows:
                lo = r[9] if isinstance(r[9], dict) else {}
                out[int(r[0])] = WineRef(
                    wine_id=int(r[0]),
                    label=r[1],
                    name=r[2],
                    winery=r[3],
                    category=r[4],
                    color=r[5],
                    grape=r[6],
                    region=r[7],
                    label_color=(str(lo["color"]) if lo.get("color") not in (None, "") else None),
                    label_type=(str(lo["type"]) if lo.get("type") not in (None, "") else None),
                    vintage_year=_year(lo.get("vintage_year")),
                    foundation_year=_year(lo.get("foundation_year")),
                )
                setattr(out[int(r[0])], "slug", r[8])
    return out


# ----------------------------------------------------------------------------
# Дедупликация и split
# ----------------------------------------------------------------------------
def dedup_by_sha256(records: list[ScanRecord]) -> tuple[list[ScanRecord], int]:
    """Одно фото могли сканировать несколько раз: оставляем размеченный (FP/FN) или самый новый."""
    by_sha: dict[str, list[ScanRecord]] = defaultdict(list)
    for r in records:
        by_sha[r.sha256 or f"noid:{r.scan_id}"].append(r)
    kept: list[ScanRecord] = []
    dropped = 0
    for _, group in by_sha.items():
        group.sort(key=lambda r: (int(bool(r.false_positive or r.false_negative)), r.scan_id))
        kept.append(group[-1])
        dropped += len(group) - 1
    kept.sort(key=lambda r: r.scan_id)
    return kept, dropped


def split_by_group(
    records: list[ScanRecord], train: float, val: float, test: float, seed: int
) -> None:
    """Делим группы (group_id = id найденного вина) так, чтобы доли по сканам были ≈ train/val/test."""
    total = train + val + test
    train, val, test = train / total, val / total, test / total
    groups: dict[int, list[ScanRecord]] = defaultdict(list)
    for r in records:
        groups[int(r.group_id or -1)].append(r)
    gids = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(gids)
    n_total = len(records)
    acc = 0
    for gid in gids:
        frac = acc / n_total if n_total else 0.0
        if frac < train:
            sp = "train"
        elif frac < train + val:
            sp = "validation"
        else:
            sp = "test"
        for r in groups[gid]:
            r.split = sp
        acc += len(groups[gid])


# ----------------------------------------------------------------------------
# Пары и признаки
# ----------------------------------------------------------------------------
def build_pairs_for_scan(
    rec: ScanRecord,
    wines: dict[int, WineRef],
    *,
    ocr_engine: str,
    ocr_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    refs: list[WineRef] = []
    cands: list[Candidate] = []
    for c in rec.candidates:
        ref = wines.get(c.wine_id)
        if ref is None:
            continue
        refs.append(ref)
        cands.append(c)
    if not refs:
        return [], []
    feats = group_features(ocr_text, refs, [c.cosine for c in cands])

    pairs: list[dict[str, Any]] = []
    feat_rows: list[dict[str, float]] = []
    text1 = ocr_cross_encoder_text(ocr_text)
    for ref, c, f in zip(refs, cands, feats):
        is_matched = c.wine_id == rec.matched_wine_id
        if rec.label_policy == "fp_hard":
            if not is_matched:
                continue  # истинное вино неизвестно — остальные не размечаем
            label, kind = 0, "fp_hard"
        else:
            label = 1 if is_matched else 0
            kind = "tp" if is_matched else "tn"
        pair = {
            "scan_id": rec.scan_id,
            "wine_id": c.wine_id,
            "group_id": rec.group_id,
            "split": rec.split,
            "label": label,
            "label_kind": kind,
            "ocr_engine": ocr_engine,
            "ocr_text": ocr_text,
            "text1": text1,
            "text2": ref.cross_encoder_text(),
            "wine_label": ref.label,
            "wine_name": ref.name,
            "wine_winery": ref.winery,
            "wine_category": ref.category,
            "wine_grape": ref.grape,
            "wine_region": ref.region,
            "siglip_cosine": c.cosine,
            "siglip_rank": c.rank,
            "n_candidates": len(refs),
            "matched_source": rec.matched_source,
            "algorithm_version": rec.algorithm_version,
        }
        pairs.append(pair)
        row: dict[str, Any] = {
            "scan_id": rec.scan_id,
            "wine_id": c.wine_id,
            "group_id": rec.group_id,
            "split": rec.split,
            "label": label,
            "label_kind": kind,
            "ocr_engine": ocr_engine,
        }
        row.update(f)
        feat_rows.append(row)
    return pairs, feat_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=None, help="postgresql+psycopg2://… (иначе env / backend .env)")
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"куда писать данные (default {DEFAULT_DATA_DIR})")
    p.add_argument("--sources", default=TRUSTED_SOURCES_DEFAULT,
                   help="matched_wine.source, которым доверяем как true positive; 'all' — любые")
    p.add_argument("--ocr-engines", default=OCR_ENGINES_DEFAULT,
                   help="приоритет OCR-движков (сырые строки), через запятую")
    p.add_argument("--all-ocr-engines", action="store_true",
                   help="создавать пары для каждого доступного OCR-движка скана (аугментация), а не только для первого по приоритету")
    p.add_argument("--fp-policy", choices=["hard_negative", "skip"], default="hard_negative")
    p.add_argument("--fn-policy", choices=["skip", "positive"], default="skip")
    p.add_argument("--no-dedup", action="store_true", help="не схлопывать повторные сканы одного фото (sha256)")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-ocr-chars", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="ограничить число сканов (отладка)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from sqlalchemy import create_engine

    db_url = resolve_database_url(args.database_url, args.env_file)
    engine = create_engine(db_url)
    engines = [e.strip() for e in args.ocr_engines.split(",") if e.strip()]
    trusted = {s.strip().lower() for s in args.sources.split(",") if s.strip()}

    print(f"[1/6] Загрузка search_photos …")
    rows = load_scans(engine, args.limit)
    print(f"      сканов в БД: {len(rows)}")

    print(f"[2/6] Разбор status → ScanRecord (OCR: {engines}, sources: {sorted(trusted)})")
    records: list[ScanRecord] = []
    reasons: Counter = Counter()
    for scan_id, sha, created, status in rows:
        rec, reason = build_scan_record(
            scan_id, sha, created, status,
            engines=engines, trusted_sources=trusted,
            fn_policy=args.fn_policy, fp_policy=args.fp_policy,
            min_ocr_chars=args.min_ocr_chars,
        )
        reasons[reason] += 1
        if rec is not None:
            records.append(rec)
    print(f"      принято сканов: {len(records)}")
    for k, v in reasons.most_common():
        print(f"        {k:<40} {v}")

    dropped_dup = 0
    if not args.no_dedup:
        records, dropped_dup = dedup_by_sha256(records)
        print(f"[3/6] Дедупликация по sha256: убрано {dropped_dup}, осталось {len(records)}")
    else:
        print("[3/6] Дедупликация отключена")

    wine_ids: set[int] = set()
    for r in records:
        wine_ids.update(c.wine_id for c in r.candidates)
        if r.matched_wine_id is not None:
            wine_ids.add(r.matched_wine_id)
    wines = load_wines(engine, wine_ids)
    print(f"[4/6] Загружено вин (эталонов): {len(wines)} из {len(wine_ids)} нужных")

    split_by_group(records, args.train, args.val, args.test, args.seed)
    split_counts = Counter(r.split for r in records)
    groups_per_split = {s: len({r.group_id for r in records if r.split == s}) for s in ("train", "validation", "test")}
    print(f"[5/6] Split по группам (id вина): сканы {dict(split_counts)}, групп {groups_per_split}")

    # --- пары + признаки ---
    pairs_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feats_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        variants = rec.ocr_variants if args.all_ocr_engines else {rec.ocr_engine: rec.ocr_text}
        for eng, txt in variants.items():
            pairs, feats = build_pairs_for_scan(rec, wines, ocr_engine=eng, ocr_text=txt)
            pairs_by_split[rec.split or "train"].extend(pairs)
            feats_by_split[rec.split or "train"].extend(feats)

    out = args.out_dir
    raw_dir, proc_dir, feat_dir = out / "raw", out / "processed", out / "features"
    for d in (raw_dir, proc_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_jsonl(raw_dir / "scans.jsonl", [r.to_json() for r in records])
    write_jsonl(
        raw_dir / "wines.jsonl",
        [
            {
                "wine_id": w.wine_id, "slug": getattr(w, "slug", None), "label": w.label, "name": w.name,
                "winery": w.winery, "category": w.category, "color": w.color, "grape": w.grape,
                "region": w.region, "cross_encoder_text": w.cross_encoder_text(),
            }
            for w in sorted(wines.values(), key=lambda x: x.wine_id)
        ],
    )

    stats: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "database_url"},
        "scans_in_db": len(rows),
        "scans_accepted": len(records),
        "scans_dropped_duplicates": dropped_dup,
        "skip_reasons": dict(reasons),
        "splits": {},
    }
    for sp in ("train", "validation", "test"):
        pairs = pairs_by_split.get(sp, [])
        feats = feats_by_split.get(sp, [])
        write_jsonl(proc_dir / f"{sp}.jsonl", pairs)
        df = pd.DataFrame(feats)
        if not df.empty:
            id_cols = ["scan_id", "wine_id", "group_id", "split", "label", "label_kind", "ocr_engine"]
            df = df[id_cols + FEATURE_NAMES]
        df.to_parquet(feat_dir / f"{sp}.parquet", index=False)
        kinds = Counter(p["label_kind"] for p in pairs)
        stats["splits"][sp] = {
            "scans": int(sum(1 for r in records if r.split == sp)),
            "groups": groups_per_split.get(sp, 0),
            "pairs": len(pairs),
            "positives": int(sum(1 for p in pairs if p["label"] == 1)),
            "negatives": int(sum(1 for p in pairs if p["label"] == 0)),
            "label_kinds": dict(kinds),
            "ocr_engines": dict(Counter(p["ocr_engine"] for p in pairs)),
        }
    (out / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[6/6] Записано в {out}")
    for sp, s in stats["splits"].items():
        print(f"      {sp:<11} scans={s['scans']:<4} groups={s['groups']:<4} pairs={s['pairs']:<6} "
              f"pos={s['positives']:<4} neg={s['negatives']:<6} kinds={s['label_kinds']}")
    print(f"      признаков: {len(FEATURE_NAMES)} (feature_version {FEATURE_VERSION})")
    print(f"      raw/scans.jsonl, raw/wines.jsonl, processed/*.jsonl, features/*.parquet, dataset_stats.json")


if __name__ == "__main__":
    main()
