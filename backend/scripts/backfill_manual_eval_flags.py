"""Backfill hist_fp/hist_fn + status.eval from manual_wines_id vs matched wine."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select, text
from sqlalchemy.orm.attributes import flag_modified

from app.database import SessionLocal
from app.db.models import SearchPhoto
from app.scan_history_denorm import (
    apply_status_eval_flags,
    eval_flags_for_manual_match,
    history_resolve_matched_wine_id,
    sync_search_photo_history_columns,
)


def main() -> None:
    db = SessionLocal()
    try:
        rows = list(
            db.scalars(
                select(SearchPhoto).where(SearchPhoto.manual_wines_id.is_not(None))
            ).all()
        )
        n_fp = n_fn = n_tp = n_skip = 0
        for row in rows:
            status = dict(row.status or {})
            matched = row.hist_wine_id
            if matched is None:
                matched = history_resolve_matched_wine_id(status)
            flags = eval_flags_for_manual_match(
                manual_id=row.manual_wines_id,
                matched_id=matched,
            )
            if flags is None:
                n_skip += 1
                continue
            fp, fn = flags
            if fp == int(row.hist_fp or 0) and fn == int(row.hist_fn or 0):
                # also ensure status.eval matches
                ev = status.get("eval") if isinstance(status.get("eval"), dict) else {}
                if int(ev.get("false_positive") or 0) == fp and int(
                    ev.get("false_negative") or 0
                ) == fn:
                    n_skip += 1
                    continue
            status = apply_status_eval_flags(status, fp, fn)
            row.status = status
            flag_modified(row, "status")
            sync_search_photo_history_columns(row, status)
            db.add(row)
            if fp:
                n_fp += 1
            elif fn:
                n_fn += 1
            else:
                n_tp += 1
        db.commit()
        print(
            json.dumps(
                {
                    "manual_total": len(rows),
                    "updated_fp": n_fp,
                    "updated_fn": n_fn,
                    "updated_tp_clear": n_tp,
                    "unchanged": n_skip,
                },
                ensure_ascii=False,
            )
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
