"""Re-sync hist_* denorm columns from status (official match only)."""
from __future__ import annotations

from app.database import SessionLocal
from app.db.models import SearchPhoto
from app.scan_history_denorm import (
    history_resolve_matched_wine_id,
    sync_search_photo_history_columns,
)


def main() -> None:
    db = SessionLocal()
    try:
        rows = db.query(SearchPhoto).order_by(SearchPhoto.id.asc()).all()
        cleared = 0
        kept = 0
        for row in rows:
            st = dict(row.status or {})
            before = row.hist_wine_id
            sync_search_photo_history_columns(row, st)
            after = row.hist_wine_id
            if before is not None and after is None:
                cleared += 1
            elif after is not None:
                kept += 1
        db.commit()
        # spot-check 2636
        r = db.get(SearchPhoto, 2636)
        print(
            f"done rows={len(rows)} cleared_false_matches={cleared} "
            f"with_match={kept}"
        )
        if r is not None:
            print(
                f"scan2636 hist_wine_id={r.hist_wine_id} "
                f"resolved={history_resolve_matched_wine_id(dict(r.status or {}))}"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
