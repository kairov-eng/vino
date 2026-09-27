# =============================================================================
# Подготовка датасета для fine-tune SigLIP2 (contrastive / image-text или image folders)
# =============================================================================
#
# Исходные данные:
#   - PostgreSQL `wines` с crop IN (1,2)
#   - PNG этикеток: MEDIA/crop/<stem>_crop.png  (как пишет label_detect/2.crop_wines.py)
#
# Выход (zip-ready):
#   out_dir/
#     images/train/<wine_id>/*.jpg
#     images/val/<wine_id>/*.jpg
#     meta.jsonl          # wine_id, path, slug, name, winery
#     README.txt
#
# Пример:
#   cd research/label_detect   # или training/siglip2 после копирования
#   python prepare_siglip2_finetune_dataset.py ^
#     --uploads C:\dev\Vino2026\media\uploads ^
#     --out C:\dev\Vino2026\vino-svoe\training\siglip2\dataset ^
#     --limit 0
#
# Затем: zip dataset → Google Drive → Colab (colab_train_siglip2.py)
# =============================================================================

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent


def _default_uploads() -> Path:
    # monorepo layout: Vino2026/media/uploads
    cand = HERE.parents[2] / "media" / "uploads"
    if cand.is_dir():
        return cand
    return HERE.parents[2] / "uploads"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default="", help="иначе backend/.env DATABASE_URL")
    p.add_argument("--uploads", type=Path, default=_default_uploads())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="0 = все вина с crop")
    p.add_argument("--short-side", type=int, default=384)
    p.add_argument("--jpeg-quality", type=int, default=90)
    return p.parse_args()


def resolve_db_url(cli: str) -> str:
    if cli.strip():
        return cli.strip()
    backend_env = HERE.parents[1] / "backend" / ".env"
    if not backend_env.is_file():
        backend_env = HERE.parents[2] / "vino-svoe" / "backend" / ".env"
    load_dotenv(backend_env)
    import os

    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise SystemExit("DATABASE_URL not set")
    return url.replace("postgresql+psycopg2://", "postgresql://")


def find_crop(uploads: Path, photo_name: str | None, slug: str | None) -> Path | None:
    crop_dir = uploads / "crop"
    stems: list[str] = []
    if photo_name:
        stems.append(Path(photo_name).stem)
        stems.append(Path(photo_name).name)
    if slug:
        stems.append(slug)
    for stem in stems:
        if not stem:
            continue
        for cand in (
            crop_dir / f"{stem}_crop.png",
            crop_dir / f"{stem}_crop.jpg",
            crop_dir / f"{stem}.png",
        ):
            if cand.is_file():
                return cand
    # fuzzy: any file starting with stem
    if photo_name and crop_dir.is_dir():
        stem = Path(photo_name).stem
        hits = list(crop_dir.glob(f"{stem}*_crop.png"))
        if hits:
            return hits[0]
    return None


def resize_short_jpeg(src: Path, dest: Path, short_side: int, quality: int) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    side = min(w, h)
    if side > short_side:
        scale = short_side / float(side)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=quality, optimize=False)


def main() -> None:
    args = parse_args()
    eng = create_engine(resolve_db_url(args.database_url))
    sql = text(
        """
        SELECT id, slug, name, winery, photo_name
        FROM wines
        WHERE crop IN (1, 2)
        ORDER BY id
        """
    )
    with eng.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(sql)]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    rng = random.Random(args.seed)
    wine_ids = [int(r["id"]) for r in rows]
    rng.shuffle(wine_ids)
    n_val = max(1, int(len(wine_ids) * args.val_ratio)) if wine_ids else 0
    val_ids = set(wine_ids[:n_val])

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "images" / "train").mkdir(parents=True)
    (out / "images" / "val").mkdir(parents=True)

    meta_path = out / "meta.jsonl"
    kept = 0
    missing = 0
    with meta_path.open("w", encoding="utf-8") as mf:
        for r in rows:
            wid = int(r["id"])
            crop = find_crop(args.uploads, r.get("photo_name"), r.get("slug"))
            if crop is None:
                missing += 1
                continue
            split = "val" if wid in val_ids else "train"
            dest = out / "images" / split / str(wid) / f"{wid}.jpg"
            try:
                resize_short_jpeg(crop, dest, args.short_side, args.jpeg_quality)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {wid}: {exc}")
                missing += 1
                continue
            rec = {
                "wine_id": wid,
                "slug": r.get("slug"),
                "name": r.get("name"),
                "winery": r.get("winery"),
                "split": split,
                "path": str(dest.relative_to(out)).replace("\\", "/"),
                "source_crop": str(crop),
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    (out / "README.txt").write_text(
        f"SigLIP2 finetune dataset\n"
        f"kept={kept} missing_crop={missing}\n"
        f"uploads={args.uploads}\n"
        f"short_side={args.short_side}\n"
        f"Zip this folder and upload to Google Drive for Colab.\n",
        encoding="utf-8",
    )
    print(f"OK kept={kept} missing={missing} out={out}")


if __name__ == "__main__":
    main()
