#!/usr/bin/env python3
"""Build catalog thumbs for /media/t/... (grid cards ~200px tall).

Always writes .webp under --dst with the same relative path as the source
(stem only; extension forced to .webp).

  python distrib/scripts/build_media_thumbs.py \\
    --src /var/lib/vino-svoe/media \\
    --dst /var/lib/vino-svoe/media_thumbs \\
    --max-side 400

On aidispatcher without host Pillow:

  docker run --rm \\
    -v /var/lib/vino-svoe/media:/in:ro \\
    -v /var/lib/vino-svoe/media_thumbs:/out \\
    -v /opt/vino-svoe/distrib/scripts/build_media_thumbs.py:/work/build.py:ro \\
    python:3.12-slim bash -c \\
      'pip install -q pillow && python /work/build.py --src /in --dst /out --max-side 400'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow required: pip install pillow", file=sys.stderr)
    raise SystemExit(1)

_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _EXTS:
            continue
        yield p


def make_thumb(src: Path, dst: Path, max_side: int, quality: int) -> str:
    """Returns 'built' | 'skipped'."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return "skipped"
    with Image.open(src) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1.0:
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        im.save(dst, format="WEBP", quality=quality, method=4)
    return "built"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=400)
    ap.add_argument("--quality", type=int, default=72)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src_root: Path = args.src.resolve()
    dst_root: Path = args.dst.resolve()
    if not src_root.is_dir():
        print(f"src not a dir: {src_root}", file=sys.stderr)
        raise SystemExit(2)

    built = skipped = errors = 0
    for i, src in enumerate(iter_images(src_root)):
        if args.limit and i >= args.limit:
            break
        rel = src.relative_to(src_root)
        dst = (dst_root / rel).with_suffix(".webp")
        try:
            status = make_thumb(src, dst, args.max_side, args.quality)
            if status == "built":
                built += 1
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"ERR {rel}: {e}", file=sys.stderr)
        if (i + 1) % 200 == 0:
            print(f"... {i + 1} scanned, built={built}, skipped={skipped}", flush=True)

    print(f"done built={built} skipped={skipped} errors={errors} dst={dst_root}")


if __name__ == "__main__":
    main()
