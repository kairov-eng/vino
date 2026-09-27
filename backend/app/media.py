"""Resolve wine bottle / label image paths under MEDIA_ROOT.

Matching logic mirrors label_detect/2. crop_wines.py:
  candidates: wines_sitemap.image_file → wines.photo_name → slug|photo_name
  match: exact name, or stem with '-'→'_' plus optional _<hash10> suffix.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from app.db.config import MEDIA_ROOT, MEDIA_URL_PATH

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_DERIV_PREFIXES = ("thumbnail_", "small_", "medium_", "large_")


def _image_file_parts(image_file: str) -> tuple[str, str, str]:
    name = Path(str(image_file)).name
    path = Path(name)
    stem = path.stem
    ext = path.suffix.lower() or ".webp"
    return name, stem, ext


def _safe_result_basename(image_file: str) -> str:
    raw = str(image_file).strip().replace("\\", "/").split("/")[-1]
    return raw.replace("\\", "_").replace("/", "_") or "unknown"


@lru_cache(maxsize=1)
def _uploads_index() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """exact[lowercase name] + by_stem[lowercase stem / stem without _hash10]."""
    exact: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    root = MEDIA_ROOT
    if not root.is_dir():
        return exact, by_stem
    for p in root.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.startswith(_DERIV_PREFIXES):
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        exact[low] = p
        stem = p.stem.lower()
        by_stem.setdefault(stem, []).append(p)
        # Strapi-style: name_<10 hex chars>
        if (
            len(stem) > 11
            and stem[-11] == "_"
            and all(c in "0123456789abcdef" for c in stem[-10:])
        ):
            by_stem.setdefault(stem[:-11], []).append(p)
    return exact, by_stem


def _match_one(image_file: str) -> Path | None:
    exact_index, stem_index = _uploads_index()
    _, stem, ext = _image_file_parts(image_file)
    name = Path(str(image_file)).name
    prefix = stem.replace("-", "_").lower()

    for key in (name.lower(), f"{stem}{ext}".lower(), f"{prefix}{ext}"):
        hit = exact_index.get(key)
        if hit is not None and hit.is_file():
            return hit

    hits = [p for p in stem_index.get(prefix, []) if p.suffix.lower() == ext]
    if hits:
        hits = sorted(set(hits), key=lambda p: (len(p.name), p.name))
        return hits[0]
    return None


def find_upload_file(
    *,
    photo_name: str | None = None,
    slug: str | None = None,
    sitemap_image_file: str | None = None,
) -> Path | None:
    """Same candidate order as crop_wines.find_upload_file."""
    image_file = (slug or photo_name or "").strip() or None
    for candidate in (sitemap_image_file, photo_name, image_file):
        if not candidate or not str(candidate).strip():
            continue
        found = _match_one(str(candidate).strip())
        if found is not None:
            return found
    return None


def resolve_photo_path(
    photo_name: str | None,
    slug: str | None = None,
    sitemap_image_file: str | None = None,
) -> Path | None:
    return find_upload_file(
        photo_name=photo_name,
        slug=slug,
        sitemap_image_file=sitemap_image_file,
    )


def photo_url(
    photo_name: str | None,
    slug: str | None = None,
    sitemap_image_file: str | None = None,
) -> str | None:
    path = resolve_photo_path(
        photo_name, slug=slug, sitemap_image_file=sitemap_image_file
    )
    if path is None:
        return None
    rel = path.relative_to(MEDIA_ROOT).as_posix()
    return f"{MEDIA_URL_PATH.rstrip('/')}/{quote(rel, safe='/')}"


def resolve_label_path(
    photo_name: str | None,
    slug: str | None = None,
) -> Path | None:
    """crop/<basename>_crop.png — basename as in crop_wines.safe_result_basename."""
    crop_dir = MEDIA_ROOT / "crop"
    if not crop_dir.is_dir():
        return None

    # crop_wines: COALESCE(ws.slug, w.photo_name) then safe_result_basename
    primary = (slug or photo_name or "").strip()
    bases: list[str] = []
    if primary:
        bases.append(_safe_result_basename(primary))
    if photo_name:
        bases.append(_safe_result_basename(photo_name))
    if slug:
        bases.append(_safe_result_basename(slug))
        bases.append(_safe_result_basename(f"{slug}.webp"))

    seen: set[str] = set()
    for base in bases:
        if not base or base in seen:
            continue
        seen.add(base)
        for name in (f"{base}_crop.png", f"{base}_crop.webp", f"{base}_crop.jpg"):
            path = crop_dir / name
            if path.is_file():
                return path
    return None


def label_url(photo_name: str | None, slug: str | None = None) -> str | None:
    path = resolve_label_path(photo_name, slug=slug)
    if path is None:
        return None
    rel = path.relative_to(MEDIA_ROOT).as_posix()
    return f"{MEDIA_URL_PATH.rstrip('/')}/{quote(rel, safe='/')}"


def media_stats() -> dict[str, int | str]:
    exact, by_stem = _uploads_index()
    return {
        "media_root": str(MEDIA_ROOT),
        "exact_files": len(exact),
        "stem_keys": len(by_stem),
        "exists": int(MEDIA_ROOT.exists()),
    }


def clear_media_cache() -> None:
    _uploads_index.cache_clear()
