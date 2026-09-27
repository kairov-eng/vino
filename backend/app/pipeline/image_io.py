"""Load common image formats (JPEG/PNG/WebP/HEIC/…) into RGB PIL images.

Normalize path: OpenCV decode + INTER_AREA resize + JPEG q=85 (no optimize).
Fallback to Pillow for HEIC / formats OpenCV cannot decode.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

_heif_ready = False

# Work JPEG for findwine (balance size / speed / quality)
NORMALIZE_JPEG_QUALITY = 85


def sniff_image_extension(data: bytes) -> str | None:
    """Canonical file extension (with dot) from magic bytes — not from the client name."""
    if not data or len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    # ISO BMFF: ....ftypXXXX
    if data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand == b"avif":
            return ".avif"
        if brand in (b"heic", b"heif", b"mif1", b"msf1", b"hevc"):
            return ".heic"
    return None


def align_filename_extension(filename: str, data: bytes) -> tuple[str, str | None]:
    """Rewrite client filename extension to match payload bytes.

    Returns (safe_name, sniffed_ext_or_None).
    """
    base = Path(filename or "upload.bin").name
    sniffed = sniff_image_extension(data)
    stem = Path(base).stem.strip(" ._") or "upload"
    if sniffed:
        return f"{stem}{sniffed}", sniffed
    # unknown — keep original suffix if any
    suf = Path(base).suffix
    return f"{stem}{suf}" if suf else stem, None


def ensure_heif_support() -> None:
    """Register HEIC/HEIF opener once (no-op if pillow-heif missing until used)."""
    global _heif_ready
    if _heif_ready:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        _heif_ready = True
    except ImportError as exc:
        raise RuntimeError(
            "Для HEIC/HEIF нужен пакет pillow-heif (pip install pillow-heif)"
        ) from exc


def _maybe_register_heif(data: bytes | None = None, path: Path | None = None) -> None:
    """Register HEIF if path/bytes look like HEIC or always try once if not ready."""
    global _heif_ready
    if _heif_ready:
        return
    name = (path.name if path else "").lower()
    looks_heif = name.endswith((".heic", ".heif"))
    if data and len(data) >= 12:
        head = data[4:12].lower()
        if b"ftyp" in data[:12].lower() or any(
            x in head for x in (b"heic", b"heif", b"mif1", b"msf1")
        ):
            looks_heif = True
    if looks_heif or not _heif_ready:
        try:
            ensure_heif_support()
        except RuntimeError:
            if looks_heif:
                raise


def _exif_orientation(data: bytes) -> int:
    """Read EXIF Orientation without decoding full pixels when possible."""
    try:
        im = Image.open(BytesIO(data))
        exif = im.getexif()
        if not exif:
            return 1
        return int(exif.get(0x0112, 1) or 1)
    except Exception:  # noqa: BLE001
        return 1


def _apply_exif_bgr(bgr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation to BGR ndarray (OpenCV)."""
    import cv2

    o = int(orientation or 1)
    if o == 2:
        return cv2.flip(bgr, 1)
    if o == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if o == 4:
        return cv2.flip(bgr, 0)
    if o == 5:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), 1)
    if o == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if o == 7:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if o == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def _resize_bgr_long_side(
    bgr: np.ndarray, max_side: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Downscale BGR with OpenCV INTER_AREA (fast for strong downscale)."""
    import cv2

    h0, w0 = bgr.shape[:2]
    meta: dict[str, Any] = {
        "max_side": int(max_side),
        "original_size": [w0, h0],
        "resized": False,
        "size": [w0, h0],
        "resize_backend": "opencv_inter_area",
    }
    limit = max(1, int(max_side))
    long_side = max(w0, h0)
    if long_side <= limit:
        return bgr, meta
    scale = limit / float(long_side)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    meta.update(
        {
            "resized": True,
            "size": [nw, nh],
            "scale": round(scale, 6),
        }
    )
    return out, meta


def _pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def open_image_rgb(
    source: bytes | Path | str,
) -> tuple[Image.Image, dict[str, Any]]:
    """Open any supported raster format → RGB (EXIF-transpose)."""
    path: Path | None = None
    data: bytes | None = None
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
    else:
        data = source

    _maybe_register_heif(data=data, path=path)

    try:
        img = Image.open(BytesIO(data))
        fmt = img.format
        img.load()
    except UnidentifiedImageError as exc:
        raise ValueError("Файл не является изображением") from exc
    except Exception as exc:  # noqa: BLE001
        try:
            ensure_heif_support()
            img = Image.open(BytesIO(data))
            fmt = img.format
            img.load()
        except Exception as exc2:  # noqa: BLE001
            raise ValueError(f"Не удалось прочитать изображение: {exc2}") from exc

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001
        pass

    info = {
        "format": fmt,
        "size": list(img.size),
        "mode": img.mode,
        "decode_backend": "pillow",
    }
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    return img, info


def save_rgb_jpeg(
    img: Image.Image,
    path: Path,
    *,
    quality: int = NORMALIZE_JPEG_QUALITY,
    optimize: bool = False,
) -> Path:
    """Save RGB JPEG. Default q=85, no optimize (faster than q=95+optimize)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=int(quality), optimize=bool(optimize))
    return path


def _jpeg_bytes(img: Image.Image, *, quality: int, optimize: bool = False) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=bool(optimize))
    return buf.getvalue()


def save_work_jpeg_budget(
    img: Image.Image,
    path: Path,
    *,
    max_bytes: int = 1_000_000,
    max_side: int | None = None,
    quality_start: int = NORMALIZE_JPEG_QUALITY,
) -> dict[str, Any]:
    """Write work JPEG under ``max_bytes`` (default ~1 MiB) for the pipeline.

    Strategy: drop JPEG quality, then shrink long side, repeat. Original upload
    is unchanged — this only affects the normalized work file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    budget = max(50_000, int(max_bytes))
    q0 = int(max(40, min(95, quality_start)))
    qualities = [q for q in (q0, 80, 75, 70, 65, 60, 55, 50) if q <= q0]
    if not qualities:
        qualities = [60]

    cur = img.convert("RGB") if img.mode != "RGB" else img
    side_cap = int(max_side) if max_side else max(cur.size)
    attempts: list[dict[str, Any]] = []
    best: tuple[bytes, dict[str, Any]] | None = None

    for round_i in range(6):
        if max(cur.size) > side_cap:
            cur, _rm = resize_long_side(cur, side_cap)
        for q in qualities:
            raw = _jpeg_bytes(cur, quality=q, optimize=False)
            meta = {
                "jpeg_quality": q,
                "size": list(cur.size),
                "bytes": len(raw),
                "max_side": side_cap,
                "round": round_i,
            }
            attempts.append(meta)
            if best is None or len(raw) < len(best[0]):
                best = (raw, meta)
            if len(raw) <= budget:
                path.write_bytes(raw)
                return {
                    "ok": True,
                    "path": str(path),
                    "bytes": len(raw),
                    "jpeg_quality": q,
                    "size": list(cur.size),
                    "max_bytes": budget,
                    "shrunk_for_budget": round_i > 0 or q < q0,
                    "attempts": len(attempts),
                }
        # still too big → smaller long side
        side_cap = max(640, int(side_cap * 0.82))
        if side_cap >= max(cur.size):
            # force one more shrink even if already under side_cap numerically
            side_cap = max(640, int(max(cur.size) * 0.82))
        cur, _ = resize_long_side(cur, side_cap)

    assert best is not None
    path.write_bytes(best[0])
    return {
        "ok": True,
        "path": str(path),
        "bytes": len(best[0]),
        "jpeg_quality": best[1]["jpeg_quality"],
        "size": best[1]["size"],
        "max_bytes": budget,
        "shrunk_for_budget": True,
        "budget_miss": len(best[0]) > budget,
        "attempts": len(attempts),
    }


def resize_long_side(
    img: Image.Image,
    max_side: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Downscale so max(width, height) ≤ max_side. Prefer OpenCV INTER_AREA."""
    w0, h0 = img.size
    meta: dict[str, Any] = {
        "max_side": int(max_side),
        "original_size": [w0, h0],
        "resized": False,
        "size": [w0, h0],
    }
    limit = max(1, int(max_side))
    long_side = max(w0, h0)
    if long_side <= limit:
        return img, meta
    scale = limit / float(long_side)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    try:
        import cv2

        rgb = np.asarray(img)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected RGB array")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        out = _pil_from_bgr(out_bgr)
        meta["resize_backend"] = "opencv_inter_area"
    except Exception:  # noqa: BLE001
        out = img.resize((nw, nh), Image.Resampling.BILINEAR)
        meta["resize_backend"] = "pillow_bilinear"
    meta.update({"resized": True, "size": [nw, nh], "scale": round(scale, 6)})
    return out, meta


def resize_short_side(
    img: Image.Image,
    short_side: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Downscale so min(width, height) ≤ short_side (SigLIP/HF prep)."""
    w0, h0 = img.size
    meta: dict[str, Any] = {
        "short_side": int(short_side),
        "original_size": [w0, h0],
        "resized": False,
        "size": [w0, h0],
    }
    limit = max(1, int(short_side))
    short0 = min(w0, h0)
    if short0 <= limit:
        return img, meta
    scale = limit / float(short0)
    nw = max(1, int(round(w0 * scale)))
    nh = max(1, int(round(h0 * scale)))
    try:
        import cv2

        rgb = np.asarray(img)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected RGB array")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        out = _pil_from_bgr(out_bgr)
        meta["resize_backend"] = "opencv_inter_area"
    except Exception:  # noqa: BLE001
        out = img.resize((nw, nh), Image.Resampling.BILINEAR)
        meta["resize_backend"] = "pillow_bilinear"
    meta.update({"resized": True, "size": [nw, nh], "scale": round(scale, 6)})
    return out, meta


def save_short_side_jpeg(
    src: Path | Image.Image,
    dest: Path,
    *,
    short_side: int = 384,
    quality: int = NORMALIZE_JPEG_QUALITY,
) -> dict[str, Any]:
    """Write JPEG with short side ≤ ``short_side`` (for HF SigLIP2 payload)."""
    if isinstance(src, Image.Image):
        img = src.convert("RGB") if src.mode != "RGB" else src
        src_path = None
    else:
        img, _ = open_image_rgb(Path(src))
        src_path = str(src)
    out, rmeta = resize_short_side(img, short_side)
    save_rgb_jpeg(out, dest, quality=quality, optimize=False)
    nbytes = dest.stat().st_size if dest.is_file() else 0
    return {
        "ok": True,
        "path": str(dest),
        "filename": dest.name,
        "source": src_path,
        "short_side": int(short_side),
        "original_size": rmeta.get("original_size"),
        "size": rmeta.get("size"),
        "resized": bool(rmeta.get("resized")),
        "scale": rmeta.get("scale"),
        "resize_backend": rmeta.get("resize_backend"),
        "jpeg_quality": int(quality),
        "bytes": int(nbytes),
    }


def normalize_image_bytes(
    data: bytes,
    *,
    max_side: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Fast path for findwine normalize: OpenCV decode+INTER_AREA, else Pillow.

    Returns RGB PIL image already downscaled to max_side, plus meta for status.
    """
    import cv2

    info: dict[str, Any] = {
        "normalize_max_side": int(max_side),
        "jpeg_quality": NORMALIZE_JPEG_QUALITY,
    }

    arr = np.frombuffer(data, dtype=np.uint8)
    # OpenCV 4.5+/5 auto-applies EXIF Orientation on imdecode unless IGNORE is set.
    # We apply orientation ourselves (same as Pillow path) — ignore auto-orient.
    decode_flags = cv2.IMREAD_COLOR
    ignore_orient = getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    if ignore_orient:
        decode_flags |= int(ignore_orient)
    bgr = cv2.imdecode(arr, decode_flags)
    if bgr is not None:
        orient = _exif_orientation(data)
        if orient != 1:
            bgr = _apply_exif_bgr(bgr, orient)
        h0, w0 = bgr.shape[:2]
        # Probe format via Pillow headers (cheap)
        source_format = None
        mode = "RGB"
        try:
            probe = Image.open(BytesIO(data))
            source_format = probe.format
            mode = probe.mode or "RGB"
        except Exception:  # noqa: BLE001
            pass
        bgr, resize_meta = _resize_bgr_long_side(bgr, max_side)
        img = _pil_from_bgr(bgr)
        info.update(
            {
                "ok": True,
                "decode_backend": "opencv",
                "source_format": source_format,
                "mode": mode,
                "original_size": [w0, h0],
                "size": resize_meta.get("size") or [w0, h0],
                "resized": bool(resize_meta.get("resized")),
                "scale": resize_meta.get("scale"),
                "resize_backend": resize_meta.get("resize_backend"),
                "exif_orientation": orient,
                "opencv_ignore_orientation": bool(ignore_orient),
            }
        )
        return img, info

    # HEIC / exotic → Pillow
    _maybe_register_heif(data=data)
    img, pil_info = open_image_rgb(data)
    img, resize_meta = resize_long_side(img, max_side)
    info.update(
        {
            "ok": True,
            "decode_backend": pil_info.get("decode_backend") or "pillow",
            "source_format": pil_info.get("format"),
            "mode": pil_info.get("mode"),
            "original_size": resize_meta.get("original_size") or pil_info.get("size"),
            "size": resize_meta.get("size"),
            "resized": bool(resize_meta.get("resized")),
            "scale": resize_meta.get("scale"),
            "resize_backend": resize_meta.get("resize_backend"),
        }
    )
    return img, info
