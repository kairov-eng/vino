# =============================================================================
# 5.-x.generate_synthetic_yolo - 2 класса bottle label.py
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   То же, что «6.generate_synthetic_yolo.py», но ДВА класса YOLO:
#     0 = bottle
#     1 = label
#   Эталонные бутылки из каталога вклеиваются на полевые фото; в .txt
#   пишутся оба бокса (бутылка и этикетка), посчитанные из тех же
#   геометрических преобразований, что и при вклейке.
#
# ВХОДНЫЕ ДАННЫЕ
#   --catalog   → uploads/ + cropjson/ (Gemini: bottle_box_2d, label_box_2d)
#   --my-photos → uploads.my/yolo/images/{train,val} + опционально cropjson/
#                 (боксы «занятых» зон на фоне)
#
# АЛГОРИТМ (кратко)
#   1. Cutout бутылки по маске фона/альфе + известные Gemini-боксы.
#   2. Случайный фон 768×1024, аугментации, несколько эталонов на кадр.
#   3. Фильтр сильной окклюзии этикетки/бутылки.
#   4. YOLO-строки: class bottle и class label (норм. xc yc w h).
#   5. train/val split, data.yaml, preview.
#
# ВЫХОД (--out → uploads.syntetic\yolo по умолчанию)
#   images/{train,val}/, labels/{train,val}/, data.yaml:
#     names:
#       0: bottle
#       1: label
#
# КОГДА ИСПОЛЬЗОВАТЬ
#   Нужна совместная детекция бутылки и этикетки (YOLO11n и т.п.).
#   Если нужна только этикетка — используйте скрипт 6 (один класс label).
#
# ПРИМЕРЫ ВЫЗОВА
#   python "5.-x.generate_synthetic_yolo - 2 класса bottle label.py"
#   python "5.-x.generate_synthetic_yolo - 2 класса bottle label.py" --n 1500 --seed 42
#   python "5.-x.generate_synthetic_yolo - 2 класса bottle label.py" --n 50 --preview 8
#
# ЗАВИСИМОСТИ
#   numpy, Pillow, scipy
#
# =============================================================================

"""Генерация синтетического YOLO-датасета: эталоны на фоне полевых фото (bottle+label)."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage as ndi

SCRIPT_DIR = Path(__file__).resolve().parent
_MONOREPO = Path(__file__).resolve().parents[2]  # Vino2026
DEFAULT_CATALOG = _MONOREPO / "media" / "uploads"
DEFAULT_MY = _MONOREPO / "uploads.my"
DEFAULT_OUT = _MONOREPO / "uploads.syntetic" / "yolo"

CLASS_BOTTLE = 0
CLASS_LABEL = 1
MAX_CUTOUT_H = 1100
OUT_H = 1024
OUT_W = 768  # 3:4, как телефон
MIN_BOX_PX = 10
MIN_BOX_FRAC = 0.0015
LABEL_OCCLUDE_THR = 0.60
BOTTLE_OCCLUDE_THR = 0.70


@dataclass
class CutoutSpec:
    path: Path
    bottle_box: list[int]  # gemini [ymin, xmin, ymax, xmax] 0..1000
    label_box: list[int]


@dataclass
class Cutout:
    rgba: np.ndarray  # HxWx4 uint8, плотный кроп по маске
    bottle: tuple[float, float, float, float]  # x1,y1,x2,y2 в кропе
    label: tuple[float, float, float, float]
    name: str


@dataclass
class Background:
    path: Path
    cover_boxes: list[tuple[float, float, float, float]]  # xyxy 0..1


@dataclass
class Placed:
    bottle: tuple[float, float, float, float]
    label: tuple[float, float, float, float]
    mask: np.ndarray  # bool HxW видимой части бутылки на холсте


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def gemini_to_xyxy(
    box: list[int], width: int, height: int
) -> tuple[float, float, float, float]:
    ymin, xmin, ymax, xmax = (float(v) for v in box)
    x1 = xmin / 1000.0 * width
    y1 = ymin / 1000.0 * height
    x2 = xmax / 1000.0 * width
    y2 = ymax / 1000.0 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def clamp_box(
    box: tuple[float, float, float, float], w: int, h: int
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), w)
    y1 = min(max(y1, 0.0), h)
    x2 = min(max(x2, 0.0), w)
    y2 = min(max(y2, 0.0), h)
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return None
    if (x2 - x1) * (y2 - y1) < MIN_BOX_FRAC * w * h:
        return None
    return x1, y1, x2, y2


def yolo_line(cls: int, box: tuple[float, float, float, float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def valid_gemini(box) -> list[int] | None:
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    ymin, xmin, ymax, xmax = vals
    if xmax <= xmin or ymax <= ymin:
        return None
    if xmax - xmin < 8 or ymax - ymin < 8:
        return None
    return vals


def collect_cutout_specs(catalog_dir: Path) -> list[CutoutSpec]:
    cropjson = catalog_dir / "cropjson"
    specs: list[CutoutSpec] = []
    skipped = 0
    for jp in sorted(cropjson.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        src = data.get("source_file") or data.get("image_file")
        if not isinstance(src, str) or not src.strip():
            skipped += 1
            continue
        path = catalog_dir / Path(src.strip()).name
        if not path.is_file():
            skipped += 1
            continue
        label = valid_gemini(data.get("label_box_2d"))
        if label is None:
            skipped += 1
            continue
        bottle = valid_gemini(data.get("bottle_box_2d")) or [0, 0, 1000, 1000]
        specs.append(CutoutSpec(path=path, bottle_box=bottle, label_box=label))
    print(f"эталоны: {len(specs)} (пропущено json/файлов {skipped})", flush=True)
    return specs


def collect_backgrounds(my_dir: Path) -> list[Background]:
    yolo_root = my_dir / "yolo"
    images: list[Path] = []
    for split in ("train", "val"):
        d = yolo_root / "images" / split
        if d.is_dir():
            images.extend(sorted(d.glob("*.jpg")))
            images.extend(sorted(d.glob("*.jpeg")))
    if not images:
        raise SystemExit(f"Нет JPEG-фонов в {yolo_root / 'images'}")

    cover_by_stem: dict[str, list[tuple[float, float, float, float]]] = {}
    cropjson = my_dir / "cropjson"
    if cropjson.is_dir():
        for jp in cropjson.glob("*.json"):
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            src = str(data.get("source_file") or data.get("image_file") or "")
            stem = Path(src).stem if src else jp.stem.split(".")[0]
            if "_" in jp.stem and jp.stem.rsplit("_", 1)[-1].isdigit():
                # IMG_3861.HEIC_1 → stem IMG_3861
                stem = Path(src).stem if src else stem
            boxes: list[tuple[float, float, float, float]] = []
            items = []
            raw = data.get("gemini_raw")
            if isinstance(raw, dict) and isinstance(raw.get("items"), list):
                items = raw["items"]
            elif data.get("bottle_box_2d") or data.get("label_box_2d"):
                items = [data]
            for it in items:
                b = valid_gemini(it.get("bottle_box_2d")) or valid_gemini(
                    it.get("label_box_2d")
                )
                if b is None:
                    continue
                ymin, xmin, ymax, xmax = (v / 1000.0 for v in b)
                boxes.append((xmin, ymin, xmax, ymax))
            if boxes:
                cover_by_stem.setdefault(stem, []).extend(boxes)

    # YOLO-этикетки — запасной источник регионов для замазывания
    for img in images:
        lbl = None
        for split in ("train", "val"):
            cand = yolo_root / "labels" / split / f"{img.stem}.txt"
            if cand.is_file():
                lbl = cand
                break
        if lbl is None:
            continue
        extra: list[tuple[float, float, float, float]] = []
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = (float(parts[0]),) + tuple(float(x) for x in parts[1:])
            # этикетка → грубый бокс бутылки
            bw = min(w * 1.25, 0.95)
            bh = min(h * 1.85, 0.98)
            yc = yc - h * 0.15
            extra.append((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
        if extra:
            cover_by_stem.setdefault(img.stem, []).extend(extra)

    bgs = [
        Background(path=p, cover_boxes=cover_by_stem.get(p.stem, [])) for p in images
    ]
    with_cover = sum(1 for b in bgs if b.cover_boxes)
    print(f"фоны: {len(bgs)} (с регионами замазывания {with_cover})", flush=True)
    return bgs


def mask_from_alpha(alpha: np.ndarray) -> np.ndarray:
    return alpha > 16


def mask_from_background(rgb: np.ndarray, tol: float = 30.0) -> np.ndarray:
    """Пиксели фона = похожи на цвет рамки и связаны с краем (белая этикетка жива)."""
    h, w = rgb.shape[:2]
    border = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
    )
    bg = np.median(border.astype(np.float32), axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    similar = dist < tol
    if similar.mean() < 0.02:
        similar = dist < (tol * 1.8)
    edge = np.zeros_like(similar, dtype=bool)
    edge[0, :] = similar[0, :]
    edge[-1, :] = similar[-1, :]
    edge[:, 0] = similar[:, 0]
    edge[:, -1] = similar[:, -1]
    bg_mask = ndi.binary_propagation(edge, mask=similar)
    bg_mask = ndi.binary_dilation(bg_mask, iterations=2)
    bottle = ~bg_mask
    bottle = ndi.binary_opening(bottle, iterations=1)
    bottle = ndi.binary_fill_holes(bottle)
    return bottle


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    out = labeled == keep
    return ndi.binary_fill_holes(out)


def extract_cutout(spec: CutoutSpec) -> Cutout | None:
    im = Image.open(spec.path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    w0, h0 = im.size
    if h0 > MAX_CUTOUT_H:
        scale = MAX_CUTOUT_H / h0
        im = im.resize(
            (max(1, int(w0 * scale)), MAX_CUTOUT_H), Image.Resampling.BILINEAR
        )
    arr = np.array(im.convert("RGBA"), dtype=np.uint8, copy=True)
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    if float((alpha < 10).mean()) >= 0.05:
        mask = mask_from_alpha(alpha)
    else:
        mask = largest_component(mask_from_background(arr[:, :, :3]))
    if mask.mean() < 0.01:
        return None
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = 1
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = arr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[:, :, 3] = np.where(crop_mask, np.maximum(crop[:, :, 3], 220), 0)

    # gemini 0..1000 относительные — считаем уже в текущем размере
    bx = gemini_to_xyxy(spec.bottle_box, w, h)
    lx = gemini_to_xyxy(spec.label_box, w, h)
    bottle = (bx[0] - x1, bx[1] - y1, bx[2] - x1, bx[3] - y1)
    label = (lx[0] - x1, lx[1] - y1, lx[2] - x1, lx[3] - y1)
    ch, cw = crop.shape[:2]
    bottle_c = clamp_box(bottle, cw, ch) or (0.0, 0.0, float(cw), float(ch))
    label_c = clamp_box(label, cw, ch)
    if label_c is None:
        return None
    return Cutout(rgba=crop, bottle=bottle_c, label=label_c, name=spec.path.stem)


@lru_cache(maxsize=256)
def load_cutout_cached(path_str: str, bottle_t: tuple, label_t: tuple) -> Cutout | None:
    spec = CutoutSpec(path=Path(path_str), bottle_box=list(bottle_t), label_box=list(label_t))
    return extract_cutout(spec)


def get_cutout(spec: CutoutSpec) -> Cutout | None:
    try:
        return load_cutout_cached(
            str(spec.path), tuple(spec.bottle_box), tuple(spec.label_box)
        )
    except Exception:
        return None


def homography_from_quads(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-8:
        return None
    h = h / h[2, 2]
    if not np.isfinite(h).all():
        return None
    return h


def warp_points(hmat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    w = p @ hmat.T
    den = np.maximum(np.abs(w[:, 2:3]), 1e-8)
    return w[:, :2] / den


def pil_perspective_coeffs(hmat: np.ndarray) -> tuple[float, ...]:
    inv = np.linalg.inv(hmat)
    inv = inv / inv[2, 2]
    a, b, c = inv[0]
    d, e, f = inv[1]
    g, h = inv[2, 0], inv[2, 1]
    return (float(a), float(b), float(c), float(d), float(e), float(f), float(g), float(h))


def rotate_rgba(
    rgba: np.ndarray, angle: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Поворот вокруг центра, expand. Возвращает RGBA и (cx_old, cy_old, cx_new, cy_new)."""
    im = Image.fromarray(rgba, "RGBA")
    w, h = im.size
    rot = im.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
    nw, nh = rot.size
    return np.asarray(rot), (w / 2.0, h / 2.0, nw / 2.0, nh / 2.0)


def rotate_box(
    box: tuple[float, float, float, float],
    centers: tuple[float, float, float, float],
    angle: float,
) -> tuple[float, float, float, float]:
    cx, cy, ncx, ncy = centers
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    out = []
    for x, y in corners:
        dx, dy = x - cx, y - cy
        # PIL.rotate CCW при оси Y вниз (пиксельные координаты)
        xr = cx + dx * cos_a + dy * sin_a
        yr = cy - dx * sin_a + dy * cos_a
        out.append((xr - cx + ncx, yr - cy + ncy))
    xs = [p[0] for p in out]
    ys = [p[1] for p in out]
    return min(xs), min(ys), max(xs), max(ys)


def add_glare(rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = rgba.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx = rng.uniform(0.28, 0.72) * w
    cy = rng.uniform(0.12, 0.48) * h
    rx = max(2.0, rng.uniform(0.07, 0.18) * w)
    ry = max(2.0, rng.uniform(0.16, 0.42) * h)
    ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    g = np.clip(1.0 - ell, 0.0, 1.0) ** 2
    strength = rng.uniform(0.18, 0.48)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgba.astype(np.float32)
    add = g[:, :, None] * strength * 255.0 * alpha
    out[:, :, :3] = np.clip(out[:, :, :3] + add, 0, 255)
    return out.astype(np.uint8)


def match_lighting(rgba: np.ndarray, bg_patch: np.ndarray, strength: float) -> np.ndarray:
    alpha = rgba[:, :, 3]
    sel = alpha > 40
    if sel.sum() < 40 or bg_patch.size == 0:
        return rgba
    src = rgba[:, :, :3][sel].astype(np.float32).mean(axis=0)
    dst = bg_patch.reshape(-1, 3).astype(np.float32).mean(axis=0)
    gain = dst / np.maximum(src, 1.0)
    gain = 1.0 + (gain - 1.0) * strength
    out = rgba.astype(np.float32)
    out[:, :, :3] *= gain
    return np.clip(out, 0, 255).astype(np.uint8)


def color_jitter_rgba(rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    im = Image.fromarray(rgba, "RGBA")
    rgb, a = im.convert("RGB"), im.getchannel("A")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.88, 1.14))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.88, 1.18))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.82, 1.18))
    arr = np.asarray(rgb).astype(np.float32)
    temp = rng.uniform(-0.10, 0.12)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + temp), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 - temp), 0, 255)
    gamma = rng.uniform(0.82, 1.22)
    arr = np.clip(255.0 * ((arr / 255.0) ** gamma), 0, 255)
    out = np.dstack([arr.astype(np.uint8), np.asarray(a)])
    return out


def feather_alpha(rgba: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    out = rgba.copy()
    a = ndi.gaussian_filter(out[:, :, 3].astype(np.float32), sigma=sigma)
    out[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    return out


def motion_blur_rgb(rgb: np.ndarray, ksize: int, angle: float) -> np.ndarray:
    if ksize < 3:
        return rgb
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    kernel = ndi.rotate(kernel, angle, reshape=False, order=1)
    s = kernel.sum()
    if s <= 1e-6:
        return rgb
    kernel /= s
    out = np.empty_like(rgb)
    for c in range(3):
        out[:, :, c] = ndi.convolve(rgb[:, :, c].astype(np.float32), kernel, mode="nearest")
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_recompress(rgb: np.ndarray, quality: int) -> np.ndarray:
    im = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def cover_regions(
    rgb: np.ndarray, boxes01: list[tuple[float, float, float, float]]
) -> np.ndarray:
    if not boxes01:
        return rgb
    h, w = rgb.shape[:2]
    # сильный блюр через даунскейл — быстрее gaussian sigma=22 на 12 МП
    small = np.array(
        Image.fromarray(rgb, "RGB").resize(
            (max(1, w // 8), max(1, h // 8)), Image.Resampling.BILINEAR
        )
    )
    blurred = np.array(
        Image.fromarray(small, "RGB").resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    out = rgb.astype(np.float32)
    for x1, y1, x2, y2 in boxes01:
        xa = int(max(0, min(w, x1 * w)))
        xb = int(max(0, min(w, x2 * w)))
        ya = int(max(0, min(h, y1 * h)))
        yb = int(max(0, min(h, y2 * h)))
        if xb - xa < 4 or yb - ya < 4:
            continue
        # чуть расширить, чтобы не оставлять край этикетки
        px, py = int(0.04 * (xb - xa)), int(0.04 * (yb - ya))
        xa, ya = max(0, xa - px), max(0, ya - py)
        xb, yb = min(w, xb + px), min(h, yb + py)
        out[ya:yb, xa:xb] = blurred[ya:yb, xa:xb]
    return np.clip(out, 0, 255).astype(np.uint8)


def random_phone_crop(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = rgb.shape[:2]
    zoom = rng.uniform(0.55, 1.0)
    crop_h = max(int(h * zoom), OUT_H)
    crop_w = int(crop_h * OUT_W / OUT_H)
    if crop_w > w:
        crop_w = w
        crop_h = int(crop_w * OUT_H / OUT_W)
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    x0 = rng.randint(0, max(0, w - crop_w))
    y0 = rng.randint(0, max(0, h - crop_h))
    crop = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    im = Image.fromarray(crop, "RGB").resize((OUT_W, OUT_H), Image.Resampling.BILINEAR)
    return np.array(im, dtype=np.uint8, copy=True)


def alpha_paste(
    canvas: np.ndarray, src_rgba: np.ndarray, x: int, y: int
) -> np.ndarray:
    ch, cw = canvas.shape[:2]
    sh, sw = src_rgba.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + sw), min(ch, y + sh)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((ch, cw), dtype=bool)
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    src = src_rgba[sy1:sy2, sx1:sx2].astype(np.float32)
    a = src[:, :, 3:4] / 255.0
    dst = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = np.clip(src[:, :, :3] * a + dst * (1.0 - a), 0, 255).astype(
        np.uint8
    )
    vis = np.zeros((ch, cw), dtype=bool)
    vis[y1:y2, x1:x2] = src_rgba[sy1:sy2, sx1:sx2, 3] > 24
    return vis


def box_corners(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def aabb(pts: np.ndarray) -> tuple[float, float, float, float]:
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(
        pts[:, 1].max()
    )


def occluded_fraction(
    box: tuple[float, float, float, float], occ: np.ndarray
) -> float:
    h, w = occ.shape
    clipped = clamp_box(box, w, h)
    if clipped is None:
        return 1.0
    x1, y1, x2, y2 = (int(round(v)) for v in clipped)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    region = occ[y1:y2, x1:x2]
    return float(region.mean()) if region.size else 1.0


def transform_bottle(
    cut: Cutout,
    canvas: np.ndarray,
    rng: random.Random,
    target_h: int,
    x: int,
    y: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    rgba = cut.rgba
    h0, w0 = rgba.shape[:2]
    scale = target_h / h0
    nw, nh = max(8, int(w0 * scale)), max(8, int(h0 * scale))
    rgba = np.asarray(
        Image.fromarray(rgba, "RGBA").resize((nw, nh), Image.Resampling.BILINEAR)
    )
    bottle = tuple(v * scale for v in cut.bottle)
    label = tuple(v * scale for v in cut.label)

    angle = rng.uniform(-15.0, 15.0)
    rgba, centers = rotate_rgba(rgba, angle)
    bottle = rotate_box(bottle, centers, angle)  # type: ignore[arg-type]
    label = rotate_box(label, centers, angle)  # type: ignore[arg-type]
    h1, w1 = rgba.shape[:2]

    mag = rng.uniform(0.02, 0.10)
    src_q = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float64)
    dst_q = src_q.copy()
    jitter = mag * min(w1, h1)
    for i in range(4):
        dst_q[i, 0] += rng.uniform(-jitter, jitter)
        dst_q[i, 1] += rng.uniform(-jitter, jitter)
    dst_q[:, 0] -= dst_q[:, 0].min()
    dst_q[:, 1] -= dst_q[:, 1].min()
    out_w = max(8, int(math.ceil(dst_q[:, 0].max())))
    out_h = max(8, int(math.ceil(dst_q[:, 1].max())))
    hmat = homography_from_quads(src_q, dst_q)
    if hmat is not None:
        try:
            coeffs = pil_perspective_coeffs(hmat)
            warped = Image.fromarray(rgba, "RGBA").transform(
                (out_w, out_h),
                Image.Transform.PERSPECTIVE,
                coeffs,
                resample=Image.Resampling.BILINEAR,
            )
            rgba = np.asarray(warped)
            bottle = aabb(warp_points(hmat, box_corners(bottle)))  # type: ignore[arg-type]
            label = aabb(warp_points(hmat, box_corners(label)))  # type: ignore[arg-type]
        except (np.linalg.LinAlgError, ValueError):
            pass

    rgba = color_jitter_rgba(rgba, rng)
    if rng.random() < 0.75:
        rgba = add_glare(rgba, rng)

    bh, bw = rgba.shape[:2]
    # патч фона под бутылкой для подгонки яркости
    xa, ya = max(0, x), max(0, y)
    xb, yb = min(canvas.shape[1], x + bw), min(canvas.shape[0], y + bh)
    if xb > xa and yb > ya:
        rgba = match_lighting(
            rgba, canvas[ya:yb, xa:xb], strength=rng.uniform(0.25, 0.50)
        )
    rgba = feather_alpha(rgba, sigma=rng.uniform(1.2, 2.4))

    vis = alpha_paste(canvas, rgba, x, y)
    bottle_t = (bottle[0] + x, bottle[1] + y, bottle[2] + x, bottle[3] + y)
    label_t = (label[0] + x, label[1] + y, label[2] + x, label[3] + y)
    return vis, bottle_t, label_t


def draw_price_tag(canvas: np.ndarray, rng: random.Random, near_x: int, near_y: int) -> None:
    h, w = canvas.shape[:2]
    tw, th = rng.randint(48, 90), rng.randint(28, 50)
    x = min(max(near_x + rng.randint(-30, 40), 0), max(0, w - tw))
    y = min(max(near_y + rng.randint(-20, 30), 0), max(0, h - th))
    color = (
        rng.randint(230, 255),
        rng.randint(220, 245),
        rng.randint(120, 190),
    )
    canvas[y : y + th, x : x + tw] = color
    # «текст» — тёмные полоски
    for i in range(rng.randint(2, 4)):
        yy = y + 6 + i * (th // 5)
        xx1 = x + 6
        xx2 = x + tw - 6
        if 0 <= yy < h and xx2 > xx1:
            canvas[yy : yy + 2, xx1:xx2] = (40, 40, 40)


def draw_shelf_bar(canvas: np.ndarray, rng: random.Random) -> None:
    h, w = canvas.shape[:2]
    bar_h = rng.randint(10, 22)
    y = rng.choice([0, h - bar_h, int(h * rng.uniform(0.62, 0.92))])
    y = min(max(y, 0), h - bar_h)
    shade = rng.randint(20, 70)
    alpha = rng.uniform(0.35, 0.7)
    band = canvas[y : y + bar_h, :].astype(np.float32)
    canvas[y : y + bar_h, :] = np.clip(
        band * (1 - alpha) + shade * alpha, 0, 255
    ).astype(np.uint8)


def place_scene(
    canvas: np.ndarray,
    specs: list[CutoutSpec],
    rng: random.Random,
) -> list[Placed]:
    h, w = canvas.shape[:2]
    placed: list[Placed] = []

    closeup = rng.random() < 0.40
    n = 1 if closeup else 1 + (rng.randint(1, 3) if rng.random() < 0.60 else 0)
    n = min(n, 4)

    cuts: list[Cutout] = []
    for _ in range(n * 4):
        if len(cuts) >= n:
            break
        spec = specs[rng.randrange(len(specs))]
        cut = get_cutout(spec)
        if cut is not None:
            cuts.append(cut)
    if not cuts:
        return []

    if closeup:
        cut = cuts[0]
        target_h = int(h * rng.uniform(0.55, 1.05))
        scale = target_h / cut.rgba.shape[0]
        bw = int(cut.rgba.shape[1] * scale)
        # часто обрезана сверху/снизу
        x = int((w - bw) / 2 + rng.uniform(-0.12, 0.12) * w)
        if rng.random() < 0.45:
            y = int(rng.uniform(-0.18 * target_h, 0.12 * h))
        else:
            y = int(h - target_h * rng.uniform(0.82, 1.12))
        res = transform_bottle(cut, canvas, rng, target_h, x, y)
        if res is not None:
            vis, b, l = res
            placed.append(Placed(bottle=b, label=l, mask=vis))
        return placed

    # полка: общий «пол» и ряд бутылок
    target_h = int(h * rng.uniform(0.42, 0.78))
    bottoms = int(h * rng.uniform(0.88, 1.02))
    gap = rng.randint(-8, 18)
    # суммарная ширина ряда — грубая оценка
    widths = []
    for cut in cuts:
        sc = target_h / cut.rgba.shape[0]
        widths.append(int(cut.rgba.shape[1] * sc * rng.uniform(0.92, 1.08)))
    total = sum(widths) + gap * (len(cuts) - 1)
    x_cursor = int((w - total) / 2 + rng.uniform(-0.08, 0.08) * w)
    for i, cut in enumerate(cuts):
        th = int(target_h * rng.uniform(0.90, 1.10))
        sc = th / cut.rgba.shape[0]
        bw = int(cut.rgba.shape[1] * sc)
        x = x_cursor
        y = bottoms - th + rng.randint(-12, 18)
        x_cursor += bw + gap
        res = transform_bottle(cut, canvas, rng, th, x, y)
        if res is None:
            continue
        vis, b, l = res
        placed.append(Placed(bottle=b, label=l, mask=vis))

    return placed


def filter_occluded(placed: list[Placed], w: int, h: int) -> list[Placed]:
    """Бутылки рисуются по порядку (первые — сзади). Поздние закрывают ранние."""
    kept: list[Placed] = []
    occ = np.zeros((h, w), dtype=bool)
    for p in reversed(placed):
        # видимость относительно того, что уже лежит сверху
        bottle = clamp_box(p.bottle, w, h)
        label = clamp_box(p.label, w, h)
        drop_label = label is None or occluded_fraction(label, occ) > LABEL_OCCLUDE_THR
        drop_bottle = bottle is None or occluded_fraction(bottle, occ) > BOTTLE_OCCLUDE_THR
        if drop_bottle and drop_label:
            occ |= p.mask
            continue
        kept.append(
            Placed(
                bottle=bottle if not drop_bottle and bottle is not None else (0.0, 0.0, 0.0, 0.0),
                label=label if not drop_label and label is not None else (0.0, 0.0, 0.0, 0.0),
                mask=p.mask,
            )
        )
        # помечаем этикетки, которые реально пишем, и маску бутылки сверху
        occ |= p.mask
    kept.reverse()
    # выкинуть полностью пустые
    out: list[Placed] = []
    for p in kept:
        b = clamp_box(p.bottle, w, h)
        l = clamp_box(p.label, w, h) if p.label != (0, 0, 0, 0) else None
        if b is None and l is None:
            continue
        out.append(Placed(bottle=b or (0, 0, 0, 0), label=l or (0, 0, 0, 0), mask=p.mask))
    return out


def add_distractors(canvas: np.ndarray, placed: list[Placed], rng: random.Random) -> None:
    if rng.random() < 0.45:
        draw_shelf_bar(canvas, rng)
    if placed and rng.random() < 0.55:
        p = rng.choice(placed)
        box = p.label if p.label != (0, 0, 0, 0) else p.bottle
        draw_price_tag(
            canvas,
            rng,
            int((box[0] + box[2]) / 2),
            int(box[1] - 8),
        )
    if rng.random() < 0.25:
        # рука/палец — тёмный полупрозрачный эллипс с края
        h, w = canvas.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx = rng.choice([-0.05, 1.05]) * w
        cy = rng.uniform(0.35, 0.85) * h
        rx, ry = w * rng.uniform(0.12, 0.22), h * rng.uniform(0.10, 0.20)
        ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        m = ell < 1.0
        shade = rng.randint(30, 90)
        a = 0.45
        patch = canvas[m].astype(np.float32)
        canvas[m] = np.clip(patch * (1 - a) + shade * a, 0, 255).astype(np.uint8)


@lru_cache(maxsize=80)
def load_bg_rgb(path_str: str) -> np.ndarray:
    im = Image.open(path_str)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    im = im.convert("RGB")
    bw, bh = im.size
    max_bg = 1280
    if max(bw, bh) > max_bg:
        s = max_bg / max(bw, bh)
        im = im.resize(
            (max(1, int(bw * s)), max(1, int(bh * s))), Image.Resampling.BILINEAR
        )
    return np.array(im, dtype=np.uint8, copy=True)


def make_one(
    idx: int,
    specs: list[CutoutSpec],
    backgrounds: list[Background],
    rng: random.Random,
) -> tuple[np.ndarray, list[str]] | None:
    bg = backgrounds[rng.randrange(len(backgrounds))]
    try:
        rgb = load_bg_rgb(str(bg.path)).copy()
    except OSError:
        return None
    rgb = cover_regions(rgb, bg.cover_boxes)
    # весь фон в расфокусе — иначе неразмеченные бутылки полки станут ложными негативами
    h0, w0 = rgb.shape[:2]
    factor = rng.uniform(7.0, 12.0)
    small = Image.fromarray(rgb, "RGB").resize(
        (max(8, int(w0 / factor)), max(8, int(h0 / factor))), Image.Resampling.BILINEAR
    )
    rgb = np.array(small.resize((w0, h0), Image.Resampling.BILINEAR), dtype=np.uint8, copy=True)
    canvas = random_phone_crop(rgb, rng)
    placed = place_scene(canvas, specs, rng)
    if not placed:
        return None
    add_distractors(canvas, placed, rng)
    if rng.random() < 0.35:
        canvas = motion_blur_rgb(
            canvas, ksize=rng.choice([3, 5, 7]), angle=rng.uniform(-25, 25)
        )
    canvas = jpeg_recompress(canvas, quality=rng.randint(48, 88))

    h, w = canvas.shape[:2]
    placed = filter_occluded(placed, w, h)
    lines: list[str] = []
    for p in placed:
        b = clamp_box(p.bottle, w, h)
        l = clamp_box(p.label, w, h) if p.label != (0, 0, 0, 0) else None
        if b is not None:
            lines.append(yolo_line(CLASS_BOTTLE, b, w, h))
        if l is not None:
            lines.append(yolo_line(CLASS_LABEL, l, w, h))
    if not lines:
        return None
    return canvas, lines


def write_data_yaml(out_dir: Path) -> None:
    text = (
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: bottle\n"
        "  1: label\n"
    )
    (out_dir / "data.yaml").write_text(text, encoding="utf-8")


def save_previews(
    out_dir: Path, items: list[tuple[Path, Path]], n: int, rng: random.Random
) -> None:
    if n <= 0 or not items:
        return
    prev = out_dir / "preview"
    prev.mkdir(parents=True, exist_ok=True)
    sample = items if len(items) <= n else rng.sample(items, n)
    colors = {0: (40, 200, 70), 1: (230, 50, 40)}
    for img_p, lbl_p in sample:
        im = Image.open(img_p).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        for line in lbl_p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            draw.rectangle([x1, y1, x2, y2], outline=colors.get(cls, (255, 255, 0)), width=3)
        im.save(prev / img_p.name, quality=90)
    print(f"preview: {len(sample)} → {prev}", flush=True)


def generate(
    catalog_dir: Path,
    my_dir: Path,
    out_dir: Path,
    n: int,
    val_ratio: float,
    seed: int,
    jpeg_quality: int,
    clean: bool,
    preview: int,
    workers: int,
) -> None:
    print("индексация эталонов…", flush=True)
    specs = collect_cutout_specs(catalog_dir)
    if not specs:
        raise SystemExit("Нет эталонов с label_box_2d")
    print("индексация фонов…", flush=True)
    backgrounds = collect_backgrounds(my_dir)
    rng = random.Random(seed)

    if clean and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    if n_val >= n:
        n_val = max(1, n // 10)

    seeds = [rng.randrange(1 << 30) for _ in range(n)]

    def job(i: int) -> tuple[Path, Path] | None:
        one_rng = random.Random(seeds[i])
        result = None
        for k in range(6):
            try:
                result = make_one(i, specs, backgrounds, random.Random(seeds[i] + k * 9973))
            except Exception:
                result = None
            if result is not None:
                break
        if result is None:
            return None
        canvas, lines = result
        split = "val" if i < n_val else "train"
        stem = f"syn_{i:06d}"
        img_p = out_dir / "images" / split / f"{stem}.jpg"
        lbl_p = out_dir / "labels" / split / f"{stem}.txt"
        Image.fromarray(canvas, "RGB").save(img_p, format="JPEG", quality=jpeg_quality)
        lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return img_p, lbl_p

    saved: list[tuple[Path, Path]] = []
    failed = 0
    done = 0
    workers = max(1, workers)
    print(f"генерация {n} кадров (val={n_val}, workers={workers}) → {out_dir}", flush=True)
    if workers == 1:
        for i in range(n):
            pair = job(i)
            if pair is None:
                failed += 1
            else:
                saved.append(pair)
            done += 1
            if done % 50 == 0 or done == n:
                print(f"  {done}/{n} ok={len(saved)} fail={failed}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(job, i): i for i in range(n)}
            for fut in as_completed(futs):
                pair = fut.result()
                if pair is None:
                    failed += 1
                else:
                    saved.append(pair)
                done += 1
                if done % 50 == 0 or done == n:
                    print(f"  {done}/{n} ok={len(saved)} fail={failed}", flush=True)

    saved.sort(key=lambda p: p[0].name)
    write_data_yaml(out_dir)
    save_previews(out_dir, saved, preview, rng)
    n_train = len(list((out_dir / "images" / "train").glob("*.jpg")))
    n_val_real = len(list((out_dir / "images" / "val").glob("*.jpg")))
    print(
        f"done → {out_dir}\n"
        f"  images={len(saved)} train={n_train} val={n_val_real}\n"
        f"  failed={failed}\n"
        f"  data.yaml: {out_dir / 'data.yaml'}",
        flush=True,
    )
    if len(saved) < n:
        print(f"предупреждение: собрано {len(saved)} из {n}", flush=True)


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Synthetic YOLO dataset: catalog bottles on field-photo backgrounds"
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--my-photos", type=Path, default=DEFAULT_MY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=1500, help="Сколько синтетических кадров")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--preview", type=int, default=12)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(2, min(6, os.cpu_count() or 4)),
        help="Потоки генерации",
    )
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n должен быть >= 1")
    if not 0.0 <= args.val_ratio < 1.0:
        raise SystemExit("--val-ratio в [0, 1)")
    generate(
        catalog_dir=args.catalog.resolve(),
        my_dir=args.my_photos.resolve(),
        out_dir=args.out.resolve(),
        n=args.n,
        val_ratio=args.val_ratio,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        clean=args.clean,
        preview=args.preview,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
