# =============================================================================
# 8.prepare_yolo_dataset.py
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   Собирает РЕАЛЬНЫЙ YOLO-датасет (0=bottle, 1=label) из полевых фото и
#   разметки Gemini в cropjson/ (результат скрипта 3.crop_heic_folder).
#
# ВХОД (--uploads, по умолчанию C:\dev\Vino2026\uploads.my)
#   <uploads>/*.HEIC|jpg|png|webp     — исходные фото (для полного экспорта)
#   <uploads>/cropjson/*.json         — разметка:
#       status == 1, bottle_box_2d / label_box_2d [ymin,xmin,ymax,xmax] 0..1000
#       имена: IMG.HEIC.json или IMG.HEIC_1.json, _2, … (несколько объектов)
#   <uploads>/yolo/images/{train,val} — уже сконвертированные JPG (--labels-only)
#
# РЕЖИМЫ
#   --labels-only (по умолчанию) — не трогает images/, только пишет labels/
#                                  и обновляет data.yaml по существующему split.
#   --full                       — конвертирует HEIC→JPEG, заново режет train/val.
#
# ВЫХОД
#   <uploads>/yolo/
#     images/train|val/*.jpg
#     labels/train|val/*.txt   — строки "cls xc yc w h", cls: 0=bottle, 1=label
#     data.yaml
#
# ПРИМЕРЫ
#   python "8.prepare_yolo_dataset.py"
#   python "8.prepare_yolo_dataset.py" --labels-only
#   python "8.prepare_yolo_dataset.py" --full --val-ratio 0.2 --seed 42
#
# СВЯЗЬ
#   3 → cropjson/; 5.-x → синтетика с теми же классами; 8 → реальный сет.
#
# =============================================================================

"""Подготовка YOLO-датасета (bottle + label) из uploads.my."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
_MONOREPO = Path(__file__).resolve().parents[2]  # Vino2026
DEFAULT_UPLOADS = _MONOREPO / "uploads.my"
HEIC_EXTS = {".heic", ".heif"}
IMAGE_EXTS = HEIC_EXTS | {".jpg", ".jpeg", ".png", ".webp"}
CLASS_BOTTLE = 0
CLASS_LABEL = 1
_heif_ready = False

JSON_NAME_RE = re.compile(
    r"^(?P<stem>.+?\.(?:heic|heif|jpg|jpeg|png|webp))(?:_(?P<idx>\d+))?\.json$",
    re.IGNORECASE,
)

# Одна запись: (class_id, gemini_box)
BoxRec = tuple[int, list[int]]
YoloRec = tuple[int, float, float, float, float]


def ensure_heif_support() -> None:
    global _heif_ready
    if _heif_ready:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError as exc:
        raise SystemExit(
            "Нужен пакет pillow-heif. Установите: pip install pillow-heif"
        ) from exc
    _heif_ready = True


def open_image(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() in HEIC_EXTS:
        ensure_heif_support()
    img = Image.open(image_path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def normalize_box(box) -> list[int] | None:
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None


def gemini_to_yolo(box: list[int]) -> tuple[float, float, float, float] | None:
    """[ymin, xmin, ymax, xmax] 0..1000 → (xc, yc, w, h) 0..1."""
    ymin, xmin, ymax, xmax = (v / 1000.0 for v in box)
    if xmax <= xmin or ymax <= ymin:
        return None
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    if w < 1e-4 or h < 1e-4:
        return None
    return xc, yc, w, h


def parse_source_name(json_path: Path, data: dict) -> str | None:
    for key in ("source_file", "image_file"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return Path(val.strip()).name
    m = JSON_NAME_RE.match(json_path.name)
    if m:
        return m.group("stem")
    return None


def collect_boxes_by_image(cropjson_dir: Path) -> dict[str, list[BoxRec]]:
    """source_filename → [(cls, gemini_box), ...] для bottle и label."""
    by_image: dict[str, list[BoxRec]] = defaultdict(list)
    skipped = 0
    n_bottle = 0
    n_label = 0

    for jp in sorted(cropjson_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        if data.get("status") != 1:
            skipped += 1
            continue

        source = parse_source_name(jp, data)
        if not source:
            skipped += 1
            continue

        bottle = normalize_box(data.get("bottle_box_2d"))
        label = normalize_box(data.get("label_box_2d"))
        if bottle is None and label is None:
            skipped += 1
            continue

        if bottle is not None:
            by_image[source].append((CLASS_BOTTLE, bottle))
            n_bottle += 1
        if label is not None:
            by_image[source].append((CLASS_LABEL, label))
            n_label += 1

    for name, boxes in list(by_image.items()):
        uniq: list[BoxRec] = []
        seen: set[tuple[int, int, int, int, int]] = set()
        for cls, b in boxes:
            key = (cls, b[0], b[1], b[2], b[3])
            if key in seen:
                continue
            seen.add(key)
            uniq.append((cls, b))
        by_image[name] = uniq

    print(
        f"JSON: images={len(by_image)} bottle_boxes={n_bottle} "
        f"label_boxes={n_label} skipped_json~={skipped}"
    )
    return dict(by_image)


def to_yolo_recs(boxes: list[BoxRec]) -> list[YoloRec]:
    out: list[YoloRec] = []
    for cls, box in boxes:
        yb = gemini_to_yolo(box)
        if yb is None:
            continue
        xc, yc, w, h = yb
        out.append((cls, xc, yc, w, h))
    return out


def find_image(uploads_dir: Path, name: str) -> Path | None:
    p = uploads_dir / name
    if p.is_file():
        return p
    lower = name.lower()
    for cand in uploads_dir.iterdir():
        if cand.is_file() and cand.name.lower() == lower:
            return cand
    return None


def source_stem(source_name: str) -> str:
    return Path(source_name).stem


def write_label_file(path: Path, recs: list[YoloRec]) -> None:
    lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in recs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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


def index_boxes_by_stem(
    boxes_by_image: dict[str, list[BoxRec]],
) -> dict[str, list[BoxRec]]:
    """IMG_3347.HEIC → stem IMG_3347 (для матча с JPG)."""
    by_stem: dict[str, list[BoxRec]] = defaultdict(list)
    for source_name, boxes in boxes_by_image.items():
        by_stem[source_stem(source_name)].extend(boxes)
    # дедуп после склейки
    for stem, boxes in list(by_stem.items()):
        uniq: list[BoxRec] = []
        seen: set[tuple[int, int, int, int, int]] = set()
        for cls, b in boxes:
            key = (cls, b[0], b[1], b[2], b[3])
            if key in seen:
                continue
            seen.add(key)
            uniq.append((cls, b))
        by_stem[stem] = uniq
    return dict(by_stem)


def prepare_labels_only(uploads_dir: Path, out_dir: Path) -> None:
    """Пишет labels/ под уже существующие images/{train,val}/*.jpg."""
    cropjson_dir = uploads_dir / "cropjson"
    if not cropjson_dir.is_dir():
        raise SystemExit(f"Нет папки cropjson: {cropjson_dir}")

    images_root = out_dir / "images"
    if not (images_root / "train").is_dir() and not (images_root / "val").is_dir():
        raise SystemExit(
            f"Нет images/train|val в {out_dir}. Сначала --full или положите JPG."
        )

    boxes_by_stem = index_boxes_by_stem(collect_boxes_by_image(cropjson_dir))

    stats = {
        "train_img": 0,
        "val_img": 0,
        "train_bottle": 0,
        "train_label": 0,
        "val_bottle": 0,
        "val_label": 0,
        "no_ann": 0,
    }

    for split in ("train", "val"):
        img_dir = images_root / split
        lbl_dir = out_dir / "labels" / split
        if not img_dir.is_dir():
            continue
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_dir.glob("*")):
            if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            boxes = boxes_by_stem.get(img.stem)
            if not boxes:
                stats["no_ann"] += 1
                print(f"  no annotation for {img.name}")
                continue
            recs = to_yolo_recs(boxes)
            if not recs:
                stats["no_ann"] += 1
                continue
            write_label_file(lbl_dir / f"{img.stem}.txt", recs)
            stats[f"{split}_img"] += 1
            stats[f"{split}_bottle"] += sum(1 for c, *_ in recs if c == CLASS_BOTTLE)
            stats[f"{split}_label"] += sum(1 for c, *_ in recs if c == CLASS_LABEL)

    write_data_yaml(out_dir)
    print(
        f"done (labels-only) -> {out_dir}\n"
        f"  images labeled train={stats['train_img']} val={stats['val_img']}\n"
        f"  boxes train bottle={stats['train_bottle']} label={stats['train_label']}\n"
        f"  boxes val   bottle={stats['val_bottle']} label={stats['val_label']}\n"
        f"  no_ann={stats['no_ann']}\n"
        f"  data.yaml: {out_dir / 'data.yaml'}"
    )


def prepare_full(
    uploads_dir: Path,
    out_dir: Path,
    val_ratio: float,
    seed: int,
    jpeg_quality: int,
    clean: bool,
) -> None:
    cropjson_dir = uploads_dir / "cropjson"
    if not cropjson_dir.is_dir():
        raise SystemExit(f"Нет папки cropjson: {cropjson_dir}")

    boxes_by_image = collect_boxes_by_image(cropjson_dir)
    if not boxes_by_image:
        raise SystemExit("Нет валидных разметок")

    items: list[tuple[Path, list[YoloRec]]] = []
    missing = 0
    for name, boxes in sorted(boxes_by_image.items()):
        img_path = find_image(uploads_dir, name)
        if img_path is None:
            missing += 1
            print(f"  missing image: {name}")
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            missing += 1
            continue
        recs = to_yolo_recs(boxes)
        if not recs:
            continue
        items.append((img_path, recs))

    if not items:
        raise SystemExit("После фильтрации не осталось изображений")

    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    if n_val >= len(items):
        n_val = max(1, len(items) // 5) if len(items) >= 5 else 1
        if n_val >= len(items):
            n_val = 0
    train_items = items[n_val:]
    val_items = items[:n_val]

    if clean and out_dir.exists():
        # не удаляем весь out, если там чужие zip — чистим только images/labels
        for sub in ("images", "labels"):
            p = out_dir / sub
            if p.exists():
                shutil.rmtree(p)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def export(split: str, pair: tuple[Path, list[YoloRec]]) -> None:
        src, recs = pair
        stem = src.stem
        out_img = out_dir / "images" / split / f"{stem}.jpg"
        out_lbl = out_dir / "labels" / split / f"{stem}.txt"
        img = open_image(src)
        img.save(out_img, format="JPEG", quality=jpeg_quality)
        write_label_file(out_lbl, recs)

    print(f"export: train={len(train_items)} val={len(val_items)} (missing_images={missing})")
    for pair in train_items:
        export("train", pair)
    for pair in val_items:
        export("val", pair)

    write_data_yaml(out_dir)

    def count_cls(pairs: list[tuple[Path, list[YoloRec]]], cls: int) -> int:
        return sum(1 for _, recs in pairs for c, *_ in recs if c == cls)

    print(
        f"done (full) -> {out_dir}\n"
        f"  images train={len(train_items)} val={len(val_items)}\n"
        f"  boxes train bottle={count_cls(train_items, CLASS_BOTTLE)} "
        f"label={count_cls(train_items, CLASS_LABEL)}\n"
        f"  boxes val   bottle={count_cls(val_items, CLASS_BOTTLE)} "
        f"label={count_cls(val_items, CLASS_LABEL)}\n"
        f"  data.yaml: {out_dir / 'data.yaml'}"
    )


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset (bottle+label) from cropjson"
    )
    parser.add_argument("--uploads", type=Path, default=DEFAULT_UPLOADS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        default=True,
        help="Только labels/ по существующим images/ (по умолчанию)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Полный экспорт: конвертация фото + новый train/val split",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no-clean", action="store_false", dest="clean")
    args = parser.parse_args()

    uploads = args.uploads.resolve()
    out = (args.out or (uploads / "yolo")).resolve()

    print(f"uploads={uploads}")
    print(f"out={out}")

    if args.full:
        if not 0.0 <= args.val_ratio < 1.0:
            raise SystemExit("--val-ratio должен быть в [0, 1)")
        prepare_full(
            uploads_dir=uploads,
            out_dir=out,
            val_ratio=args.val_ratio,
            seed=args.seed,
            jpeg_quality=args.jpeg_quality,
            clean=args.clean,
        )
    else:
        prepare_labels_only(uploads_dir=uploads, out_dir=out)


if __name__ == "__main__":
    main()
