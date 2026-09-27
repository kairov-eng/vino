# =============================================================================
# detect_bottle_label.py
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   Локальный тест детекции бутылки/этикетки и OCR текста через Google Gemini
#   (модель gemini-3.6-flash по умолчанию). Без базы данных: берёт все картинки
#   из указанной папки, для каждой вызывает Gemini и сохраняет визуальный
#   результат в подпапку results/ рядом со скриптом.
#
# ЧТО ДЕЛАЕТ ПО ШАГАМ
#   1. Читает ключи GEMINI_API_KEY_1, _2, … и GEMINI_MODEL из .env в этой папке.
#   2. Загружает текстовый промпт из prompt.txt.
#   3. Перебирает изображения в каталоге-аргументе (jpg/png/webp/…/heic).
#   4. Отправляет картинку + промпт в Gemini; замеряет время ответа.
#   5. Парсит JSON ответа:
#        bottle_box_2d  — рамка бутылки [ymin, xmin, ymax, xmax] в шкале 0..1000
#        label_box_2d   — рамка этикетки в той же шкале
#        label_text     — текст с этикетки
#   6. Рисует на копии исходника зелёную (бутылка) и красную (этикетка) рамки,
#      слева сверху пишет время «x.xxx сек» и считанный текст.
#   7. Сохраняет файл:
#        results/<имя>_[x.xxx]_YYYYMMDD_HHMMSS.<ext>
#
# ЗАВИСИМОСТИ / КОНФИГ
#   .env          — ключи Gemini, опционально GEMINI_MODEL
#   prompt.txt    — промпт для модели
#   Пакеты: google-genai, Pillow, python-dotenv
#
# ПРИМЕРЫ ВЫЗОВА
#   python detect_bottle_label.py C:\dev\Vino2026\temp
#   python detect_bottle_label.py .\images
#   python detect_bottle_label.py "D:\photos\wine"
#
# ПРИМЕЧАНИЕ
#   Этот же файл — библиотека функций (call_gemini_one_key, normalize_box,
#   box_to_pixels, load_api_keys, …) для скриптов 2 и 3:
#     from detect_bottle_label import …
#
# =============================================================================

"""Детекция бутылки/этикетки и OCR текста этикетки через Gemini 3.6 Flash."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PROMPT_PATH = SCRIPT_DIR / "prompt.txt"


def resolve_env_path() -> Path:
    """Find .env: crop/ → Подготовка данных/ → backend/.env."""
    here = Path(__file__).resolve().parent
    for c in (
        here / ".env",
        here.parent / ".env",
        here.parents[1] / "backend" / ".env",
    ):
        if c.is_file():
            return c
    return here / ".env"


ENV_PATH = resolve_env_path()

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


def load_api_keys() -> list[str]:
    load_dotenv(ENV_PATH)
    import os

    keys: list[str] = []
    i = 1
    while True:
        val = (os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()
        if not val:
            break
        keys.append(val)
        i += 1
    legacy = (os.getenv("GEMINI_API_KEY") or "").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    if not keys:
        raise SystemExit(f"Не найдены GEMINI_API_KEY_* в {ENV_PATH}")
    return keys


def load_model_name() -> str:
    import os

    return (os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").strip()


def load_prompt() -> str:
    if not PROMPT_PATH.is_file():
        raise SystemExit(f"Не найден промпт: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"Каталог не найден: {directory}")
    files = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not files:
        raise SystemExit(f"В каталоге нет изображений: {directory}")
    return files


def mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".heic", ".heif"}:
        return "image/heic"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/jpeg"


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_box(box) -> list[int] | None:
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    return vals


def box_to_pixels(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    """[ymin, xmin, ymax, xmax] 0..1000 → (x1, y1, x2, y2) в пикселях."""
    ymin, xmin, ymax, xmax = box
    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)
    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _usage_tokens(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    tin = getattr(usage, "prompt_token_count", None)
    tout = getattr(usage, "candidates_token_count", None)
    if tin is None:
        tin = getattr(usage, "promptTokenCount", None)
    if tout is None:
        tout = getattr(usage, "candidatesTokenCount", None)
    try:
        tin_i = int(tin) if tin is not None else None
    except (TypeError, ValueError):
        tin_i = None
    try:
        tout_i = int(tout) if tout is not None else None
    except (TypeError, ValueError):
        tout_i = None
    return tin_i, tout_i


def call_gemini_one_key(
    image_path: Path,
    prompt: str,
    model: str,
    api_key: str,
    *,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
) -> tuple[dict, float, int | None, int | None]:
    """Один запрос к Gemini (AI Studio или Vertex). Возвращает data, sec, tokens_in, tokens_out."""
    from gemini_client import generate_json, make_genai_client

    payload_bytes = image_bytes if image_bytes is not None else image_path.read_bytes()
    mime = mime_type or mime_for(image_path)
    client = make_genai_client(api_key=api_key, model=model)
    t0 = time.perf_counter()
    response = generate_json(
        client,
        model=model,
        contents=[
            types.Part.from_bytes(data=payload_bytes, mime_type=mime),
            prompt,
        ],
    )
    elapsed = time.perf_counter() - t0
    tokens_in, tokens_out = _usage_tokens(response)
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("Пустой ответ модели")
    data = extract_json(raw)
    return data, elapsed, tokens_in, tokens_out


def call_gemini(
    image_path: Path,
    prompt: str,
    model: str,
    api_keys: list[str],
) -> tuple[dict, float]:
    last_error: Exception | None = None

    for key in api_keys:
        t0 = time.perf_counter()
        try:
            data, elapsed, _tin, _tout = call_gemini_one_key(
                image_path, prompt, model, key
            )
            return data, elapsed
        except Exception as exc:  # noqa: BLE001 — перебор ключей
            last_error = exc
            elapsed = time.perf_counter() - t0
            print(f"  ! key ...{key[-6:]} failed in {elapsed:.3f}s: {exc}")
            continue

    raise RuntimeError(f"All Gemini keys failed. Last error: {last_error}")


def annotate_image(
    image_path: Path,
    data: dict,
    elapsed: float,
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bottle = normalize_box(data.get("bottle_box_2d"))
    label = normalize_box(data.get("label_box_2d"))
    label_text = str(data.get("label_text") or "").strip()

    line_w = max(2, min(width, height) // 250)

    if bottle:
        x1, y1, x2, y2 = box_to_pixels(bottle, width, height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 80, 255), width=line_w)

    if label:
        x1, y1, x2, y2 = box_to_pixels(label, width, height)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60, 255), width=line_w)

    # Текст слева сверху вниз: время, затем OCR
    font_size = max(14, min(width, height) // 45)
    font = pick_font(font_size)
    lines = [f"{elapsed:.3f} сек"]
    if label_text:
        lines.extend(label_text.splitlines() or [label_text])
    else:
        lines.append("(текст этикетки пуст)")

    pad = max(8, font_size // 2)
    line_gap = max(4, font_size // 5)
    text_block = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), text_block, font=font, spacing=line_gap)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    panel_w = min(width, tw + pad * 2)
    panel_h = min(height, th + pad * 2)
    draw.rectangle([0, 0, panel_w, panel_h], fill=(0, 0, 0, 160))
    draw.multiline_text(
        (pad, pad),
        text_block,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=line_gap,
    )

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = 92
    composed.save(out_path, **save_kwargs)


def result_filename(src: Path, elapsed: float) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{src.stem}_[{elapsed:.3f}]_{stamp}{src.suffix.lower()}"


def process_one(
    image_path: Path,
    prompt: str,
    model: str,
    api_keys: list[str],
) -> Path:
    print(f"> {image_path.name}")
    data, elapsed = call_gemini(image_path, prompt, model, api_keys)
    out_name = result_filename(image_path, elapsed)
    out_path = RESULTS_DIR / out_name
    annotate_image(image_path, data, elapsed, out_path)
    print(f"  Gemini: {elapsed:.3f} sec")
    print(f"  bottle_box_2d: {data.get('bottle_box_2d')}")
    print(f"  label_box_2d:  {data.get('label_box_2d')}")
    preview = str(data.get("label_text") or "").replace("\n", " / ")
    if len(preview) > 120:
        preview = preview[:117] + "..."
    print(f"  label_text:    {preview}")
    print(f"  saved:         {out_path}")
    return out_path


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Детекция бутылки/этикетки и OCR через Gemini 3.6 Flash"
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Каталог с исходными изображениями",
    )
    args = parser.parse_args(argv)

    api_keys = load_api_keys()
    model = load_model_name()
    prompt = load_prompt()
    images = list_images(args.images_dir.resolve())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Модель: {model}")
    print(f"Ключей: {len(api_keys)}")
    print(f"Картинок: {len(images)}")
    print(f"Результаты: {RESULTS_DIR}")
    print()

    ok = 0
    for path in images:
        try:
            process_one(path, prompt, model, api_keys)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}", file=sys.stderr)
        print()

    print(f"Готово: {ok}/{len(images)}")
    return 0 if ok == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
