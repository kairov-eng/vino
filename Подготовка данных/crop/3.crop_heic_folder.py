# =============================================================================
# 3.crop_heic_folder.py
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   Crop этикеток с фото iPhone (HEIC/HEIF) из папки — БЕЗ SQL и без таблицы
#   wines. Обрабатывает только файлы, для которых ещё нет JSON в cropjson/.
#   Умеет несколько бутылок на одном кадре: каждая этикетка → отдельный
#   json + png, нумерация _1, _2, … от большей рамки к меньшей.
#
# КАК ВЫБИРАЕТСЯ ОЧЕРЕДЬ
#   Сканирует uploads/ на *.heic / *.heif.
#   Файл считается уже обработанным, если есть:
#     cropjson/<имя.HEIC>.json      (legacy, один файл)
#     или cropjson/<имя.HEIC>_1.json
#   Иначе попадает в очередь (не больше --limit за батч).
#
# ВЫЗОВ GEMINI
#   HEIC открывается через pillow-heif, конвертируется в JPEG (макс. сторона
#   2048) и уходит в модель. Промпт — отдельный prompt_heic.txt: просит
#   массив items[] (бутылка + этикетка + текст на каждую).
#   Ответ парсится: items[] либо одиночный объект (совместимость).
#   Этикетки с валидным label_box_2d сортируются по площади рамки ↓.
#
# ЧТО ПИШЕТ НА ДИСК (относительно папки с фото)
#   При N этикетках (N≥1), индекс i = 1..N:
#     cropjson/<имя.HEIC>_1.json
#     crop/<имя.HEIC>_1_crop.png
#   Если рамок несколько — сохраняется только primary (largest /
#   middle_of_three_similar), как в findwine select_primary_label.
#   В JSON одной этикетки: label_index, labels_total, боксы, текст, токены,
#   status. Поле gemini_raw:
#     — всегда, если этикеток > 1
#     — при одной этикетке — если нет текста (или нет рамки / status=2)
#   Нет ни одной рамки (status=2): *_1.json + *_1_crop.png = исходник целиком.
#   Ошибка (status=-1): пишется *_1.json с error, чтобы файл не крутился снова.
#
# РЕЖИМ --file
#   Обработать один файл для теста (абсолютный путь или имя внутри uploads).
#   Допускаются также jpg/jpeg/png/webp. Старые json этого файла в cropjson/
#   удаляются перед прогоном. Папка uploads для crop/cropjson — родитель файла.
#
# ПАРАЛЛЕЛИЗМ И ЛИМИТЫ
#   Число потоков = числу GEMINI_API_KEY_* в .env.
#   На ключ ≤ 15 запросов / 60 с (KeyRateLimiter).
#
# РЕЖИМ --infinitive 1
#   Повторное сканирование папки, пока есть HEIC без json; стоп при пустой
#   очереди или том же наборе имён (защита от цикла). С --file не используется.
#
# ЛОГИРОВАНИЕ
#   Префикс [YYYY-MM-DD HH:MM:SS] в терминале и в
#   logs/crop_heic_folder_YYYYMMDD_HHMMSS.log.
#
# ЗАВИСИМОСТИ / КОНФИГ
#   .env             — GEMINI_API_KEY_*, GEMINI_MODEL
#   prompt_heic.txt  — промпт с поддержкой нескольких бутылок
#   Функции Gemini   — из detect_bottle_label.py
#   Пакеты: google-genai, Pillow, pillow-heif, python-dotenv
#
# ПРИМЕРЫ ВЫЗОВА
#   python "3.crop_heic_folder.py" C:\photos\iphone
#   python "3.crop_heic_folder.py" C:\photos\iphone --limit 50
#   python "3.crop_heic_folder.py" C:\photos\iphone --infinitive 1
#   python "3.crop_heic_folder.py" C:\photos\iphone --file IMG_1234.HEIC
#   python "3.crop_heic_folder.py" --file C:\photos\iphone\IMG_1234.HEIC
#
# =============================================================================

"""Crop этикеток с HEIC-фото (iPhone) из папки, без обращения к БД."""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
PROMPT_HEIC_PATH = SCRIPT_DIR / "prompt_heic.txt"
sys.path.insert(0, str(SCRIPT_DIR))
from detect_bottle_label import (  # noqa: E402
    ENV_PATH,
    box_to_pixels,
    call_gemini_one_key,
    load_api_keys,
    load_model_name,
    normalize_box,
)

MAX_RPM_PER_KEY = 15
RPM_WINDOW_SEC = 60.0
HEIC_EXTS = {".heic", ".heif"}
# --file также принимает обычные форматы для теста
FILE_EXTS = HEIC_EXTS | {".jpg", ".jpeg", ".png", ".webp"}

_print_lock = threading.Lock()
_log_file: Path | None = None
_heif_ready = False


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def init_logging() -> Path:
    global _log_file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = LOGS_DIR / f"crop_heic_folder_{stamp}.log"
    with _log_file.open("a", encoding="utf-8") as f:
        f.write(
            f"=== crop_heic_folder start "
            f"{datetime.now().isoformat(sep=' ', timespec='seconds')} ===\n"
        )
    return _log_file


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    with _print_lock:
        print(line, flush=True)
        if _log_file is not None:
            try:
                with _log_file.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


def load_heic_prompt() -> str:
    if not PROMPT_HEIC_PATH.is_file():
        raise SystemExit(f"Не найден промпт: {PROMPT_HEIC_PATH}")
    return PROMPT_HEIC_PATH.read_text(encoding="utf-8").strip()


def ensure_heif_support() -> None:
    global _heif_ready
    if _heif_ready:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError as exc:
        raise SystemExit(
            "Нужен пакет pillow-heif для HEIC. Установите:\n"
            "  pip install pillow-heif"
        ) from exc
    _heif_ready = True


class KeyRateLimiter:
    def __init__(
        self,
        key_label: str,
        max_rpm: int = MAX_RPM_PER_KEY,
        window_sec: float = RPM_WINDOW_SEC,
    ) -> None:
        self.key_label = key_label
        self.max_rpm = max_rpm
        self.window_sec = window_sec
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.window_sec:
                    self._times.popleft()
                if len(self._times) < self.max_rpm:
                    self._times.append(now)
                    return
                wait = self.window_sec - (now - self._times[0]) + 0.01
            log(
                f"  [rate {self.key_label}] {self.max_rpm} req / {self.window_sec:.0f}s — "
                f"пауза {wait:.1f}s"
            )
            time.sleep(max(wait, 0.05))


def json_path_for(image: Path, cropjson_dir: Path, index: int | None = None) -> Path:
    if index is None:
        return cropjson_dir / f"{image.name}.json"
    return cropjson_dir / f"{image.name}_{index}.json"


def crop_path_for(image: Path, crop_dir: Path, index: int) -> Path:
    return crop_dir / f"{image.name}_{index}_crop.png"


def already_processed(image: Path, cropjson_dir: Path) -> bool:
    """Считаем обработанным, если есть legacy .json или любой _N.json."""
    if json_path_for(image, cropjson_dir).is_file():
        return True
    if json_path_for(image, cropjson_dir, 1).is_file():
        return True
    return False


def list_pending_heic(uploads_dir: Path, cropjson_dir: Path, limit: int) -> list[Path]:
    pending: list[Path] = []
    for p in sorted(uploads_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in HEIC_EXTS:
            continue
        if already_processed(p, cropjson_dir):
            continue
        pending.append(p)
        if len(pending) >= limit:
            break
    return pending


def open_image(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() in HEIC_EXTS:
        ensure_heif_support()
    img = Image.open(image_path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def image_to_jpeg_bytes(image_path: Path, max_side: int = 2048) -> bytes:
    img = open_image(image_path)
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def box_area(box: list[int] | None) -> float:
    if not box or len(box) != 4:
        return 0.0
    ymin, xmin, ymax, xmax = box
    return max(0, ymax - ymin) * max(0, xmax - xmin)


def parse_label_items(data: dict) -> list[dict]:
    """Достаёт список этикеток из ответа Gemini (items[] или одиночный объект)."""
    items: list[dict] = []
    if not isinstance(data, dict):
        return items

    raw_list = data.get("items")
    if isinstance(raw_list, list):
        for it in raw_list:
            if isinstance(it, dict):
                items.append(it)
    elif normalize_box(data.get("label_box_2d")) is not None or data.get("label_text") is not None:
        items.append(
            {
                "bottle_box_2d": data.get("bottle_box_2d"),
                "label_box_2d": data.get("label_box_2d"),
                "label_text": data.get("label_text"),
            }
        )

    # только с валидной рамкой этикетки; без рамки отфильтруем позже
    normalized: list[dict] = []
    for it in items:
        box = normalize_box(it.get("label_box_2d"))
        entry = {
            "bottle_box_2d": normalize_box(it.get("bottle_box_2d")),
            "label_box_2d": box,
            "label_text": it.get("label_text"),
            "_area": box_area(box),
        }
        normalized.append(entry)

    # от большей этикетки к меньшей
    normalized.sort(key=lambda x: x["_area"], reverse=True)
    return normalized


def select_primary_label_item(
    with_box: list[dict],
    *,
    similar_ratio: float = 0.12,
) -> dict:
    """Одна этикетка для crop: largest; при 3 похожих по площади — средняя."""
    if not with_box:
        raise ValueError("with_box пуст")
    ordered = list(with_box)
    if len(ordered) == 3:
        areas = [float(it.get("_area") or 0.0) for it in ordered]
        amin, amax = min(areas), max(areas)
        if amin > 0 and (amax - amin) / amax <= similar_ratio:
            pick = dict(ordered[1])
            pick["select_reason"] = "middle_of_three_similar"
            return pick
    pick = dict(ordered[0])
    pick["select_reason"] = "largest"
    return pick


def crop_by_box(image_path: Path, box: list[int], out_path: Path) -> None:
    img = open_image(image_path)
    width, height = img.size
    x1, y1, x2, y2 = box_to_pixels(box, width, height)
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    cropped = img.crop((x1, y1, x2, y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.convert("RGB").save(out_path, format="PNG")


def save_original_as_crop(image_path: Path, out_path: Path) -> None:
    img = open_image(image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, format="PNG")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_image(
    image_path: Path,
    *,
    cropjson_dir: Path,
    crop_dir: Path,
    prompt: str,
    model: str,
    api_key: str,
    key_label: str,
    rate_limiter: KeyRateLimiter,
) -> str:
    try:
        jpeg_bytes = image_to_jpeg_bytes(image_path)
        rate_limiter.acquire()
        data, elapsed, tokens_in, tokens_out = call_gemini_one_key(
            image_path,
            prompt,
            model,
            api_key,
            image_bytes=jpeg_bytes,
            mime_type="image/jpeg",
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"Ожидался JSON-объект, получено: {type(data)}")

        items = parse_label_items(data)
        with_box = [it for it in items if it.get("label_box_2d")]
        multi = len(with_box) > 1

        # Нет ни одной рамки этикетки
        if not with_box:
            raw_text = None
            if items:
                t = items[0].get("label_text")
                raw_text = t if t is not None else data.get("label_text")
            label_text = str(raw_text).strip() if raw_text else None
            if label_text == "":
                label_text = None

            payload = {
                "image_file": image_path.name,
                "source_file": image_path.name,
                "label_index": 1,
                "labels_total": 0,
                "elapsed_sec": round(elapsed, 3),
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "bottle_box_2d": None,
                "label_box_2d": None,
                "label_text": raw_text,
                "status": 2,
                "gemini_raw": data,
            }
            jp = json_path_for(image_path, cropjson_dir, 1)
            cp = crop_path_for(image_path, crop_dir, 1)
            save_original_as_crop(image_path, cp)
            write_json(jp, payload)
            return (
                f"no-label [{key_label}] Gemini {elapsed:.3f}s "
                f"in={tokens_in} out={tokens_out} | {image_path.name} | "
                f"json={jp.name} original->crop={cp.name} -> status=2"
            )

        total = len(with_box)
        it = select_primary_label_item(with_box)
        raw_text = it.get("label_text")
        if raw_text is None:
            label_text = None
        else:
            label_text = str(raw_text).strip() or None

        payload = {
            "image_file": image_path.name,
            "source_file": image_path.name,
            "label_index": 1,
            "labels_total": total,
            "labels_detected": total,
            "select_reason": it.get("select_reason") or "largest",
            "elapsed_sec": round(elapsed, 3),
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "bottle_box_2d": it.get("bottle_box_2d"),
            "label_box_2d": it.get("label_box_2d"),
            "label_text": raw_text,
            "status": 1,
        }
        # Несколько этикеток — всегда gemini_raw; одна — только если нет текста
        if multi or label_text is None:
            payload["gemini_raw"] = data

        jp = json_path_for(image_path, cropjson_dir, 1)
        cp = crop_path_for(image_path, crop_dir, 1)
        crop_by_box(image_path, it["label_box_2d"], cp)
        write_json(jp, payload)
        reason = it.get("select_reason") or "largest"
        return (
            f"ok [{key_label}] Gemini {elapsed:.3f}s "
            f"in={tokens_in} out={tokens_out} | {image_path.name} | "
            f"labels_detected={total} primary={reason} | "
            f"{jp.name}/{cp.name} -> status=1"
        )
    except Exception as exc:  # noqa: BLE001
        err_payload = {
            "image_file": image_path.name,
            "source_file": image_path.name,
            "label_index": 1,
            "labels_total": 0,
            "status": -1,
            "error": str(exc),
        }
        jp = json_path_for(image_path, cropjson_dir, 1)
        try:
            write_json(jp, err_payload)
        except Exception as write_exc:  # noqa: BLE001
            return f"error [{key_label}]: {exc}; json write failed: {write_exc} -> status=-1"
        return f"error [{key_label}]: {exc} -> status=-1"


def run_batch(
    *,
    files: list[Path],
    api_keys: list[str],
    cropjson_dir: Path,
    crop_dir: Path,
    prompt: str,
    model: str,
    stats: dict[str, int],
    stats_lock: threading.Lock,
    rate_limiters: list[KeyRateLimiter],
) -> None:
    task_queue = list(files)
    queue_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def _worker(api_key: str, key_index: int, limiter: KeyRateLimiter) -> None:
        key_label = f"key{key_index + 1}/...{api_key[-6:]}"
        while True:
            with queue_lock:
                if not task_queue:
                    return
                image_path = task_queue.pop(0)

            log(f"> {image_path.name} [{key_label}]")
            msg = process_image(
                image_path,
                cropjson_dir=cropjson_dir,
                crop_dir=crop_dir,
                prompt=prompt,
                model=model,
                api_key=api_key,
                key_label=key_label,
                rate_limiter=limiter,
            )
            log(f"  {msg}")
            with stats_lock:
                if "-> status=1" in msg:
                    stats["ok"] += 1
                elif "-> status=2" in msg:
                    stats["no_label"] += 1
                else:
                    stats["error"] += 1

    for i, key in enumerate(api_keys):
        t = threading.Thread(
            target=_worker,
            name=f"heic-worker-{i + 1}",
            args=(key, i, rate_limiters[i]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def resolve_single_file(uploads_dir: Path | None, file_arg: Path) -> Path:
    """--file: абсолютный путь или имя относительно uploads."""
    p = file_arg
    if not p.is_file() and uploads_dir is not None:
        cand = uploads_dir / file_arg.name
        if cand.is_file():
            p = cand
    p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"Файл не найден: {file_arg}")
    if p.suffix.lower() not in FILE_EXTS:
        raise SystemExit(
            f"Неподдерживаемое расширение {p.suffix}. Ожидается: {', '.join(sorted(FILE_EXTS))}"
        )
    return p


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    log_path = init_logging()
    parser = argparse.ArgumentParser(
        description="Crop этикеток с HEIC из папки (без SQL), потоки = число ключей Gemini"
    )
    parser.add_argument(
        "uploads",
        type=Path,
        nargs="?",
        default=None,
        help="Папка с HEIC-файлами (внутри создаются crop/ и cropjson/)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Один файл для теста (вместо сканирования папки). Путь или имя в uploads",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Сколько файлов взять за один проход (по умолчанию 100)",
    )
    parser.add_argument(
        "--infinitive",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = повторять сканирование, пока есть HEIC без json",
    )
    args = parser.parse_args(argv)

    if args.uploads is None and args.file is None:
        raise SystemExit("Укажите папку uploads и/или --file путь_к_картинке")

    load_dotenv(ENV_PATH)

    if args.file is not None:
        single = resolve_single_file(
            args.uploads.resolve() if args.uploads else None,
            args.file,
        )
        uploads_dir = single.parent
        if single.suffix.lower() in HEIC_EXTS:
            ensure_heif_support()
    else:
        uploads_dir = args.uploads.resolve()
        if not uploads_dir.is_dir():
            raise SystemExit(f"Каталог не найден: {uploads_dir}")
        ensure_heif_support()
        single = None

    cropjson_dir = uploads_dir / "cropjson"
    crop_dir = uploads_dir / "crop"
    cropjson_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    api_keys = load_api_keys()
    if not api_keys:
        raise SystemExit("Нет GEMINI_API_KEY_*")
    model = load_model_name()
    prompt = load_heic_prompt()

    log(f"Лог-файл: {log_path}")
    log(f"Модель: {model}")
    log(f"Промпт: {PROMPT_HEIC_PATH.name}")
    log(f"Uploads: {uploads_dir}")
    if single is not None:
        log(f"Файл: {single}")
    log(f"Limit: {args.limit}")
    log(f"Infinitive: {0 if single else args.infinitive}")
    log(f"Потоков/ключей: {len(api_keys)}, лимит {MAX_RPM_PER_KEY} req/min на ключ")
    log("")

    stats = {"ok": 0, "no_label": 0, "error": 0}
    stats_lock = threading.Lock()
    rate_limiters = [
        KeyRateLimiter(f"key{i + 1}/...{k[-6:]}") for i, k in enumerate(api_keys)
    ]

    if single is not None:
        # Для теста перезаписываем: удаляем старые json/_N.json этого файла
        for old in cropjson_dir.glob(f"{single.name}*.json"):
            try:
                old.unlink()
            except OSError:
                pass
        run_batch(
            files=[single],
            api_keys=api_keys,
            cropjson_dir=cropjson_dir,
            crop_dir=crop_dir,
            prompt=prompt,
            model=model,
            stats=stats,
            stats_lock=stats_lock,
            rate_limiters=rate_limiters,
        )
        log(
            f"Готово: ok={stats['ok']} no_label={stats['no_label']} "
            f"error={stats['error']} / files=1"
        )
        return 0 if stats["error"] == 0 else 1

    batch_no = 0
    total_files = 0
    prev_names: tuple[str, ...] | None = None

    while True:
        batch_no += 1
        files = list_pending_heic(uploads_dir, cropjson_dir, args.limit)
        if not files:
            if batch_no == 1:
                log("Нет HEIC без json в cropjson/")
            else:
                log(f"Infinitive: больше нет файлов после батча {batch_no - 1}, стоп")
            break

        names = tuple(p.name for p in files)
        if args.infinitive and prev_names is not None and names == prev_names:
            log("Infinitive: тот же набор файлов — стоп")
            break
        prev_names = names

        total_files += len(files)
        log(f"=== батч {batch_no}: {len(files)} HEIC ===")
        run_batch(
            files=files,
            api_keys=api_keys,
            cropjson_dir=cropjson_dir,
            crop_dir=crop_dir,
            prompt=prompt,
            model=model,
            stats=stats,
            stats_lock=stats_lock,
            rate_limiters=rate_limiters,
        )
        log("")

        if not args.infinitive:
            break

    log(
        f"Готово: ok={stats['ok']} no_label={stats['no_label']} "
        f"error={stats['error']} / files={total_files} "
        f"batches={batch_no if total_files else 0}"
    )
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
