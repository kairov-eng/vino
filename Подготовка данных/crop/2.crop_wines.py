# =============================================================================
# 2.crop_wines.py
# =============================================================================
#
# НАЗНАЧЕНИЕ
#   Массовый crop этикеток для вин из PostgreSQL (таблица wines). Для каждой
#   необработанной записи находит фото в uploads/, вызывает Gemini, сохраняет
#   JSON + PNG кропа и пишет статус в wines.crop (+ текст в wines.label и
#   структурированный OCR в wines.label_ocr — в том же вызове Gemini).
#
# ОТКУДА БЕРЁТ ЗАДАНИЯ (SQL)
#   SELECT w.*, COALESCE(ws.slug, w.photo_name) AS image_file,
#          ws.image_file AS sitemap_image_file
#   FROM wines w
#   LEFT JOIN wines_sitemap ws ON w.slug = ws.slug
#   WHERE …
#     по умолчанию:          (w.crop <> 1 OR w.crop IS NULL)
#     --empty_label 1:       w.label IS NULL
#     --empty_label_ocr 1:   w.label_ocr IS NULL
#   ORDER BY w.id
#   LIMIT N          (--limit, по умолчанию 100)
#
# ПОИСК ФАЙЛА В uploads/
#   Порядок кандидатов: wines_sitemap.image_file → wines.photo_name → image_file
#   (slug). Сопоставление: точное имя, либо stem с '-'→'_' и тем же расширением
#   (часто stem_<hash10>.webp). Производные thumbnail_/small_/… пропускаются.
#
# СТАТУСЫ wines.crop
#   1  — этикетка найдена, PNG-кроп по label_box_2d, label = текст OCR,
#        label_ocr = структурированные поля (как в 11.parse_label_ocr /
#        search_photos.status.steps.ocr_*.json)
#   2  — рамки этикетки нет (ответ Gemini без размеров): в crop/ кладётся
#        исходник (PNG), label / label_ocr могут быть заполнены из текста
#   0  — файла на диске нет / пустой image_file; label = NULL, label_ocr = NULL
#  -1  — любая ошибка (Gemini, запись файлов и т.д.); label/label_ocr = NULL
#
# ЧТО ПИШЕТ НА ДИСК (относительно uploads/)
#   cropjson/<image_file>.json  — метаданные, боксы, текст, label_ocr,
#                                 tokens_input/output; gemini_raw — если нет
#                                 рамки или нет текста
#   crop/<image_file>_crop.png  — кроп этикетки или оригинал при crop=2
#
# ПАРАЛЛЕЛИЗМ И ЛИМИТЫ
#   AI Studio: число потоков = числу ключей GEMINI_API_KEY_* в .env.
#   Vertex AI: --vertex-threads N (по умолчанию 10).
#   На каждый поток — не более 15 запросов за скользящие 60 секунд
#   (KeyRateLimiter): при исчерпании квоты поток ждёт до конца окна.
#
# РЕЖИМ --infinitive 1
#   После обработки батча снова выполняет SELECT. Останавливается, когда
#   выборка пуста, либо вернулся тот же набор id (защита от зацикливания
#   на вечных crop=0/-1 или label_ocr IS NULL при ошибках).
#
# ЛОГИРОВАНИЕ
#   Каждая строка в терминале с префиксом [YYYY-MM-DD HH:MM:SS].
#   Дублируется в logs/crop_wines_YYYYMMDD_HHMMSS.log.
#
# ЗАВИСИМОСТИ / КОНФИГ
#   .env          — GEMINI_API_KEY_* / Vertex (GOOGLE_AI_API_VERTEX), GEMINI_MODEL,
#                   DATABASE_URL, UPLOADS_DIR
#   Промпт        — встроенный CROP_PROMPT (боксы + label_ocr, схема как
#                   ocr_prompt.txt / скрипт 11); в запрос добавляются подсказки
#                   каталога по вину
#   Функции Gemini — detect_bottle_label + gemini_client (AI Studio / Vertex)
#   Пакеты: google-genai, Pillow, python-dotenv, psycopg2-binary
#   Колонка wines.label_ocr — alembic 009_wines_label_ocr
#
# ПРИМЕРЫ ВЫЗОВА
#   python 2.crop_wines.py
#   python 2.crop_wines.py --limit 100
#   python 2.crop_wines.py --infinitive 1
#   python 2.crop_wines.py --id 3099
#   python 2.crop_wines.py --id 3099 3100 3101
#   python 2.crop_wines.py --empty_label 1 --infinitive 1
#   python 2.crop_wines.py --empty_label_ocr 1 --limit 50
#   python 2.crop_wines.py --vertex-threads 10
#   python 2.crop_wines.py --limit 50 --uploads C:\dev\Vino2026\uploads
#
# =============================================================================

"""Массовый crop этикеток для вин из таблицы wines через Gemini."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

from datetime import datetime

import psycopg2
from dotenv import load_dotenv
from PIL import Image
from psycopg2.extras import Json

# Переиспользуем Gemini-логику из detect_bottle_label.py
SCRIPT_DIR = Path(__file__).resolve().parent
LOGS_DIR = SCRIPT_DIR / "logs"
sys.path.insert(0, str(SCRIPT_DIR))
from detect_bottle_label import (  # noqa: E402
    ENV_PATH,
    box_to_pixels,
    call_gemini_one_key,
    load_api_keys,
    load_model_name,
    normalize_box,
)
from gemini_client import describe_backend, make_genai_client, use_vertex_api  # noqa: E402

_MONOREPO = Path(__file__).resolve().parents[2]  # Vino2026
UPLOADS_DEFAULT = _MONOREPO / "media" / "uploads"
MAX_RPM_PER_KEY = 15
RPM_WINDOW_SEC = 60.0
DEFAULT_VERTEX_THREADS = 10

# Один вызов: рамки бутылки/этикетки + структурированный OCR (схема как ocr_prompt.txt / 11).
CROP_PROMPT = """\
Определи на этой картинке прямоугольник с этикеткой и прямоугольник с бутылкой.
Если на этикетке есть текст — прочитай его и заполни структурированные поля OCR
(русский/английский). Если текста нет или он нечитаем — label_text пустой,
поля label_ocr — null / пустые, где уместно.

Координаты box_2d — целые числа в формате [ymin, xmin, ymax, xmax],
нормализованные к диапазону 0–1000 относительно размеров изображения
(0,0 — левый верхний угол).

Если бутылки нет — bottle_box_2d: [].
Если этикетки нет — label_box_2d: [] (и label_text / label_ocr пустые).

В конце сообщения могут быть «подсказки каталога» (name, winery, color, …).
Это ТОЛЬКО контекст для понимания, не источник истины: извлекай OCR-поля
ИМЕННО с этикетки на картинке. Не копируй значения из каталога, если их нет
на этикетке. Не выдумывай. Не путай год основания с годом урожая.

Ответ верни ТОЛЬКО валидным JSON без markdown и без пояснений, строго такого вида:

{
  "bottle_box_2d": [0, 243, 1000, 797],
  "label_box_2d": [342, 255, 802, 786],
  "label_text": "ТАБИЯ\\nвинодельня\\nПино Нуар\\nполусухое\\n2025",
  "label_ocr": {
    "full_text": "весь читаемый текст этикетки, строки через \\n",
    "lines": ["строка1", "строка2"],
    "producer": "винодельня или null",
    "wine_name": "название вина или null",
    "color": "белый|розовый|красное|… или null",
    "type": "сухое|полусухое|полусладкое|сладкое|брют|dry|semidry|semisweet|sweet|brut|... или null",
    "cupage": "сорт винограда или несколько сортов или null",
    "vintage_year": "год урожая YYYY или null",
    "foundation_year": "год основания YYYY или null",
    "region": "регион или null",
    "alcohol": "крепость или null",
    "volume": "объём или null",
    "other": "прочий полезный текст или null"
  }
}
"""

OCR_KEYS = (
    "full_text",
    "lines",
    "producer",
    "wine_name",
    "color",
    "type",
    "cupage",
    "vintage_year",
    "foundation_year",
    "region",
    "alcohol",
    "volume",
    "other",
)

# SELECT TOP 100 ... → PostgreSQL LIMIT %(limit)s
# Без --slug / --id: 1 — финальный успех (не берём повторно); 2 тоже можно
# переобработать по текущему WHERE.
# С --slug / --id: только указанные вина (см. fetch_batch), фильтр crop игнорируется.
# --empty_label 1: WHERE w.label IS NULL вместо фильтра по crop.
# --empty_label_ocr 1: WHERE w.label_ocr IS NULL вместо фильтра по crop.
_FETCH_SQL_SELECT = """
SELECT
    w.*,
    COALESCE(ws.slug, w.photo_name) AS image_file,
    ws.image_file AS sitemap_image_file
FROM wines w
LEFT JOIN wines_sitemap ws ON w.slug = ws.slug
"""

FETCH_SQL = (
    _FETCH_SQL_SELECT
    + """
WHERE (w.crop <> 1 OR w.crop IS NULL)
ORDER BY w.id
LIMIT %(limit)s
"""
)

FETCH_SQL_EMPTY_LABEL = (
    _FETCH_SQL_SELECT
    + """
WHERE w.label IS NULL
ORDER BY w.id
LIMIT %(limit)s
"""
)

FETCH_SQL_EMPTY_LABEL_OCR = (
    _FETCH_SQL_SELECT
    + """
WHERE w.label_ocr IS NULL
ORDER BY w.id
LIMIT %(limit)s
"""
)

FETCH_BY_SLUGS_SQL = """
SELECT
    w.*,
    COALESCE(ws.slug, w.photo_name) AS image_file,
    ws.image_file AS sitemap_image_file
FROM wines w
LEFT JOIN wines_sitemap ws ON w.slug = ws.slug
WHERE w.slug = ANY(%(slugs)s)
ORDER BY w.id
"""

FETCH_BY_IDS_SQL = """
SELECT
    w.*,
    COALESCE(ws.slug, w.photo_name) AS image_file,
    ws.image_file AS sitemap_image_file
FROM wines w
LEFT JOIN wines_sitemap ws ON w.slug = ws.slug
WHERE w.id = ANY(%(ids)s)
ORDER BY w.id
"""

UPDATE_CROP_SQL = """
UPDATE wines
SET crop = %(crop)s,
    label = %(label)s,
    label_ocr = %(label_ocr)s
WHERE id = %(id)s
"""

_print_lock = threading.Lock()
_db_lock = threading.Lock()
_log_file: Path | None = None


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def init_logging() -> Path:
    """Создаёт logs/ и файл сессии; все log() пишутся туда же."""
    global _log_file
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = LOGS_DIR / f"crop_wines_{stamp}.log"
    with _log_file.open("a", encoding="utf-8") as f:
        f.write(f"=== crop_wines start {datetime.now().isoformat(sep=' ', timespec='seconds')} ===\n")
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


def load_database_url() -> str:
    load_dotenv(ENV_PATH)
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise SystemExit(f"DATABASE_URL не задан в {ENV_PATH}")
    if url.startswith("postgresql+psycopg2://"):
        url = "postgresql://" + url.split("://", 1)[1]
    return url


def load_uploads_dir(cli_value: Path | None) -> Path:
    load_dotenv(ENV_PATH)
    if cli_value is not None:
        return cli_value.resolve()
    env = (os.getenv("UPLOADS_DIR") or "").strip()
    path = Path(env) if env else UPLOADS_DEFAULT
    return path.resolve()


def ensure_crop_column(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'wines' AND column_name = 'crop'
            """
        )
        if not cur.fetchone():
            cur.execute(
                """
                ALTER TABLE wines
                ADD COLUMN crop integer DEFAULT 0;
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_wines_crop ON wines (crop);")
            cur.execute("UPDATE wines SET crop = 0 WHERE crop IS NULL;")
            conn.commit()
            log("Добавлена колонка wines.crop (default 0)")

        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'wines' AND column_name = 'label'
            """
        )
        if not cur.fetchone():
            cur.execute("ALTER TABLE wines ADD COLUMN label text;")
            conn.commit()
            log("Добавлена колонка wines.label")

        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'wines'
              AND column_name = 'label_ocr'
            """
        )
        if not cur.fetchone():
            raise SystemExit(
                "Нет колонки wines.label_ocr. "
                "Сначала: cd vino-svoe/backend && alembic upgrade head"
            )


def build_prompt_for_row(row: dict) -> str:
    """CROP_PROMPT + подсказки каталога (как catalog в 11.parse_label_ocr)."""
    catalog = {
        "name": row.get("name"),
        "winery": row.get("winery"),
        "color": row.get("color"),
        "region": row.get("region"),
        "grape_variety": row.get("grape_variety"),
        "category": row.get("category"),
        "slug": row.get("slug"),
    }
    hint = json.dumps(catalog, ensure_ascii=False, indent=2)
    return (
        f"{CROP_PROMPT.rstrip()}\n\n"
        f"Подсказки каталога (не источник истины):\n{hint}\n"
    )


def extract_label_ocr(data: dict) -> dict | None:
    """Достаёт label_ocr из ответа Gemini (вложенный объект или плоские поля)."""
    if not isinstance(data, dict):
        return None

    raw = data.get("label_ocr")
    if isinstance(raw, dict):
        src = raw
    else:
        # плоский ответ: OCR-ключи на верхнем уровне
        if not any(k in data for k in OCR_KEYS if k != "full_text"):
            if not data.get("full_text") and not data.get("lines"):
                return None
        src = data

    out: dict = {}
    for key in OCR_KEYS:
        out[key] = src.get(key, None)

    if out.get("vintage_year") is None and src.get("vintage") is not None:
        out["vintage_year"] = src.get("vintage")

    # full_text из label_text / lines, если модель не заполнила
    if not (isinstance(out.get("full_text"), str) and out["full_text"].strip()):
        lt = data.get("label_text")
        if isinstance(lt, str) and lt.strip():
            out["full_text"] = lt.strip()
        elif isinstance(out.get("lines"), list):
            joined = "\n".join(
                str(x).strip() for x in out["lines"] if str(x).strip()
            )
            out["full_text"] = joined or None

    # если совсем пусто — не пишем в БД
    has_any = False
    for key in OCR_KEYS:
        val = out.get(key)
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        if isinstance(val, list) and not val:
            continue
        has_any = True
        break
    return out if has_any else None


def fetch_batch(
    conn,
    limit: int,
    slugs: list[str] | None = None,
    ids: list[int] | None = None,
    *,
    empty_label: bool = False,
    empty_label_ocr: bool = False,
) -> list[dict]:
    with conn.cursor() as cur:
        if ids:
            cur.execute(FETCH_BY_IDS_SQL, {"ids": ids})
        elif slugs:
            cur.execute(FETCH_BY_SLUGS_SQL, {"slugs": slugs})
        elif empty_label:
            cur.execute(FETCH_SQL_EMPTY_LABEL, {"limit": limit})
        elif empty_label_ocr:
            cur.execute(FETCH_SQL_EMPTY_LABEL_OCR, {"limit": limit})
        else:
            cur.execute(FETCH_SQL, {"limit": limit})
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def set_crop(
    conn,
    wine_id: int,
    value: int,
    label: str | None = None,
    label_ocr: dict | None = None,
) -> None:
    with _db_lock:
        with conn.cursor() as cur:
            cur.execute(
                UPDATE_CROP_SQL,
                {
                    "crop": value,
                    "label": label,
                    "label_ocr": Json(label_ocr) if label_ocr is not None else None,
                    "id": wine_id,
                },
            )
        conn.commit()


class KeyRateLimiter:
    """Не более max_rpm запросов за window_sec на один ключ."""

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


def image_file_parts(image_file: str) -> tuple[str, str, str]:
    name = Path(str(image_file)).name
    path = Path(name)
    stem = path.stem
    ext = path.suffix.lower()
    if not ext:
        ext = ".webp"
    return name, stem, ext


def build_uploads_index(uploads_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    exact: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    for p in uploads_dir.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.startswith(("thumbnail_", "small_", "medium_", "large_")):
            continue
        if p.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
            ".gif",
            ".tif",
            ".tiff",
        }:
            continue
        exact[low] = p
        stem = p.stem.lower()
        by_stem.setdefault(stem, []).append(p)
        if (
            len(stem) > 11
            and stem[-11] == "_"
            and all(c in "0123456789abcdef" for c in stem[-10:])
        ):
            by_stem.setdefault(stem[:-11], []).append(p)
    return exact, by_stem


def find_upload_file(
    uploads_dir: Path,
    image_file: str,
    *,
    photo_name: str | None = None,
    sitemap_image_file: str | None = None,
    exact_index: dict[str, Path] | None = None,
    stem_index: dict[str, list[Path]] | None = None,
) -> Path | None:
    for candidate in (sitemap_image_file, photo_name, image_file):
        if not candidate or not str(candidate).strip():
            continue
        found = _match_one(
            uploads_dir,
            str(candidate).strip(),
            exact_index=exact_index,
            stem_index=stem_index,
        )
        if found is not None:
            return found
    return None


def _match_one(
    uploads_dir: Path,
    image_file: str,
    *,
    exact_index: dict[str, Path] | None = None,
    stem_index: dict[str, list[Path]] | None = None,
) -> Path | None:
    _, stem, ext = image_file_parts(image_file)
    name = Path(str(image_file)).name
    prefix = stem.replace("-", "_").lower()

    if exact_index is not None and stem_index is not None:
        for key in (name.lower(), f"{stem}{ext}".lower(), f"{prefix}{ext}"):
            hit = exact_index.get(key)
            if hit is not None and hit.is_file():
                return hit
        hits = [p for p in stem_index.get(prefix, []) if p.suffix.lower() == ext]
        if hits:
            hits = sorted(set(hits), key=lambda p: (len(p.name), p.name))
            return hits[0]
        return None

    exact = uploads_dir / name
    if exact.is_file():
        return exact
    candidate = uploads_dir / f"{stem}{ext}"
    if candidate.is_file():
        return candidate
    hits = []
    for p in uploads_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ext:
            continue
        if p.name.lower().startswith(("thumbnail_", "small_", "medium_", "large_")):
            continue
        pstem = p.stem.lower()
        if pstem == prefix or pstem.startswith(prefix + "_"):
            hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.name), p.name))
    return hits[0]


def safe_result_basename(image_file: str) -> str:
    raw = str(image_file).strip().replace("\\", "/").split("/")[-1]
    return raw.replace("\\", "_").replace("/", "_") or "unknown"


def _norm_key(name: str) -> str:
    """Сравнение имён: регистр и '-' vs '_' игнорируем."""
    stem = Path(str(name).strip()).stem.lower()
    if stem.endswith("_crop"):
        stem = stem[: -len("_crop")]
    return stem.replace("-", "_")


def build_crop_index(crop_dir: Path) -> dict[str, Path]:
    """Индекс crop/*.png → ключ нормализованного basename без суффикса _crop."""
    index: dict[str, Path] = {}
    if not crop_dir.is_dir():
        return index
    for p in crop_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".png":
            continue
        key = _norm_key(p.name)
        # более короткое имя предпочтительнее при коллизии
        prev = index.get(key)
        if prev is None or len(p.name) < len(prev.name):
            index[key] = p
    return index


def find_crop_path(
    row: dict,
    *,
    uploads_dir: Path,
    crop_dir: Path,
    exact_index: dict[str, Path] | None = None,
    stem_index: dict[str, list[Path]] | None = None,
    crop_index: dict[str, Path] | None = None,
    allow_source_fallback: bool = True,
) -> Path | None:
    """
    Находит файл этикетки для embedding.

    Каноническое имя (как пишет process_row в этом скрипте):
      crop/{safe_result_basename(image_file)}_crop.png
    где image_file = COALESCE(ws.slug, w.photo_name) из SELECT.

    Дальше — запасные варианты имён и (опционально) исходник из uploads/.
    """
    image_file = (row.get("image_file") or "").strip()

    # 1) То же имя, что при сохранении кропа
    tried: set[str] = set()
    if image_file:
        base = safe_result_basename(image_file)
        name = f"{base}_crop.png"
        tried.add(name)
        p = crop_dir / name
        if p.is_file():
            return p
        if crop_index is not None:
            hit = crop_index.get(_norm_key(base))
            if hit is not None and hit.is_file():
                return hit

    # 2) Прочие поля БД (legacy / если image_file менялся после кропа)
    for key in ("photo_name", "sitemap_image_file", "slug"):
        val = row.get(key)
        if not val or not str(val).strip():
            continue
        base = safe_result_basename(str(val).strip())
        for name in (f"{base}_crop.png", f"{Path(str(val)).name}_crop.png"):
            if name in tried:
                continue
            tried.add(name)
            p = crop_dir / name
            if p.is_file():
                return p
        if crop_index is not None:
            hit = crop_index.get(_norm_key(base))
            if hit is not None and hit.is_file():
                return hit

    # 3) Через найденный исходник в uploads/
    found = find_upload_file(
        uploads_dir,
        image_file,
        photo_name=row.get("photo_name"),
        sitemap_image_file=row.get("sitemap_image_file"),
        exact_index=exact_index,
        stem_index=stem_index,
    )
    if found is not None:
        for name in (
            f"{safe_result_basename(found.name)}_crop.png",
            f"{found.name}_crop.png",
            f"{found.stem}_crop.png",
        ):
            if name in tried:
                continue
            tried.add(name)
            p = crop_dir / name
            if p.is_file():
                return p
        if crop_index is not None:
            hit = crop_index.get(_norm_key(found.name))
            if hit is not None and hit.is_file():
                return hit
            hit = crop_index.get(_norm_key(found.stem))
            if hit is not None and hit.is_file():
                return hit

        # 4) Кропа нет на диске, но исходник есть — для embedding можно взять его
        if allow_source_fallback and found.is_file():
            return found

    return None


def _open_image(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img


def crop_label(image_path: Path, data: dict, out_path: Path) -> None:
    box = normalize_box(data.get("label_box_2d"))
    if not box:
        raise RuntimeError("В ответе Gemini нет label_box_2d")

    img = _open_image(image_path)
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
    """Нет рамки этикетки — в crop кладём исходную картинку (PNG)."""
    img = _open_image(image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, format="PNG")


def has_label_box(data: dict) -> bool:
    """True, если Gemini вернул валидные размеры этикетки."""
    return normalize_box(data.get("label_box_2d")) is not None


def process_row(
    conn,
    row: dict,
    *,
    uploads_dir: Path,
    cropjson_dir: Path,
    crop_dir: Path,
    model: str,
    api_key: str,
    key_label: str,
    rate_limiter: KeyRateLimiter,
    exact_index: dict[str, Path] | None = None,
    stem_index: dict[str, list[Path]] | None = None,
) -> str:
    wine_id = int(row["id"])
    image_file = (row.get("image_file") or "").strip()

    try:
        if not image_file:
            set_crop(conn, wine_id, 0, None)
            return "skip: empty image_file -> crop=0"

        found = find_upload_file(
            uploads_dir,
            image_file,
            photo_name=row.get("photo_name"),
            sitemap_image_file=row.get("sitemap_image_file"),
            exact_index=exact_index,
            stem_index=stem_index,
        )
        if found is None:
            set_crop(conn, wine_id, 0, None)
            return f"skip: file not found for {image_file!r} -> crop=0"

        base = safe_result_basename(image_file)
        json_path = cropjson_dir / f"{base}.json"
        crop_path = crop_dir / f"{base}_crop.png"

        rate_limiter.acquire()
        prompt = build_prompt_for_row(row)
        data, elapsed, tokens_in, tokens_out = call_gemini_one_key(
            found, prompt, model, api_key
        )

        # Текст на этикетке может отсутствовать — это нормально, не ошибка.
        raw_text = data.get("label_text") if isinstance(data, dict) else None
        if raw_text is None:
            label_text = None
        else:
            label_text = str(raw_text).strip() or None

        label_ocr = extract_label_ocr(data if isinstance(data, dict) else {})
        if label_text is None and label_ocr and label_ocr.get("full_text"):
            label_text = str(label_ocr["full_text"]).strip() or None
        if label_ocr is None and label_text:
            label_ocr = {k: None for k in OCR_KEYS}
            label_ocr["full_text"] = label_text
            label_ocr["lines"] = [
                ln for ln in label_text.splitlines() if ln.strip()
            ] or None

        label_box_ok = has_label_box(data if isinstance(data, dict) else {})
        has_text = label_text is not None

        payload = {
            "wine_id": wine_id,
            "image_file": image_file,
            "source_file": found.name,
            "elapsed_sec": round(elapsed, 3),
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "bottle_box_2d": data.get("bottle_box_2d") if isinstance(data, dict) else None,
            "label_box_2d": data.get("label_box_2d") if isinstance(data, dict) else None,
            "label_text": raw_text,
            "label_ocr": label_ocr,
        }
        # gemini_raw — только если нет размеров этикетки или нет текста
        if (not label_box_ok) or (not has_text):
            payload["gemini_raw"] = data

        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not label_box_ok:
            # Нет размеров этикетки — crop=2, в файл crop пишем оригинал.
            save_original_as_crop(found, crop_path)
            set_crop(conn, wine_id, 2, label_text, label_ocr)
            return (
                f"no-label [{key_label}] Gemini {elapsed:.3f}s "
                f"in={tokens_in} out={tokens_out} | {found.name} | "
                f"json={json_path.name} original->crop={crop_path.name} "
                f"label_ocr={'yes' if label_ocr else 'no'} -> crop=2"
            )

        crop_label(found, data, crop_path)
        set_crop(conn, wine_id, 1, label_text, label_ocr)
        return (
            f"ok [{key_label}] Gemini {elapsed:.3f}s "
            f"in={tokens_in} out={tokens_out} | {found.name} | "
            f"json={json_path.name} crop={crop_path.name} "
            f"label_ocr={'yes' if label_ocr else 'no'} -> crop=1"
        )
    except Exception as exc:  # noqa: BLE001 — любые ошибки -> crop=-1
        try:
            set_crop(conn, wine_id, -1, None)
        except Exception as db_exc:  # noqa: BLE001
            return f"error: {exc}; also failed to set crop=-1: {db_exc}"
        return f"error [{key_label}]: {exc} -> crop=-1"


def run_batch(
    *,
    rows: list[dict],
    api_keys: list[str],
    db_url: str,
    uploads_dir: Path,
    cropjson_dir: Path,
    crop_dir: Path,
    model: str,
    exact_index: dict[str, Path],
    stem_index: dict[str, list[Path]],
    stats: dict[str, int],
    stats_lock: threading.Lock,
    rate_limiters: list[KeyRateLimiter],
) -> None:
    task_queue = list(rows)
    queue_lock = threading.Lock()
    threads: list[threading.Thread] = []

    def _worker(
        api_key: str,
        key_index: int,
        limiter: KeyRateLimiter,
    ) -> None:
        key_label = f"key{key_index + 1}/...{api_key[-6:]}"
        conn = psycopg2.connect(db_url)
        try:
            while True:
                with queue_lock:
                    if not task_queue:
                        return
                    row = task_queue.pop(0)

                wine_id = row["id"]
                image_file = row.get("image_file")
                log(f"> id={wine_id} image_file={image_file} [{key_label}]")
                t0 = time.perf_counter()
                msg = process_row(
                    conn,
                    row,
                    uploads_dir=uploads_dir,
                    cropjson_dir=cropjson_dir,
                    crop_dir=crop_dir,
                    model=model,
                    api_key=api_key,
                    key_label=key_label,
                    rate_limiter=limiter,
                    exact_index=exact_index,
                    stem_index=stem_index,
                )
                wall_sec = time.perf_counter() - t0
                log(f"  {msg} | time={wall_sec:.3f}s")
                with stats_lock:
                    if "-> crop=1" in msg:
                        stats["ok"] += 1
                    elif "-> crop=2" in msg:
                        stats["no_label"] += 1
                    elif "-> crop=0" in msg:
                        stats["not_found"] += 1
                    else:
                        stats["error"] += 1
        finally:
            conn.close()

    for i, key in enumerate(api_keys):
        t = threading.Thread(
            target=_worker,
            name=f"gemini-worker-{i + 1}",
            args=(key, i, rate_limiters[i]),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    log_path = init_logging()
    parser = argparse.ArgumentParser(
        description="Crop этикеток вин через Gemini (wines.crop), потоки = число ключей"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Сколько записей взять за один SELECT (TOP N), по умолчанию 100",
    )
    parser.add_argument(
        "--infinitive",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = повторять SELECT, пока есть строки с crop не в (1,2)",
    )
    parser.add_argument(
        "--uploads",
        type=Path,
        default=None,
        help="Каталог uploads (по умолчанию из .env / C:\\dev\\Vino2026\\uploads)",
    )
    parser.add_argument(
        "--slug",
        nargs="+",
        default=None,
        help="Обработать только эти slug (игнорирует фильтр crop / --limit)",
    )
    parser.add_argument(
        "--id",
        nargs="+",
        type=int,
        default=None,
        dest="wine_ids",
        help="Обработать только эти wines.id (повторный прогон; игнорирует фильтр crop / --limit)",
    )
    parser.add_argument(
        "--empty_label",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = в SELECT брать WHERE label IS NULL вместо фильтра по crop",
    )
    parser.add_argument(
        "--empty_label_ocr",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = в SELECT брать WHERE label_ocr IS NULL вместо фильтра по crop",
    )
    parser.add_argument(
        "--vertex-threads",
        type=int,
        default=DEFAULT_VERTEX_THREADS,
        help=(
            f"Число потоков при Vertex AI (GOOGLE_AI_API_VERTEX=1), "
            f"по умолчанию {DEFAULT_VERTEX_THREADS}; для AI Studio игнорируется"
        ),
    )
    args = parser.parse_args(argv)
    empty_label = bool(args.empty_label)
    empty_label_ocr = bool(args.empty_label_ocr)
    if empty_label and empty_label_ocr:
        raise SystemExit("Укажите только один из флагов: --empty_label или --empty_label_ocr")
    if args.slug and args.wine_ids:
        raise SystemExit("Укажите только один из флагов: --slug или --id")
    if args.wine_ids is not None and any(i < 1 for i in args.wine_ids):
        raise SystemExit("--id: все id должны быть >= 1")
    if args.vertex_threads < 1:
        raise SystemExit("--vertex-threads должен быть >= 1")

    uploads_dir = load_uploads_dir(args.uploads)
    if not uploads_dir.is_dir():
        raise SystemExit(f"Каталог uploads не найден: {uploads_dir}")

    cropjson_dir = uploads_dir / "cropjson"
    crop_dir = uploads_dir / "crop"
    cropjson_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_name()
    if use_vertex_api():
        make_genai_client(api_key=None, model=model)
        # N одинаковых слотов: make_genai_client при Vertex игнорирует api_key
        api_keys = [f"vertex-{i + 1}" for i in range(args.vertex_threads)]
        log(f"backend: {describe_backend(model=model)}")
    else:
        api_keys = load_api_keys()
        if not api_keys:
            raise SystemExit("Нет GEMINI_API_KEY_*")
        log(f"backend: {describe_backend(api_key=api_keys[0], model=model)}")
    db_url = load_database_url()

    log(f"DB: {db_url.rsplit('@', 1)[-1]}")
    log(f"Лог-файл: {log_path}")
    log(f"Модель: {model}")
    log("Промпт: CROP_PROMPT (боксы + label_ocr)")
    log(f"Uploads: {uploads_dir}")
    log(f"Limit: {args.limit}")
    log(f"Infinitive: {args.infinitive if not (args.slug or args.wine_ids) else 0}")
    log(f"empty_label: {int(empty_label)}")
    log(f"empty_label_ocr: {int(empty_label_ocr)}")
    if args.slug:
        log(f"Slugs: {len(args.slug)} шт.")
    if args.wine_ids:
        log(f"Ids: {args.wine_ids}")
    if use_vertex_api():
        # Для одного/нескольких id не раздуваем пул зря
        if args.wine_ids or args.slug:
            n = max(1, min(len(api_keys), len(args.wine_ids or args.slug or [1])))
            api_keys = api_keys[:n]
        log(
            f"Потоков: {len(api_keys)} (Vertex AI, --vertex-threads), "
            f"лимит {MAX_RPM_PER_KEY} req/min на поток"
        )
    else:
        log(f"Потоков/ключей: {len(api_keys)}, лимит {MAX_RPM_PER_KEY} req/min на ключ")
    log("Индексация uploads...")
    exact_index, stem_index = build_uploads_index(uploads_dir)
    log(f"Файлов в индексе: {len(exact_index)}")
    log("")

    conn = psycopg2.connect(db_url)
    try:
        ensure_crop_column(conn)
    finally:
        conn.close()

    stats = {"ok": 0, "no_label": 0, "not_found": 0, "error": 0}
    stats_lock = threading.Lock()
    rate_limiters = [
        KeyRateLimiter(f"key{i + 1}/...{k[-6:]}") for i, k in enumerate(api_keys)
    ]
    t_run0 = time.perf_counter()

    def _log_total_time() -> None:
        total_sec = time.perf_counter() - t_run0
        log(f"Общее время: {total_sec:.3f}s ({total_sec / 60.0:.1f} min)")

    # Режим --slug / --id: один проход только по указанным винам
    if args.slug or args.wine_ids:
        conn = psycopg2.connect(db_url)
        try:
            rows = fetch_batch(
                conn,
                args.limit,
                slugs=args.slug,
                ids=args.wine_ids,
            )
        finally:
            conn.close()
        if not rows:
            log("Нет вин с указанными slug/id")
            _log_total_time()
            return 1
        if args.slug:
            found = {r.get("slug") for r in rows}
            missing = [s for s in args.slug if s not in found]
            if missing:
                log(f"Не найдены в БД (slug): {missing}")
            log(f"=== slug-batch: {len(rows)} записей ===")
        else:
            found_ids = {int(r["id"]) for r in rows}
            missing_ids = [i for i in args.wine_ids if i not in found_ids]
            if missing_ids:
                log(f"Не найдены в БД (id): {missing_ids}")
            log(f"=== id-batch: {len(rows)} записей ===")
        run_batch(
            rows=rows,
            api_keys=api_keys,
            db_url=db_url,
            uploads_dir=uploads_dir,
            cropjson_dir=cropjson_dir,
            crop_dir=crop_dir,
            model=model,
            exact_index=exact_index,
            stem_index=stem_index,
            stats=stats,
            stats_lock=stats_lock,
            rate_limiters=rate_limiters[: len(api_keys)],
        )
        log(
            f"Готово: ok={stats['ok']} no_label={stats['no_label']} "
            f"not_found/empty={stats['not_found']} error={stats['error']} "
            f"/ rows={len(rows)}"
        )
        _log_total_time()
        return 0 if stats["error"] == 0 else 1

    batch_no = 0
    total_rows = 0
    prev_batch_ids: tuple[int, ...] | None = None
    while True:
        batch_no += 1
        conn = psycopg2.connect(db_url)
        try:
            rows = fetch_batch(
                conn,
                args.limit,
                empty_label=empty_label,
                empty_label_ocr=empty_label_ocr,
            )
        finally:
            conn.close()

        if not rows:
            if batch_no == 1:
                if empty_label:
                    log("Нет записей с label IS NULL")
                elif empty_label_ocr:
                    log("Нет записей с label_ocr IS NULL")
                else:
                    log("Нет записей с crop не в (1, 2)")
            else:
                log(f"Infinitive: SELECT пуст после батча {batch_no - 1}, стоп")
            break

        batch_ids = tuple(int(r["id"]) for r in rows)
        if args.infinitive and prev_batch_ids is not None and batch_ids == prev_batch_ids:
            if empty_label:
                stuck_hint = "label так и остаётся NULL"
            elif empty_label_ocr:
                stuck_hint = "label_ocr так и остаётся NULL"
            else:
                stuck_hint = "только crop=0/-1"
            log(
                "Infinitive: тот же набор id, что в прошлом батче "
                f"({stuck_hint}) — стоп, чтобы не зациклиться"
            )
            break
        prev_batch_ids = batch_ids

        total_rows += len(rows)
        log(f"=== батч {batch_no}: {len(rows)} записей ===")
        run_batch(
            rows=rows,
            api_keys=api_keys,
            db_url=db_url,
            uploads_dir=uploads_dir,
            cropjson_dir=cropjson_dir,
            crop_dir=crop_dir,
            model=model,
            exact_index=exact_index,
            stem_index=stem_index,
            stats=stats,
            stats_lock=stats_lock,
            rate_limiters=rate_limiters,
        )
        log("")

        if not args.infinitive:
            break

    log(
        f"Готово: ok={stats['ok']} no_label={stats['no_label']} "
        f"not_found/empty={stats['not_found']} error={stats['error']} "
        f"/ rows={total_rows} batches={batch_no if total_rows else 0}"
    )
    _log_total_time()
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
