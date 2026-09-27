"""Общая нормализация строк перед сравнением текстов этикеток.

Таблицы замен — ocr_homoglyphs.json рядом с модулем:
  latin_to_cyrillic, digit_to_letter, token_aliases.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("vino.text_compare")

_HOMOGLYPH_JSON = Path(__file__).with_name("ocr_homoglyphs.json")

_RE_CYR_LETTER = re.compile(r"[А-Яа-яЁё]")
_RE_LAT_LETTER = re.compile(r"[A-Za-z]")
_RE_ALPHA = re.compile(r"[A-Za-zА-Яа-яЁё]")
_RE_TOKEN_SPLIT = re.compile(r"(\s+)")
_RE_YEAR_ONLY = re.compile(r"^(?:19[5-9]\d|20[0-4]\d)$")
_RE_DECIMAL_NUM = re.compile(r"\d[.,]\d")
_RE_DIGIT_NEXT_TO_LETTER = re.compile(
    r"(?<=[A-Za-zА-Яа-яЁё])\d|\d(?=[A-Za-zА-Яа-яЁё])"
)

# Filled by load_ocr_homoglyphs()
_LATIN_HOMOGLYPH_TO_CYR: dict[int, int] = {}
_HOMOGLYPH_LATIN_CHARS: frozenset[str] = frozenset()
_DIGIT_TO_LETTER: dict[int, int] = {}
_TOKEN_ALIASES: dict[str, str] = {}


def _build_case_map(pairs: dict[str, str]) -> dict[str, str]:
    """Из верхнего регистра (A→А) добавить и нижний (a→а)."""
    out: dict[str, str] = {}
    for src, dst in pairs.items():
        s = str(src)
        d = str(dst)
        if not s or not d:
            continue
        out[s] = d
        # lower/upper пары, если однобуквенные
        if len(s) == 1 and len(d) == 1:
            sl, su = s.lower(), s.upper()
            dl, du = d.lower(), d.upper()
            out[sl] = dl
            out[su] = du
    return out


def load_ocr_homoglyphs(*, path: Path | None = None) -> Path:
    """Загрузить ocr_homoglyphs.json в модульные таблицы."""
    global _LATIN_HOMOGLYPH_TO_CYR, _HOMOGLYPH_LATIN_CHARS
    global _DIGIT_TO_LETTER, _TOKEN_ALIASES
    src = path or _HOMOGLYPH_JSON
    raw: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))

    latin_raw = raw.get("latin_to_cyrillic") or {}
    if not isinstance(latin_raw, dict):
        raise ValueError("ocr_homoglyphs.json: latin_to_cyrillic должен быть объектом")
    latin_map = _build_case_map({str(k): str(v) for k, v in latin_raw.items()})
    _LATIN_HOMOGLYPH_TO_CYR = str.maketrans(latin_map)
    _HOMOGLYPH_LATIN_CHARS = frozenset(
        ch for ch in latin_map if len(ch) == 1 and ch.isascii() and ch.isalpha()
    )

    digit_raw = raw.get("digit_to_letter") or {}
    if not isinstance(digit_raw, dict):
        raise ValueError("ocr_homoglyphs.json: digit_to_letter должен быть объектом")
    digit_map = {str(k): str(v) for k, v in digit_raw.items() if str(k) and str(v)}
    _DIGIT_TO_LETTER = str.maketrans(digit_map)

    alias_raw = raw.get("token_aliases") or {}
    if not isinstance(alias_raw, dict):
        raise ValueError("ocr_homoglyphs.json: token_aliases должен быть объектом")
    aliases: dict[str, str] = {}
    for k, v in alias_raw.items():
        key = str(k).strip().lower().replace("ё", "е")
        val = str(v).strip().lower().replace("ё", "е")
        if key and val:
            aliases[key] = val
    _TOKEN_ALIASES = aliases

    logger.info(
        "ocr_homoglyphs loaded from %s: latin=%d digit=%d aliases=%d",
        src.name,
        len(_HOMOGLYPH_LATIN_CHARS),
        len(digit_map),
        len(_TOKEN_ALIASES),
    )
    return src


# Load at import; keep empty tables if file broken (tests can reload).
try:
    load_ocr_homoglyphs()
except Exception as exc:  # noqa: BLE001
    logger.error("ocr_homoglyphs.json load failed: %s", exc)


def _fix_digit_letter_token(token: str) -> str:
    """Цифры→буквы только если цифра вплетена в буквенное слово.

    Не трогаем: чистые годы (2024), десятичные (0.75 / 12,5), проценты,
    токены где цифр больше чем букв (44N, координаты).
    """
    if not token or not any(ch.isdigit() for ch in token):
        return token
    if not _RE_ALPHA.search(token):
        return token
    if _RE_YEAR_ONLY.match(token):
        return token
    if _RE_DECIMAL_NUM.search(token) or "%" in token:
        return token
    n_digits = sum(ch.isdigit() for ch in token)
    n_alpha = sum(ch.isalpha() for ch in token)
    if n_digits > n_alpha:
        return token
    if not _RE_DIGIT_NEXT_TO_LETTER.search(token):
        return token
    if not _DIGIT_TO_LETTER:
        return token
    return token.translate(_DIGIT_TO_LETTER)


def fix_digit_letter_homoglyphs(text: str | None) -> str:
    """Токенно: OCR-цифры внутри слов → визуально похожие буквы."""
    raw = str(text or "")
    if not raw or not any(ch.isdigit() for ch in raw):
        return raw
    return "".join(
        part if not part or part.isspace() else _fix_digit_letter_token(part)
        for part in _RE_TOKEN_SPLIT.split(raw)
    )


def _fix_homoglyph_token(token: str) -> str:
    """Смешанный кириллица+латиница → омоглифы в кириллицу.

    Чистая латиница — только если все буквы из набора омоглифов
    (CYXOE→СУХОЕ; KASEPHE→КАБЕРНЕ; MERLO→МЕРЛО;
    CABERNET не трогаем: есть N вне набора).
    """
    if not token or not _RE_LAT_LETTER.search(token):
        return token
    if not _LATIN_HOMOGLYPH_TO_CYR:
        return token
    if _RE_CYR_LETTER.search(token):
        return token.translate(_LATIN_HOMOGLYPH_TO_CYR)
    latin_letters = _RE_LAT_LETTER.findall(token)
    if latin_letters and all(c in _HOMOGLYPH_LATIN_CHARS for c in latin_letters):
        return token.translate(_LATIN_HOMOGLYPH_TO_CYR)
    return token


def fix_latin_cyrillic_homoglyphs(text: str | None) -> str:
    """Токенно: латинские омоглифы Vision → кириллица (см. `_fix_homoglyph_token`)."""
    raw = str(text or "")
    if not raw or not _RE_LAT_LETTER.search(raw):
        return raw
    return "".join(
        part if not part or part.isspace() else _fix_homoglyph_token(part)
        for part in _RE_TOKEN_SPLIT.split(raw)
    )


def fix_token_aliases(text: str | None) -> str:
    """Точечные замены целых токенов из token_aliases (после lower)."""
    raw = str(text or "")
    if not raw or not _TOKEN_ALIASES:
        return raw
    parts = _RE_TOKEN_SPLIT.split(raw)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        key = part.lower().replace("ё", "е")
        out.append(_TOKEN_ALIASES.get(key, part))
    return "".join(out)


def normalize_compare_text(text: str | None) -> str:
    """Перед сравнением: цифры-в-слове→буквы, token_aliases, омоглифы→кириллица,
    lower, ё→е, снова aliases, пробелы→одиночные, trim.

    Алиасы до побуквенных омоглифов: иначе MEPRO→мерро и алиас mepro не сработает.
    """
    t = fix_digit_letter_homoglyphs(text)
    t = fix_token_aliases(t)
    t = fix_latin_cyrillic_homoglyphs(t)
    t = t.replace("\u00a0", " ").replace("\t", " ").replace("\r", "\n")
    t = t.lower().replace("ё", "е")
    t = fix_token_aliases(t)
    t = t.replace("\n", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t.strip()
