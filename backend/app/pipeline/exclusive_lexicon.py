"""Взаимоисключающие слова этикетки (категория / тип сахаристости / сорта / винодельня).

Единое правило для всех справочников: если в OCR искомого найдена форма
из справочника, та же форма должна быть у кандидата (этикетка + поля карточки).
Синоним того же канона («белое» вместо «blanc») не засчитывается.
Чужой канон у кандидата отдельно не проверяется — достаточно наличия формы
из запроса.

При поиске формы на этикетке также учитывается транслитерация RU↔LAT
(ДЕНИСОВ ↔ DENISOV и наоборот).

grape/winery на запросе: проверка по убыванию частоты в каталоге; после
первого хита остальные каноны не ищутся. Многословные формы grape/winery
в проверке сводятся к первому слову.

grape/winery: допускаются OCR-опечатки (±1 буква и/или 1 замена в токене).

Статические справочники category/type — exclusive_lexicon.json рядом с модулем.
Сорт и винодельня — из БД при старте (с частотами по числу вин).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Wine
from app.pipeline.ocr_match import normalize_ocr_text

logger = logging.getLogger("vino.exclusive_lexicon")


def _console_print(msg: str) -> None:
    """Avoid UnicodeEncodeError on Windows cp125x consoles."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)

_LEXICON_JSON = Path(__file__).with_name("exclusive_lexicon.json")

# (canon, surface forms) — порядок из JSON; длинные формы раньше коротких
_CATEGORY_GROUPS: list[tuple[str, tuple[str, ...]]] = []
_TYPE_GROUPS: list[tuple[str, tuple[str, ...]]] = []
_LEXICONS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "category": _CATEGORY_GROUPS,
    "type": _TYPE_GROUPS,
}

# Filled at startup by load_grape_variety_lexicon() / load_winery_lexicon()
_GRAPE_FORMS: list[str] = []
_WINERY_FORMS: list[str] = []
# canon → число вин в каталоге, у которых встретилась эта форма
_LEXICON_FREQ: dict[str, dict[str, int]] = {"grape": {}, "winery": {}}

_FORM_RES: dict[str, list[tuple[str, list[tuple[str, re.Pattern[str]]]]]] = {}

# Кеш regex форм (в т.ч. транслит-варианты в _form_matches_hay).
_COMPILE_FORM_CACHE: dict[str, re.Pattern[str]] = {}

# Все справочники: форма из OCR обязана быть у кандидата.
_ALL_LEXICONS = ("category", "type", "grape", "winery")


def _compile_form(form: str, *, spaced: bool = True) -> re.Pattern[str]:
    """Regex для формы словаря.

    spaced=True: 0–2 пробела/дефиса между буквами («красный» ≈ «кра с  ны й»).
    spaced=False: только целые слова, без вставок между буквами.
    """
    f = normalize_ocr_text(form).strip()
    cache_key = f"{'s' if spaced else 'e'}:{f}"
    cached = _COMPILE_FORM_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not f:
        pat = re.compile(r"(?!x)x")
        _COMPILE_FORM_CACHE[cache_key] = pat
        return pat
    words = re.split(r"[\s\-]+", f)
    word_bodies: list[str] = []
    for w in words:
        if not w:
            continue
        chars = [re.escape(ch) for ch in w]
        if spaced:
            word_bodies.append(r"[\s\-]{0,2}".join(chars))
        else:
            word_bodies.append("".join(chars))
    if not word_bodies:
        pat = re.compile(r"(?!x)x")
        _COMPILE_FORM_CACHE[cache_key] = pat
        return pat
    body = r"[\s\-]+".join(word_bodies)
    pat = re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)
    _COMPILE_FORM_CACHE[cache_key] = pat
    return pat


# --- RU ↔ LAT transliteration (этикетки / producer names) -------------------
# GOST 7.79-2000 system B–like; х→kh (и h как доп. вариант при lat→cyr).

_CYR_TO_LAT: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

# Longest-first for latin → cyrillic.
_LAT_TO_CYR_MULTI: tuple[tuple[str, str], ...] = (
    ("shch", "щ"),
    ("kh", "х"),
    ("zh", "ж"),
    ("ts", "ц"),
    ("ch", "ч"),
    ("sh", "ш"),
    ("yu", "ю"),
    ("ya", "я"),
    ("yo", "ё"),
    ("ye", "е"),
    ("iy", "ий"),
    ("yy", "ый"),
)

_LAT_TO_CYR_SINGLE: dict[str, str] = {
    "a": "а",
    "b": "б",
    "v": "в",
    "g": "г",
    "d": "д",
    "e": "е",
    "z": "з",
    "i": "и",
    "y": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "f": "ф",
    "h": "х",
    "j": "й",
    "w": "в",
    "x": "кс",
    "c": "к",
    "q": "к",
}


def _translit_cyr_to_lat(text: str) -> str:
    """Кириллица → латиница; прочие символы без изменений."""
    out: list[str] = []
    for ch in text or "":
        low = ch.lower()
        if low in _CYR_TO_LAT:
            out.append(_CYR_TO_LAT[low])
        else:
            out.append(ch)
    return "".join(out)


def _translit_lat_to_cyr(text: str) -> str:
    """Латиница → кириллица (greedy multi-char); кириллица/прочее без изменений."""
    s = text or ""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        low = ch.lower()
        if "a" <= low <= "z":
            matched = False
            for lat, cyr in _LAT_TO_CYR_MULTI:
                if s[i : i + len(lat)].lower() == lat:
                    out.append(cyr)
                    i += len(lat)
                    matched = True
                    break
            if matched:
                continue
            out.append(_LAT_TO_CYR_SINGLE.get(low, ch))
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _form_script_variants(form: str, *, use_translit: bool = True) -> list[str]:
    """Нормализованная форма + опционально транслит в другой алфавит."""
    n = normalize_ocr_text(form).strip()
    if not n:
        return []
    if not use_translit:
        return [n]
    out: list[str] = []
    seen: set[str] = set()
    for cand in (n, _translit_cyr_to_lat(n), _translit_lat_to_cyr(n)):
        v = normalize_ocr_text(cand).strip() if cand != n else n
        if not v:
            v = (cand or "").strip().lower()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _expand_hay(hay: str, *, use_translit: bool) -> str:
    if not hay:
        return ""
    if not use_translit:
        return hay
    return "\n".join(
        dict.fromkeys(
            [
                hay,
                normalize_ocr_text(_translit_cyr_to_lat(hay)),
                normalize_ocr_text(_translit_lat_to_cyr(hay)),
            ]
        )
    )


def _form_matches_hay(
    form: str,
    hay: str,
    *,
    use_translit: bool = True,
    match_spaced: bool = True,
) -> bool:
    """Форма (опц. транслит) есть в hay — только word-boundary regex.

    Без голого ``v in hay``: короткие транслиты («dry»→«дри») иначе
    ловятся внутри слов («голодриги»).
    """
    if not hay:
        return False
    for v in _form_script_variants(form, use_translit=use_translit):
        if _compile_form(v, spaced=match_spaced).search(hay):
            return True
    return False


def _first_word_form(form: str) -> str:
    """Первое слово многословной формы (пробел/дефис); однословная — как есть.

    «Цитронный Магарача» → «Цитронный»; «Абрау-Дюрсо» → «Абрау»;
    «ESSE» → «ESSE».
    """
    raw = str(form or "").strip()
    if not raw:
        return ""
    parts = [p for p in re.split(r"[\s\-]+", raw) if p]
    if len(parts) <= 1:
        return raw
    return parts[0]


def _forms_first_word_only(forms: list[str]) -> list[str]:
    """Схлопнуть формы до первого слова; пустые отбросить, порядок сохранить."""
    out: list[str] = []
    seen: set[str] = set()
    for f in forms:
        fw = _first_word_form(f) or str(f or "").strip()
        if not fw:
            continue
        key = normalize_ocr_text(fw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(fw)
    return out


def find_lexicon_canons(
    text: str,
    lexicon: str,
    *,
    use_translit: bool = True,
    match_spaced: bool = True,
    first_hit_only: bool = False,
    longest_only: bool | None = None,
) -> dict[str, list[str]]:
    """Return {canon: [matched_forms...]} for hits in text.

    Сравнение в нижнем регистре. Для grape/winery флаги use_translit /
    match_spaced задаются из настроек сканера. Для category/type вызывающий
    код обязан передать False/False (без транслита и spaced).

    first_hit_only=True (grape/winery на запросе): формы по убыванию частоты
    в каталоге (tie-break — длина), после первого совпадения остальные
    каноны не проверяются.

    grape/winery: в результат попадает только первое слово найденной формы
    («Цитронный Магарача» → «Цитронный») — им же проверяют кандидата.

    longest_only — устаревший алиас first_hit_only.
    """
    if longest_only is not None:
        first_hit_only = bool(longest_only)
    hay = normalize_ocr_text(text or "")
    if not hay:
        return {}
    hay_x = _expand_hay(hay, use_translit=use_translit)
    entries = _FORM_RES.get(lexicon) or []
    gw = lexicon in ("grape", "winery")

    def _pack(canon: str, forms: list[str]) -> dict[str, list[str]]:
        use_forms = _forms_first_word_only(forms) if gw else forms
        if not use_forms:
            return {}
        return {canon: use_forms}

    if first_hit_only:
        freq = _LEXICON_FREQ.get(lexicon) or {}
        flat: list[tuple[int, int, str, str]] = []
        for canon, form_res in entries:
            for form, _cre in form_res:
                key = normalize_ocr_text(form).strip()
                if not key:
                    continue
                flat.append(
                    (int(freq.get(canon, 0)), len(key), form, canon)
                )
        # Чаще в каталоге → раньше; при равной частоте — длиннее форма.
        flat.sort(key=lambda t: (-t[0], -t[1], t[2].lower(), t[3]))
        for _freq, _n, form, canon in flat:
            if _form_matches_hay(
                form,
                hay_x,
                use_translit=use_translit,
                match_spaced=match_spaced,
            ):
                return _pack(canon, [form])
        return {}
    out: dict[str, list[str]] = {}
    for canon, form_res in entries:
        hits: list[str] = []
        for form, _cre in form_res:
            if _form_matches_hay(
                form, hay_x, use_translit=use_translit, match_spaced=match_spaced
            ):
                hits.append(form)
        if hits:
            packed = _pack(canon, hits)
            if packed:
                out.update(packed)
    return out


def _groups_from_json(raw: Any, key: str) -> list[tuple[str, tuple[str, ...]]]:
    items = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"exclusive_lexicon.json: ключ {key!r} должен быть массивом")
    out: list[tuple[str, tuple[str, ...]]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"exclusive_lexicon.json: {key}[{i}] должен быть объектом")
        canon = str(item.get("canon") or "").strip()
        forms_raw = item.get("forms")
        if not canon:
            raise ValueError(f"exclusive_lexicon.json: {key}[{i}].canon пустой")
        if not isinstance(forms_raw, list) or not forms_raw:
            raise ValueError(f"exclusive_lexicon.json: {key}[{i}].forms — непустой массив")
        forms = tuple(str(f).strip() for f in forms_raw if str(f).strip())
        if not forms:
            raise ValueError(f"exclusive_lexicon.json: {key}[{i}].forms пустой после trim")
        out.append((canon, forms))
    return out


def _compile_lexicon_groups(
    groups: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, list[tuple[str, re.Pattern[str]]]]]:
    return [
        (canon, [(form, _compile_form(form)) for form in forms])
        for canon, forms in groups
    ]


def load_static_lexicons(*, path: Path | None = None) -> Path:
    """Load category/type from exclusive_lexicon.json and compile matchers."""
    global _CATEGORY_GROUPS, _TYPE_GROUPS
    src = path or _LEXICON_JSON
    raw = json.loads(src.read_text(encoding="utf-8"))
    category = _groups_from_json(raw, "category")
    typ = _groups_from_json(raw, "type")
    _CATEGORY_GROUPS[:] = category
    _TYPE_GROUPS[:] = typ
    _LEXICONS["category"] = _CATEGORY_GROUPS
    _LEXICONS["type"] = _TYPE_GROUPS
    _FORM_RES["category"] = _compile_lexicon_groups(_CATEGORY_GROUPS)
    _FORM_RES["type"] = _compile_lexicon_groups(_TYPE_GROUPS)
    msg = (
        f"static lexicon loaded from {src.name}: "
        f"category={len(_CATEGORY_GROUPS)} type={len(_TYPE_GROUPS)}"
    )
    logger.info(msg)
    _console_print(f"[exclusive_lexicon] {msg}")
    return src


# Load at import so unit checks / early calls work before lifespan.
try:
    load_static_lexicons()
except Exception:  # noqa: BLE001
    logger.exception("failed to load exclusive_lexicon.json at import")
    _console_print("[exclusive_lexicon] FAILED to load exclusive_lexicon.json")


def _split_grape_tokens(raw: Any) -> list[str]:
    """Разбить grape_variety / label_ocr.cupage на отдельные сорта."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            out.extend(_split_grape_tokens(item))
        return out
    if isinstance(raw, dict):
        # редкий случай: {"name": "..."} / вложенность
        for key in ("name", "grape", "value", "text"):
            if key in raw:
                return _split_grape_tokens(raw.get(key))
        return []
    s = str(raw).strip()
    if not s or s.lower() in {"null", "none", "-", "—", "n/a"}:
        return []
    # запятая / ; / bullet / слэш / " - " / " + " между сортами
    # (дефис внутри слова «Каберне-Совиньон» не трогаем — нет пробелов вокруг)
    parts = re.split(r"[,;•·|/]+|\s+[-–—+]\s+", s)
    return [p.strip(" .\t") for p in parts if p and p.strip(" .\t")]


def _cupage_from_label_ocr(label_ocr: Any) -> Any:
    """wines.label_ocr: cupage (факт в БД) или coupage (алиас)."""
    if not isinstance(label_ocr, dict):
        return None
    if "cupage" in label_ocr and label_ocr.get("cupage") not in (None, ""):
        return label_ocr.get("cupage")
    if "coupage" in label_ocr and label_ocr.get("coupage") not in (None, ""):
        return label_ocr.get("coupage")
    return None


def load_grape_variety_lexicon(db: Session) -> list[str]:
    """Load unique grapes from wines.grape_variety + label_ocr.cupage/coupage.

    Частота канона = число вин, у которых токен встретился в grape_variety
    и/или cupage/coupage (вино считается один раз на канон).
    """
    global _GRAPE_FORMS
    by_canon: dict[str, str] = {}
    freq: dict[str, int] = {}

    def _canon_of(form: str) -> str | None:
        form = form.strip()
        if not form:
            return None
        canon = normalize_ocr_text(form).strip()
        if len(canon) < 3:
            return None
        prev = by_canon.get(canon)
        if prev is None or len(form) > len(prev):
            by_canon[canon] = form
        return canon

    wines = db.execute(
        select(Wine.grape_variety, Wine.label_ocr)
    ).all()
    n_gv_raw = 0
    n_cupage_raw = 0
    n_cupage_wines = 0
    n_gv_wines = 0
    for gv_raw, lo in wines:
        wine_canons: set[str] = set()
        gv_toks = _split_grape_tokens(gv_raw) if gv_raw else []
        if gv_toks:
            n_gv_wines += 1
            n_gv_raw += len(gv_toks)
            for part in gv_toks:
                c = _canon_of(part)
                if c:
                    wine_canons.add(c)
        cup = _cupage_from_label_ocr(lo)
        if cup is not None:
            cup_toks = _split_grape_tokens(cup)
            if cup_toks:
                n_cupage_wines += 1
                n_cupage_raw += len(cup_toks)
                for part in cup_toks:
                    c = _canon_of(part)
                    if c:
                        wine_canons.add(c)
        for c in wine_canons:
            freq[c] = freq.get(c, 0) + 1

    # Чаще в каталоге → раньше; tie-break — длина формы.
    ordered = sorted(
        by_canon.items(),
        key=lambda kv: (-int(freq.get(kv[0], 0)), -len(kv[0]), kv[0]),
    )
    _FORM_RES["grape"] = [
        (canon, [(surface, _compile_form(surface))]) for canon, surface in ordered
    ]
    _GRAPE_FORMS = [surface for _, surface in ordered]
    _LEXICON_FREQ["grape"] = dict(freq)
    top = ", ".join(
        f"{by_canon[c]}×{freq[c]}"
        for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:8]
    )
    msg = (
        f"grape variety lexicon loaded: unique={len(_GRAPE_FORMS)} "
        f"(wines={len(wines)}; grape_variety wines={n_gv_wines} tokens={n_gv_raw}; "
        f"label_ocr.cupage/coupage wines={n_cupage_wines} tokens={n_cupage_raw}; "
        f"top={top})"
    )
    logger.info(msg)
    _console_print(f"[exclusive_lexicon] {msg}")
    forms_msg = "grape variety lexicon forms: " + ", ".join(_GRAPE_FORMS)
    logger.info(forms_msg)
    _console_print(f"[exclusive_lexicon] {forms_msg}")
    return list(_GRAPE_FORMS)


def _dedupe_repeated_words(form: str) -> str:
    """Убрать повторяющиеся слова внутри названия (порядок первого вхождения)."""
    out: list[str] = []
    seen: set[str] = set()
    for part in form.split():
        key = normalize_ocr_text(part).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(part)
    return " ".join(out)


def _winery_stop_key(part: str) -> str:
    """Ключ для стоп-слов: lower без OCR-омоглифов (S→Б ломает ESTATES)."""
    t = unicodedata.normalize("NFKD", part or "")
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.lower().replace("ё", "е").strip()


# Фиксированные стоп-слова винодельни (не бренды). Бренды вроде denisov/abrau
# не трогаем — даже если встречаются в нескольких написаниях одной строки справочника.
_WINERY_STOP_TOKENS: frozenset[str] = frozenset(
    {
        # RU
        "винодельня",
        "винодельни",
        "винодельныи",
        "семейная",
        "семеиная",
        "семейныи",
        "семеиныи",
        "фермерская",
        "фермерскои",
        "кубанская",
        "кубанскои",
        "шато",
        "дом",
        "хозяйство",
        "поместье",
        "усадьба",
        "имение",
        "завод",
        "кооператив",
        "центр",
        "энологии",
        "винныи",
        "виннои",
        "вин",
        "вина",
        "вино",
        "шампанских",
        "марочных",
        # EN / FR
        "winery",
        "wineries",
        "wine",
        "wines",
        "chateau",
        "domaine",
        "vineyard",
        "vineyards",
        "estate",
        "estates",
        "manor",
        "valley",
        "family",
        "brothers",
        "organic",
        "agricole",
        "production",
        "premium",
        "премиум",
        "original",
        "by",
        # частицы
        "de",
        "la",
        "di",
        "le",
        "of",
        "the",
    }
)


def _strip_winery_stopwords(form: str) -> str:
    """Вырезать только стоп-слова; брендовые токены оставляем."""
    parts = [p for p in (form or "").split() if p]
    if not parts:
        return ""
    kept = [p for p in parts if _winery_stop_key(p) not in _WINERY_STOP_TOKENS]
    return " ".join(kept)


def load_winery_lexicon(db: Session) -> list[str]:
    """Load unique wineries from wines.winery + label_ocr.producer.

    Из форм вырезаются только стоп-слова (винодельня / winery / chateau / …).
    Брендовые имена (Denisov, Abrau, Inkerman…) не удаляются.

    Частота канона = число вин, у которых после очистки winery и/или
    producer совпал с этим каноном (вино один раз на канон).
    """
    global _WINERY_FORMS
    from app.pipeline.ocr_match import _is_bad_producer_name

    by_canon: dict[str, str] = {}
    freq: dict[str, int] = {}

    def _canon_of(raw: str) -> str | None:
        form = _dedupe_repeated_words(str(raw).strip())
        if not form or _is_bad_producer_name(form):
            return None
        form = _strip_winery_stopwords(form)
        form = _dedupe_repeated_words(form)
        if not form or _is_bad_producer_name(form):
            return None
        canon = normalize_ocr_text(form).strip()
        if len(canon) < 4:
            return None
        prev = by_canon.get(canon)
        if prev is None or len(form) > len(prev):
            by_canon[canon] = form
        return canon

    wines = db.execute(select(Wine.winery, Wine.label_ocr)).all()
    n_winery_src = 0
    n_producer_src = 0
    n_producer_wines = 0
    for winery_raw, lo in wines:
        wine_canons: set[str] = set()
        if winery_raw:
            c = _canon_of(str(winery_raw))
            if c:
                wine_canons.add(c)
                n_winery_src += 1
        if isinstance(lo, dict):
            prod = lo.get("producer")
            if prod is not None and prod != "":
                n_producer_wines += 1
                c = _canon_of(str(prod))
                if c:
                    wine_canons.add(c)
                    n_producer_src += 1
        for c in wine_canons:
            freq[c] = freq.get(c, 0) + 1

    ordered = sorted(
        by_canon.items(),
        key=lambda kv: (-int(freq.get(kv[0], 0)), -len(kv[0]), kv[0]),
    )
    _FORM_RES["winery"] = [
        (canon, [(surface, _compile_form(surface))]) for canon, surface in ordered
    ]
    _WINERY_FORMS = [surface for _, surface in ordered]
    _LEXICON_FREQ["winery"] = dict(freq)
    top = ", ".join(
        f"{by_canon[c]}×{freq[c]}"
        for c, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:8]
    )
    msg = (
        f"winery lexicon loaded: unique={len(_WINERY_FORMS)} "
        f"(stopwords={len(_WINERY_STOP_TOKENS)}; wines={len(wines)}; "
        f"winery hits={n_winery_src}; "
        f"label_ocr.producer wines={n_producer_wines} accepted={n_producer_src}; "
        f"top={top})"
    )
    logger.info(msg)
    _console_print(f"[exclusive_lexicon] {msg}")
    forms_msg = "winery lexicon forms: " + ", ".join(_WINERY_FORMS)
    logger.info(forms_msg)
    _console_print(f"[exclusive_lexicon] {forms_msg}")
    return list(_WINERY_FORMS)


def get_grape_variety_forms() -> list[str]:
    return list(_GRAPE_FORMS)


def _flatten_forms(canon_map: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for forms in canon_map.values():
        for f in forms:
            key = normalize_ocr_text(f).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(f)
    return out


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute = 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if abs(len(a) - len(b)) > 2:
        return 99
    # Ensure a is shorter or equal for slightly less work
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, start=1):
        cur = [i]
        for j, ca in enumerate(a, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _tokens_near_equal(a: str, b: str) -> bool:
    """Одно слово: одинаковые, либо ±1 буква / 1 замена / оба (edit ≤ 2, |Δlen| ≤ 1).

    Короткие токены (<4) — только точное совпадение (не путать «сира»/«сиро» легко).
    Сравнение всегда в нижнем регистре.
    """
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    return _levenshtein(a, b) <= 2


def _forms_near_equal(form_a: str, form_b: str) -> bool:
    """Многословная форма: попарно токены near-equal, то же число слов."""
    wa = [t for t in normalize_ocr_text(form_a).split() if t]
    wb = [t for t in normalize_ocr_text(form_b).split() if t]
    if not wa or not wb or len(wa) != len(wb):
        return False
    return all(_tokens_near_equal(x, y) for x, y in zip(wa, wb))


def _form_in_text_fuzzy(
    form: str,
    text: str,
    *,
    use_translit: bool = True,
    match_spaced: bool = True,
) -> bool:
    """Форма есть в тексте: regex и/или near-equal окно."""
    variants = _form_script_variants(form, use_translit=use_translit)
    if not variants:
        return True
    hay = normalize_ocr_text(text or "")
    if not hay:
        return False
    hay_x = _expand_hay(hay, use_translit=use_translit)
    for v in variants:
        if _form_matches_hay(
            v, hay_x, use_translit=False, match_spaced=match_spaced
        ):
            return True
        q = [t for t in v.split() if t]
        if not q:
            continue
        joined = " ".join(q)
        if joined in hay_x:
            return True
        for hay_part in hay_x.split("\n"):
            t_words = [t for t in hay_part.split() if t]
            n = len(q)
            if n > len(t_words):
                continue
            for i in range(len(t_words) - n + 1):
                window = t_words[i : i + n]
                if all(_tokens_near_equal(qw, tw) for qw, tw in zip(q, window)):
                    return True
    return False


def _missing_forms(
    query_forms: list[str],
    catalog_forms: list[str],
    *,
    catalog_text: str | None = None,
    fuzzy: bool = False,
    use_translit: bool = True,
    match_spaced: bool = True,
) -> list[str]:
    """Формы из OCR искомого, которых нет у кандидата."""
    if not query_forms:
        return []
    present: set[str] = set()
    catalog_norm_forms: list[str] = []
    for f in catalog_forms:
        for v in _form_script_variants(f, use_translit=use_translit):
            present.add(v)
            catalog_norm_forms.append(v)
    present.discard("")
    missing: list[str] = []
    seen: set[str] = set()
    for f in query_forms:
        key = normalize_ocr_text(f).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        q_vars = _form_script_variants(f, use_translit=use_translit)
        if any(v in present for v in q_vars):
            continue
        if fuzzy:
            if any(
                _forms_near_equal(qv, cf)
                for qv in q_vars
                for cf in catalog_norm_forms
            ):
                continue
            if catalog_text and _form_in_text_fuzzy(
                f,
                catalog_text,
                use_translit=use_translit,
                match_spaced=match_spaced,
            ):
                continue
        missing.append(f)
    return missing


def wine_catalog_blobs(wine: dict[str, Any]) -> dict[str, str]:
    """Где искать слова справочника у кандидата.

    category — этикетка + Category;
    type — этикетка + name (часто «Брют» / dry в названии);
    grape — только текст этикетки (grape_variety карточки не засчитывается);
    winery — только текст этикетки (поле winery карточки не засчитывается).
    """
    name = str(wine.get("name") or "")
    label = str(wine.get("label") or "")
    category = str(wine.get("category") or "")
    category_blob = "\n".join(x for x in (label, category) if x)
    type_blob = "\n".join(x for x in (label, name) if x)
    # Сорт и винодельня: форма из OCR искомого обязана быть в тексте этикетки.
    # Колонки grape_variety / winery не подставляют слово, которого на этикетке нет.
    grape_blob = label
    winery_blob = label
    return {
        "category": category_blob,
        "type": type_blob,
        "grape": grape_blob,
        "winery": winery_blob,
    }


def check_wine_exclusive(
    query_hits: dict[str, dict[str, list[str]]],
    wine: dict[str, Any],
    *,
    use_translit: bool = True,
    match_spaced: bool = True,
) -> dict[str, Any] | None:
    """If query lexicon form missing on candidate — return detail; else None."""
    blobs = wine_catalog_blobs(wine)
    conflicts: list[dict[str, Any]] = []
    for lex in _ALL_LEXICONS:
        q_map = query_hits.get(lex) or {}
        if not q_map:
            continue
        blob = blobs.get(lex) or ""
        # category/type: только точные формы (без spaced/транслита) —
        # иначе dry→дри ловится в «голодриги», red spaced ломает EN-формы.
        gw = lex in ("grape", "winery")
        lex_translit = bool(use_translit) if gw else False
        lex_spaced = bool(match_spaced) if gw else False
        c_map = find_lexicon_canons(
            blob,
            lex,
            use_translit=lex_translit,
            match_spaced=lex_spaced,
        )
        q_forms = _flatten_forms(q_map)
        c_forms = _flatten_forms(c_map)
        if gw:
            # Проверка только по первому слову многословных форм.
            q_forms = _forms_first_word_only(q_forms)
            c_extra: list[str] = []
            for f in c_forms:
                fw = _first_word_form(f)
                if fw and fw != f:
                    c_extra.append(fw)
            if c_extra:
                c_forms = list(c_forms) + c_extra
        fuzzy = gw
        missing_forms = _missing_forms(
            q_forms,
            c_forms,
            catalog_text=blob if fuzzy else None,
            fuzzy=fuzzy,
            use_translit=lex_translit,
            match_spaced=lex_spaced,
        )
        if not missing_forms:
            continue
        conflicts.append(
            {
                "lexicon": lex,
                "rule": "query_form_missing_on_catalog",
                "query_canons": sorted(q_map.keys()),
                "query_forms": {k: v for k, v in q_map.items()},
                "catalog_canons": sorted(c_map.keys()),
                "catalog_forms": {k: v for k, v in c_map.items()},
                "missing_forms": missing_forms,
                "fuzzy_match": fuzzy,
            }
        )
    if not conflicts:
        return None
    return {
        "wine_id": int(wine["id"]),
        "name": wine.get("name"),
        "conflicts": conflicts,
    }


def evaluate_exclusive_lexicon(
    query_text: str,
    wines: list[dict[str, Any]],
    *,
    use_translit: bool = True,
    match_spaced: bool = True,
) -> dict[str, Any]:
    """Compare query OCR vs candidate wines.

    Для каждого справочника: форма, найденная в OCR, обязана быть у кандидата
    (этикетка + релевантные поля). Чужой канон отдельно не проверяется.
    use_translit / match_spaced — только для grape и winery (из настроек сканера).
    """
    t0 = time.perf_counter()
    # category/type — без translit/spaced (короткие EN-формы иначе дают FP)
    q_cat = find_lexicon_canons(
        query_text, "category", use_translit=False, match_spaced=False
    )
    q_type = find_lexicon_canons(
        query_text, "type", use_translit=False, match_spaced=False
    )
    q_grape = (
        find_lexicon_canons(
            query_text,
            "grape",
            use_translit=use_translit,
            match_spaced=match_spaced,
            first_hit_only=True,
        )
        if _FORM_RES.get("grape")
        else {}
    )
    q_winery = (
        find_lexicon_canons(
            query_text,
            "winery",
            use_translit=use_translit,
            match_spaced=match_spaced,
            first_hit_only=True,
        )
        if _FORM_RES.get("winery")
        else {}
    )
    query_hits = {
        "category": q_cat,
        "type": q_type,
        "grape": q_grape,
        "winery": q_winery,
    }
    rejected: list[dict[str, Any]] = []
    rejected_ids: list[int] = []
    for w in wines:
        if w.get("id") is None:
            continue
        detail = check_wine_exclusive(
            query_hits,
            w,
            use_translit=use_translit,
            match_spaced=match_spaced,
        )
        if detail is None:
            continue
        rejected.append(detail)
        rejected_ids.append(int(detail["wine_id"]))
    ms = round((time.perf_counter() - t0) * 1000, 1)
    spaced_note = (
        "0–2 пробела/дефиса между буквами («красный» ≈ «кра с  ны й»)"
        if match_spaced
        else "без вставок пробелов между буквами"
    )
    translit_note = (
        "транслит RU↔LAT (денисов ↔ denisov)"
        if use_translit
        else "без транслитерации"
    )
    rule = (
        "если в OCR искомого найдена форма справочника (category/type/grape/winery), "
        "та же форма обязана быть на этикетке кандидата "
        "(category ещё смотрит поле category, type — name; "
        "grape и winery — только текст этикетки); "
        "grape/winery на запросе: формы по убыванию частоты в каталоге "
        "(tie-break — длина), после первого хита остальные каноны не проверяются; "
        "grape/winery: многословная форма в проверке → только первое слово "
        "(«Цитронный Магарача»→«Цитронный», «Абрау-Дюрсо»→«Абрау»); "
        f"для grape/winery: {spaced_note}; {translit_note}; "
        "для grape/winery допускается OCR-опечатка: ±1 буква и/или 1 замена "
        "в токене (edit≤2); "
        "category/type — точное совпадение форм (без spaced/транслита); "
        "синоним канона не засчитывается; чужой канон отдельно не проверяется"
    )
    return {
        "ok": True,
        "ms": ms,
        "use_translit": bool(use_translit),
        "match_spaced": bool(match_spaced),
        "query_hits": {
            "category": {k: v for k, v in q_cat.items()},
            "type": {k: v for k, v in q_type.items()},
            "grape": {k: v for k, v in q_grape.items()},
            "winery": {k: v for k, v in q_winery.items()},
        },
        "lexicons": {
            "category": {
                "field": "category",
                "source": "exclusive_lexicon.json",
                "note": "форма из OCR обязана быть у кандидата (label + category); "
                "0–2 пробела между буквами",
                "forms": [f for _, forms in _CATEGORY_GROUPS for f in forms],
            },
            "type": {
                "field": None,
                "source": "exclusive_lexicon.json",
                "note": "нет колонки type — label + name; та же форма из OCR обязана быть; "
                "0–2 пробела между буквами",
                "forms": [f for _, forms in _TYPE_GROUPS for f in forms],
            },
            "grape": {
                "field": "grape_variety",
                "note": (
                    "unique grape_variety + label_ocr.cupage/coupage при старте; "
                    "частота = число вин с этим сортом; на запросе: самый частый "
                    "хит, остальные сорта не ищутся; многословный сорт → в проверке "
                    "только первое слово; та же форма обязана быть "
                    "в тексте этикетки кандидата (колонка grape_variety не засчитывается); "
                    f"fuzzy ±1–2 буквы; {spaced_note}; {translit_note}"
                ),
                "forms": list(_GRAPE_FORMS),
                "count": len(_GRAPE_FORMS),
                "query_mode": "freq_first_hit",
                "check_mode": "first_word_only",
            },
            "winery": {
                "field": "winery",
                "note": (
                    "unique winery + label_ocr.producer при старте; "
                    "повтор слов внутри названия схлопнут; "
                    "стоп-слова (винодельня/winery/chateau/…) вырезаны, бренды сохранены; "
                    "частота = число вин с этой винодельней; на запросе: самый "
                    "частый хит, остальные винодельни не ищутся; многословное "
                    "название → в проверке только первое слово; та же форма "
                    "обязана быть в тексте этикетки кандидата; "
                    f"fuzzy ±1–2 буквы; {spaced_note}; {translit_note}"
                ),
                "forms": list(_WINERY_FORMS),
                "count": len(_WINERY_FORMS),
                "query_mode": "freq_first_hit",
                "check_mode": "first_word_only",
            },
        },
        "checked": len(wines),
        "rejected_ids": rejected_ids,
        "rejected": rejected[:40],
        "reject_count": len(rejected_ids),
        "rule": rule,
    }
