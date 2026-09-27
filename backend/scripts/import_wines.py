"""Import wines from CSV into the wines table, linking wineries_id."""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from app.db.config import DATABASE_URL
from app.db.models import Wine, Winery

DEFAULT_CSV = PROJECT / "strapi_output0709.csv"

PREFIX_RE = re.compile(
    r"^(фото:\s*|команда\s+винодельни\s+|семейная\s+винодельня\s+|винный\s+кооператив\s+|винодельня\s+|вина\s+винодельни\s+|вина\s+|усадьба\s+)",
    re.IGNORECASE,
)
NON_ALNUM_RE = re.compile(r"[^0-9a-zа-яё]+", re.IGNORECASE)
CYR_LAT = str.maketrans(
    {
        "а": "a",
        "в": "v",
        "е": "e",
        "к": "k",
        "м": "m",
        "н": "n",
        "о": "o",
        "р": "r",
        "с": "s",
        "т": "t",
        "х": "h",
        "у": "u",
        "р": "r",
    }
)

# Explicit aliases: CSV winery name -> key that matches a winery record
ALIASES: dict[str, str] = {
    "валерий захарьин": "valeriy zaharin",
    "золотая балка": "zolotaya balka",
    "фанагория": "fanagoriya",
    "коммуналка": "kommunalka",
    "усадьба родное гнездо": "usadba rodnoe gnezdo",
    "родное гнездо": "usadba rodnoe gnezdo",
    "абрау дюрсо": "abrau dyurso",
    "абрау-дюрсо": "abrau dyurso",
    "галицкий и галицкий": "galiczkij i galiczkij",
    "шато де талю": "chateau de talu",
    "шато пино": "shato pino",
    "шато ай даниль": "shato ay danil",
    "шато ай-даниль": "shato ay danil",
    "мысхако": "vinodelnya myshako",
    "ведерниковъ": "vinodelnya vedernikov",
    "ведерников": "vinodelnya vedernikov",
    "кубань вино": "kuban vino",
    "кубань-вино": "kuban vino",
    "долина лефкадия": "dolina lefkadiya",
    "лефкадия": "dolina lefkadiya",
    "массандра": "massandra",
    "инкерман": "inkermanskiy zmv",
    "солнечная долина": "solnechnaya dolina",
    "новый свет": "noviy svet",
    "гаврас": "gavras",
    "шумринка": "shumrinka",
    "криница": "vinodelnya krinitsa",
    "марко": "vinodelnya marko",
    "марченко": "vinodelnya marchenko",
    "жаков": "vinodelnya zhakov",
    "покровская": "vinodelnya pokrovskaya",
    "батрак": "vinodelnya batrak",
    "молчановы": "vinodelnya molchanova",
    "узунов": "vinodelnya uzunov",
    "берильдаева": "vinodelnya begildeeva",
    "бегильдеева": "vinodelnya begildeeva",
    "братьев мельниковых": "vinodelnya bratev melnikovih",
    "винодельня братьев мельниковых": "vinodelnya bratev melnikovih",
    "цмлянские вина": "tsimlyanskie vina",
    "цимлянские вина": "tsimlyanskie vina",
    "вино и небо": "vino nebo",
    "уркаста": "villa urkusta",
    "вилла уркуста": "villa urkusta",
    "вилла софия": "villa sofiya",
    "уссадьба перовских": "usadba perovskih",
    "усадьба перовских": "usadba perovskih",
    "усадьба мекотан": "usadba merkotan",
    "усадьба меркотан": "usadba merkotan",
    "усадьба маскага": "usadba maskaga",
    "усадьба мангуп": "usadba mangup",
    "усадьба саркел": "usadba sarkel",
    "усадьба мезыбь": "usadba mezyb",
    "усадьба дивноморское": "usadba divnomorskoe",
    "усадьба маркотх": "usadba markoth",
    "усадьба аммонит": "usadba ammonit",
    "шато дюрсо": "usadba shato dyurso chateau durso",
    "chateau durso": "usadba shato dyurso chateau durso",
    "дербент вино": "derbent vino",
    "гай кодзор": "vinogradniki gay kodzora",
    "гай-кодзор": "vinogradniki gay kodzora",
    "виноградники гай кодзора": "vinogradniki gay kodzora",
    "эльбузд": "donskoe vinodelcheskoe hozyaystvo elbuzd",
    "вина арпачина": "vina arpachina",
    "два петра": "dva petra",
    "два сердца": "dva serdtsa",
    "сикоры": "imenie sikory",
    "имение сикоры": "imenie sikory",
    "реликта": "relikta",
    "раевское": "raevskoe",
    "левокумское": "levokumskoe",
    "плиогория": "vinodelnya pliogoriya",
    "коктебель": "zmv koktebel",
    "звм коктебель": "zmv koktebel",
    "агролайн": "agrolayn",
    "темпельхофф": "tempelhof winery",
    "темпельхоф": "tempelhof winery",
    "вайкрафт": "vajn kraft",
    "вайн крафт": "vajn kraft",
    "lorio": "lorio semeinaya vinodelnya logunovyh",
    "лорио": "lorio semeinaya vinodelnya logunovyh",
    "белбек": "belbek",
    "бельбек": "belbek",
    "домино бонами": "domaine bonami",
    "domaine bonami": "domaine bonami",
    "domaine lipko": "domaine lipko",
    "липко": "domaine lipko",
    "vivandiere": "domaine de la vivandiere",
    "le grand vostock": "chateau le grand vostock",
    "шато ле гран восток": "chateau le grand vostock",
    "yaiyla": "yaiyla winery",
    "дача сердюка": "dacha serdyuka",
    "вилла ди альма": "villa di alma",
    "тристория": "tristoriya",
    "шато алвиса": "shato alvisa",
    "уппа": "uppa winery",
    "уссадьба белогорье": "manufacturer",
    "усадьба белогорье": "manufacturer",
    "дом gale": "dom gale dom gale",
    "dom gale": "dom gale dom gale",
    "винный форт адагум": "vinnyy fort adagum",
    "сенетх": "seneth",
    "гусевъ": "gusev",
    "гусев": "gusev",
    "нестеров": "nesterov winery",
    "андрей орлов": "andrey orlov",
    "константин дзитоев": "vinodelnya dzitoeva",
    "дзитоев": "vinodelnya dzitoeva",
    "илья защук": "chateau cachalot",
    "chateau cachalot": "chateau cachalot",
    "jd winery": "jd winery",
    "winepark": "winepark",
    "winemafia": "winemafia",
    "51 параллель": "51 parallel winery",
    "urban winery": "urban winery",
    "stn winery": "stn winery",
    "винодельня 78": "vinodelnya 78",
    "gunko winery": "gunko winery",
    "denisov winery": "denisov winery",
    "dubinin winery": "dubinin winery",
    "ferrum winery": "ferrum winery",
    "cellar master": "cellar master",
    "cock t'est belle": "cock test belle",
    "uva vallis": "uva vallis",
    "legato": "legato",
    "millstream": "millstream",
    "милльстрим": "millstream",
    "andryus yutsis": "andryus yutsis",
    "cloudy winery": "cloudy winery",
    "elpa winery": "elpa winery",
    "leto": "leto",
    "ароматное": "aromatnoe",
    "v2r": "v2r",
    "vinabani": "vinabani",
    "belmas winery": "belmas winery",
    "bakla vines": "bakla vines",
    "agora winery": "agora winery",
    "агора": "agora winery",
    "mantra": "mantra estate",
    "mantra estate": "mantra estate",
    "loco cimbali": "loco cimbali",
    "oleg repin": "oleg repin",
    "олег репин": "oleg repin",
    "mons albus": "mons albus",
    "artvin": "artvin",
    "артвин": "artvin",
    "esse": "esse",
    "kalos limen": "kalos limen",
    "благолюбов": "blagolyubov",
    "alma valley": "alma valley",
    "rem akchurin": "rem akchurin",
    "акчурин": "rem akchurin",
    "aratti": "aratti",
    "аратти": "aratti",
    "симферопольский винзавод": "simferopolskij vinodelcheskij zavod",
    "литовщук": "litavshchuk vineyards and winery",
    "litavshchuk": "litavshchuk vineyards and winery",
    "aya": "aya organic wine vineyards",
    "aya organic wine & vineyards": "aya organic wine vineyards",
    "бюрнье": "vinodelnya byurne",
    "скалистый берег": "skalistyy bereg",
    "golubitskoe estate": "golubitskoe estate",
    "голубицкое": "golubitskoe estate",
    "vibes": "vibes",
    "pithos": "pithos",
    "satera": "satera",
    "сатера": "satera",
    "radio wine": "radio wine",
    "katharon": "katharon",
    "olymp winery": "olymp winery",
    "soyuz vino": "soyuz vino",
    "союз вино": "soyuz vino",
    "табия": "tabiya",
    "виктор сташко": "viktor stashko",
    "ivan ksenia kruz": "ivan ksenia kruz",
    "ivan & ksenia kruz": "ivan ksenia kruz",
    "oxana istratova wine": "oxana istratova wine",
    "one barrel": "one barrel uan barrel",
    "зимовец": "zimovec",
    "интуиция": "vinnyy kooperativ intuitsiya",
    "винный кооператив интуиция": "vinnyy kooperativ intuitsiya",
    "бердяева": "vinodelnya berdyaeva",
    "собер баш": "sober bash",
    "золотое поле": "zolotoe pole",
    "усадьба александровская": "usadba alexandrovskaya",
    "николаев и сыновья": "nikolaev i synovia",
    "а гордиенко м николаев": "a gordienko m nikolaev",
    "а гордиенко & м николаев": "a gordienko m nikolaev",
    "bogovich wine & vineyard": "bogovich wine vineyard",
    "bogovich wine and vineyard": "bogovich wine vineyard",
    "лорио семейная винодельня логуновых": "lorio semeinaya vinodelnya logunovyh",
    "лорио - семейная винодельня логуновых": "lorio semeinaya vinodelnya logunovyh",
    "вайнкрафт": "vajn kraft",
    "в2р": "v2r",
    "винодельня молчанова": "vinodelnya molchanova",
    "молчанова": "vinodelnya molchanova",
    "семейная винодельня михаила колесникова": "vinodelnya kolesnikova",
    "колесникова": "vinodelnya kolesnikova",
    "vino & nebo": "vino nebo",
    "vino and nebo": "vino nebo",
    "вино и небо": "vino nebo",
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip().lower()
    value = value.replace("ё", "е").replace("&", " and ").replace("—", " ").replace("–", " ")
    value = PREFIX_RE.sub("", value)
    value = value.replace("ъ", "").replace("ь", "")
    value = NON_ALNUM_RE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def latinish(value: str) -> str:
    return normalize(value).translate(CYR_LAT)


def slug_key(url_or_slug: str) -> str:
    slug = url_or_slug.rstrip("/").split("/")[-1]
    return normalize(slug.replace("_", " ").replace("-", " "))


def token_set(value: str) -> frozenset[str]:
    stop = {"and", "i", "the", "winery", "wine", "vineyard", "vineyards", "estate"}
    return frozenset(t for t in normalize(value).split() if t and t not in stop)


def build_winery_index(wineries: list[Winery]) -> tuple[dict[str, int], list[tuple[frozenset[str], int]]]:
    index: dict[str, int] = {}
    token_index: list[tuple[frozenset[str], int]] = []
    for w in wineries:
        keys = {
            normalize(w.name),
            latinish(w.name),
            slug_key(w.url),
        }
        for key in list(keys):
            keys.add(key.replace("vinodelnya ", "").replace("usadba ", "").strip())
            keys.add(key.replace("chateau ", "shato ").strip())
            keys.add(key.replace("shato ", "chateau ").strip())
            keys.add(key.replace(" and ", " ").strip())
        for key in keys:
            if key:
                index.setdefault(key, w.id)
        tokens = token_set(w.name) | token_set(slug_key(w.url))
        if tokens:
            token_index.append((tokens, w.id))
    for alias, target in ALIASES.items():
        alias_n = normalize(alias)
        target_n = normalize(target)
        if target_n in index:
            index[alias_n] = index[target_n]
            index[latinish(alias)] = index[target_n]
            index.setdefault(alias_n.replace(" and ", " "), index[target_n])
    return index, token_index


def resolve_winery_id(
    name: str,
    index: dict[str, int],
    token_index: list[tuple[frozenset[str], int]],
) -> int | None:
    key = normalize(name)
    candidates = [key, latinish(name), key.replace(" and ", " ")]
    for cand in candidates:
        if cand in index:
            return index[cand]
        if cand in ALIASES:
            target = normalize(ALIASES[cand])
            if target in index:
                return index[target]
    for cand, wid in index.items():
        if key and cand and (key in cand or cand in key) and min(len(key), len(cand)) >= 5:
            return wid
    name_tokens = token_set(name)
    if len(name_tokens) >= 1:
        best_id = None
        best_score = 0.0
        for tokens, wid in token_index:
            if not tokens:
                continue
            inter = len(name_tokens & tokens)
            if inter == 0:
                continue
            score = inter / max(len(name_tokens), len(tokens))
            if score > best_score:
                best_score = score
                best_id = wid
        if best_score >= 0.5:
            return best_id
    return None


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        wineries = list(session.scalars(select(Winery)))
        if not wineries:
            raise SystemExit("wineries table is empty — run import_wineries.py first")
        index, token_index = build_winery_index(wineries)

        session.execute(delete(Wine))
        session.flush()

        seen: set[tuple] = set()
        inserted = 0
        linked = 0
        unmatched: dict[str, int] = {}

        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("Название вина") or "").strip()
                category = (row.get("Категория") or "").strip() or None
                color = (row.get("Цвет") or "").strip() or None
                region = (row.get("Регион") or "").strip() or None
                grape = (row.get("Сорт винограда") or "").strip() or None
                description = (row.get("Описание") or "").strip() or None
                winery_name = (row.get("Винодельня") or "").strip() or None
                slug = (row.get("Slug") or "").strip() or None
                photo_name = (row.get("Название фото") or "").strip() or None

                key = (name, category, color, region, grape, description, winery_name, slug, photo_name)
                if key in seen:
                    continue
                seen.add(key)

                wid = resolve_winery_id(winery_name or "", index, token_index) if winery_name else None
                if wid:
                    linked += 1
                elif winery_name:
                    unmatched[winery_name] = unmatched.get(winery_name, 0) + 1

                session.add(
                    Wine(
                        wineries_id=wid,
                        name=name or (slug or "unnamed"),
                        category=category,
                        color=color,
                        region=region,
                        grape_variety=grape,
                        description=description,
                        winery=winery_name,
                        slug=slug,
                        photo_name=photo_name,
                    )
                )
                inserted += 1

        session.commit()
        print(f"Imported {inserted} wines ({linked} linked to wineries)")
        if unmatched:
            print(f"Unmatched winery names ({len(unmatched)}):")
            for name, count in sorted(unmatched.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {count:4d}  {name}")


if __name__ == "__main__":
    main()
