"""Import wines_sitemap and wineries_sitemap from XML into PostgreSQL."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from app.db.config import DATABASE_URL
from app.db.models import WineSitemap, WinerySitemap

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "img": "http://www.google.com/schemas/sitemap-image/1.1",
}

DEFAULT_WINES = PROJECT / "uploads" / "wines_sitemap_d778a8e06a.xml"
DEFAULT_WINERIES = PROJECT / "uploads" / "wineries_sitemap_c4680f8036.xml"


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    value = el.text.strip()
    return value or None


def _parse_lastmod(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _image_file(image_loc: str | None) -> str | None:
    if not image_loc:
        return None
    path = unquote(urlparse(image_loc).path)
    name = path.rstrip("/").split("/")[-1]
    return name or None


def parse_sitemap(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    rows: list[dict] = []
    for url in root.findall("sm:url", NS):
        loc = _text(url.find("sm:loc", NS))
        if not loc:
            continue
        slug = loc.rstrip("/").split("/")[-1]
        lastmod = _parse_lastmod(_text(url.find("sm:lastmod", NS)))
        image_loc = image_title = image_caption = None
        img = url.find("img:image", NS)
        if img is not None:
            image_loc = _text(img.find("img:loc", NS))
            image_title = _text(img.find("img:title", NS))
            image_caption = _text(img.find("img:caption", NS))
        rows.append(
            {
                "loc": loc,
                "slug": slug,
                "lastmod": lastmod,
                "image_loc": image_loc,
                "image_file": _image_file(image_loc),
                "image_title": image_title,
                "image_caption": image_caption,
            }
        )
    return rows


def import_table(session: Session, model, path: Path) -> int:
    rows = parse_sitemap(path)
    session.execute(delete(model))
    session.flush()
    for row in rows:
        session.add(model(**row))
    session.commit()
    return len(session.scalars(select(model)).all())


def main() -> None:
    wines_xml = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WINES
    wineries_xml = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WINERIES

    engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})
    with Session(engine) as session:
        wines_n = import_table(session, WineSitemap, wines_xml)
        print(f"Imported {wines_n} rows into wines_sitemap from {wines_xml}")
        wineries_n = import_table(session, WinerySitemap, wineries_xml)
        print(f"Imported {wineries_n} rows into wineries_sitemap from {wineries_xml}")


if __name__ == "__main__":
    main()
