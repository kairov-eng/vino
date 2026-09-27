"""Import wineries from sitemap XML into the wineries table."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from app.db.config import DATABASE_URL
from app.db.models import Wine, Winery

NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "img": "http://www.google.com/schemas/sitemap-image/1.1",
}

DEFAULT_SITEMAP = PROJECT / "uploads" / "wineries_sitemap_c4680f8036.xml"

PREFIX_RE = re.compile(
    r"^(фото:\s*|винодельня\s+|вина\s+винодельни\s+|вина\s+)",
    re.IGNORECASE,
)


def clean_name(raw: str | None, slug: str) -> str:
    if not raw or not raw.strip():
        return slug.replace("_", "-").replace("-", " ").strip().title()
    name = raw.strip()
    name = PREFIX_RE.sub("", name).strip(" «»\"'")
    return name or slug.replace("-", " ").title()


def parse_sitemap(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    rows: list[dict] = []
    for url in root.findall("sm:url", NS):
        loc_el = url.find("sm:loc", NS)
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        slug = loc.rstrip("/").split("/")[-1]
        photo = None
        title = None
        img = url.find("img:image", NS)
        if img is not None:
            loc_img = img.find("img:loc", NS)
            title_el = img.find("img:title", NS)
            caption_el = img.find("img:caption", NS)
            if loc_img is not None and loc_img.text:
                photo = loc_img.text.strip()
            if title_el is not None and title_el.text:
                title = title_el.text
            elif caption_el is not None and caption_el.text:
                title = caption_el.text
        rows.append({"name": clean_name(title, slug), "url": loc, "photo": photo})
    return rows


def main() -> None:
    sitemap = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SITEMAP
    rows = parse_sitemap(sitemap)
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        session.execute(delete(Wine))
        session.execute(delete(Winery))
        session.flush()
        for row in rows:
            session.add(Winery(**row))
        session.commit()
        total = len(session.scalars(select(Winery)).all())
        print(f"Imported {total} wineries from {sitemap}")


if __name__ == "__main__":
    main()
