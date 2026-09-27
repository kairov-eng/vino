from datetime import datetime
from typing import Any

from sqlalchemy import (
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Winery(Base):
    __tablename__ = "wineries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    photo: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    wines: Mapped[list["Wine"]] = relationship(back_populates="winery_ref")

    __table_args__ = (
        Index("ix_wineries_id", "id"),
        Index("ix_wineries_name", "name"),
        Index("ix_wineries_url", "url"),
        Index("ix_wineries_photo", "photo"),
    )


class Wine(Base):
    __tablename__ = "wines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wineries_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("wineries.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    color: Mapped[str | None] = mapped_column(String(256), nullable=True)
    region: Mapped[str | None] = mapped_column(String(256), nullable=True)
    grape_variety: Mapped[str | None] = mapped_column(String(512), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    winery: Mapped[str | None] = mapped_column(String(512), nullable=True)
    slug: Mapped[str | None] = mapped_column(String(512), nullable=True)
    photo_name: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # 0 = нет файла, 1 = crop этикетки ок, 2 = этикетки нет (сохранён оригинал), -1 = ошибка
    crop: Mapped[int | None] = mapped_column(Integer, nullable=True, server_default="0")
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Структурированный OCR JSON (как search_photos.status.steps.ocr_*.json)
    label_ocr: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # L2-normalized HSV hist H×S 30×32, float32 row-major (как hsv_match.hsv_hist).
    hsv: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    winery_ref: Mapped[Winery | None] = relationship(back_populates="wines")

    __table_args__ = (
        Index("ix_wines_id", "id"),
        Index("ix_wines_wineries_id", "wineries_id"),
        Index("ix_wines_name", "name"),
        Index("ix_wines_category", "category"),
        Index("ix_wines_color", "color"),
        Index("ix_wines_region", "region"),
        Index("ix_wines_grape_variety", "grape_variety"),
        # Hash index: description can exceed btree key size limit
        Index("ix_wines_description", "description", postgresql_using="hash"),
        Index("ix_wines_winery", "winery"),
        Index("ix_wines_slug", "slug"),
        Index("ix_wines_photo_name", "photo_name"),
        Index("ix_wines_crop", "crop"),
    )


class WineSitemap(Base):
    """Raw meaningful rows from wines_sitemap.xml (loc → image)."""

    __tablename__ = "wines_sitemap"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loc: Mapped[str] = mapped_column(String(1024), nullable=False)
    slug: Mapped[str] = mapped_column(String(512), nullable=False)
    lastmod: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    image_loc: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    image_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    image_title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    image_caption: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    __table_args__ = (
        Index("ix_wines_sitemap_id", "id"),
        Index("ix_wines_sitemap_loc", "loc"),
        Index("ix_wines_sitemap_slug", "slug"),
        Index("ix_wines_sitemap_lastmod", "lastmod"),
        Index("ix_wines_sitemap_image_loc", "image_loc"),
        Index("ix_wines_sitemap_image_file", "image_file"),
        Index("ix_wines_sitemap_image_title", "image_title"),
        Index("ix_wines_sitemap_image_caption", "image_caption"),
    )


class WinerySitemap(Base):
    """Raw meaningful rows from wineries_sitemap.xml (loc → image)."""

    __tablename__ = "wineries_sitemap"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loc: Mapped[str] = mapped_column(String(1024), nullable=False)
    slug: Mapped[str] = mapped_column(String(512), nullable=False)
    lastmod: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    image_loc: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    image_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    image_title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    image_caption: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    __table_args__ = (
        Index("ix_wineries_sitemap_id", "id"),
        Index("ix_wineries_sitemap_loc", "loc"),
        Index("ix_wineries_sitemap_slug", "slug"),
        Index("ix_wineries_sitemap_lastmod", "lastmod"),
        Index("ix_wineries_sitemap_image_loc", "image_loc"),
        Index("ix_wineries_sitemap_image_file", "image_file"),
        Index("ix_wineries_sitemap_image_title", "image_title"),
        Index("ix_wineries_sitemap_image_caption", "image_caption"),
    )


class SearchPhoto(Base):
    """Uploaded query photo for wine search pipeline."""

    __tablename__ = "search_photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="now()",
    )
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    filesize: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # Ручная отметка «Соответствие» на карточке кандидата (одно вино на поиск)
    manual_wines_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("wines.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Denormalized history fields (from status) for scan-history SQL filters/sort
    hist_wine_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hist_wine_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    hist_wine_slug: Mapped[str | None] = mapped_column(String(512), nullable=True)
    hist_total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    hist_fp: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, server_default="0")
    hist_fn: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, server_default="0")
    # YOLO confidence этикетки, выбранной для кропа и дальнейшего анализа.
    # Источник: status.steps.yolo.selected.conf. NULL, если этикетка не выбрана.
    coef: Mapped[float | None] = mapped_column(
        Float,
        Computed(
            "CASE "
            "WHEN jsonb_typeof(status #> '{steps,yolo,selected,conf}') = 'number' "
            "THEN (status #>> '{steps,yolo,selected,conf}')::double precision "
            "ELSE NULL END",
            persisted=True,
        ),
        nullable=True,
    )
    # Bhattacharyya HSV: искомая этикетка ↔ победитель.
    # Источник: status.steps.hsv.winner. NULL, если победителя или кропа нет.
    hsv: Mapped[float | None] = mapped_column(
        Float,
        Computed(
            "CASE "
            "WHEN jsonb_typeof(status #> '{steps,hsv,winner}') = 'number' "
            "THEN (status #>> '{steps,hsv,winner}')::double precision "
            "ELSE NULL END",
            persisted=True,
        ),
        nullable=True,
    )

    embeddings: Mapped[list["SearchPhotoEmbedding"]] = relationship(
        back_populates="search_photo", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_search_photos_id", "id"),
        Index("ix_search_photos_created_at", "created_at"),
        Index("ix_search_photos_sha256", "sha256"),
        Index("ix_search_photos_manual_wines_id", "manual_wines_id"),
        Index("ix_search_photos_hist_wine_id", "hist_wine_id"),
        Index("ix_search_photos_hist_wine_slug", "hist_wine_slug"),
        Index("ix_search_photos_hist_total_ms", "hist_total_ms"),
        Index("ix_search_photos_hist_wine_confidence", "hist_wine_confidence"),
        Index("ix_search_photos_hist_fp", "hist_fp"),
        Index("ix_search_photos_hist_fn", "hist_fn"),
        Index("ix_search_photos_coef", "coef"),
        Index("ix_search_photos_hsv", "hsv"),
    )


class SearchPhotoEmbedding(Base):
    """Visual embedding metadata for a search photo (vector stored via raw SQL)."""

    __tablename__ = "search_photo_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    search_photos_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("search_photos.id", ondelete="CASCADE"), nullable=False
    )
    embedding_type: Mapped[str] = mapped_column(String(64), nullable=False)

    search_photo: Mapped[SearchPhoto] = relationship(back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint(
            "search_photos_id",
            "embedding_type",
            name="ux_search_photo_embeddings_photo_type",
        ),
        Index("ix_search_photo_embeddings_search_photos_id", "search_photos_id"),
        Index("ix_search_photo_embeddings_embedding_type", "embedding_type"),
    )


class EmbeddingSiglip2(Base):
    """Label crop embedding (SigLIP2); vector column managed via raw SQL / Alembic."""

    __tablename__ = "embeddings_siglip2"

    wines_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("wines.id", ondelete="CASCADE"), primary_key=True
    )


class EmbeddingDinov3(Base):
    """Label crop embedding (DINOv3); vector column managed via raw SQL / Alembic."""

    __tablename__ = "embeddings_dinov3"

    wines_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("wines.id", ondelete="CASCADE"), primary_key=True
    )


class XgbExcludePhoto(Base):
    """Файлы, поиски по которым не включать в датасет XGBoost (match по sha256).

    sha256 — тот же алгоритм, что search_photos: hashlib.sha256(file_bytes).hexdigest().
    """

    __tablename__ = "xgb_exclude_photos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default="now()",
    )
    folder: Mapped[str] = mapped_column(String(2048), nullable=False)
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    filesize: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "filename",
            "sha256",
            name="ux_xgb_exclude_photos_filename_sha256",
        ),
        Index("ix_xgb_exclude_photos_sha256", "sha256"),
        Index("ix_xgb_exclude_photos_folder", "folder"),
    )
