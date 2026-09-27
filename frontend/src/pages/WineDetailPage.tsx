import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchWine, type Wine } from '../api/client'
import {
  catalogHrefCategory,
  catalogHrefCategoryWithParents,
  catalogHrefColor,
  catalogHrefColorWithParents,
  catalogHrefGrape,
  catalogHrefGrapeWithParents,
  catalogHrefRegion,
  catalogHrefRegionWithWinery,
  catalogHrefWinery,
} from '../catalogFilters'
import './WineDetailPage.css'

function Field({
  label,
  value,
  valueHref,
  labelHref,
}: {
  label: string
  value?: string | null
  valueHref?: string | null
  labelHref?: string | null
}) {
  if (!value) return null
  return (
    <div className="wine-detail__field">
      {labelHref ? (
        <a
          className="wine-detail__link wine-detail__link--label"
          href={labelHref}
          target="_blank"
          rel="noopener noreferrer"
        >
          {label}
        </a>
      ) : (
        <span>{label}</span>
      )}
      {valueHref ? (
        <a
          className="wine-detail__link wine-detail__link--value"
          href={valueHref}
          target="_blank"
          rel="noopener noreferrer"
        >
          {value}
        </a>
      ) : (
        <strong>{value}</strong>
      )}
    </div>
  )
}

export function WineDetailPage() {
  const { slug = '' } = useParams()
  const [wine, setWine] = useState<Wine | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [imgBroken, setImgBroken] = useState(false)
  const [labelBroken, setLabelBroken] = useState(false)

  useEffect(() => {
    setWine(null)
    setError(null)
    setImgBroken(false)
    setLabelBroken(false)
    fetchWine(slug)
      .then(setWine)
      .catch((err) => setError(err instanceof Error ? err.message : 'Ошибка'))
  }, [slug])

  if (error) {
    return (
      <main className="page">
        <p className="page-error">{error}</p>
        <Link to="/wines">← К каталогу</Link>
      </main>
    )
  }

  if (!wine) {
    return (
      <main className="page">
        <p className="page-loading">Загрузка…</p>
      </main>
    )
  }

  const wineryHref = wine.winery ? catalogHrefWinery(wine.winery) : null
  const regionValueHref = wine.region ? catalogHrefRegion(wine.region) : null
  const regionLabelHref =
    wine.region || wine.winery
      ? catalogHrefRegionWithWinery(wine.region, wine.winery)
      : null
  const grapeValueHref = wine.grape_variety
    ? catalogHrefGrape(wine.grape_variety)
    : null
  const grapeLabelHref =
    wine.grape_variety || wine.region || wine.winery
      ? catalogHrefGrapeWithParents(
          wine.grape_variety,
          wine.region,
          wine.winery,
        )
      : null
  const categoryValueHref = wine.category
    ? catalogHrefCategory(wine.category)
    : null
  const categoryLabelHref =
    wine.category || wine.region || wine.winery
      ? catalogHrefCategoryWithParents(wine.category, wine.region, wine.winery)
      : null
  const colorValueHref = wine.color ? catalogHrefColor(wine.color) : null
  const colorLabelHref =
    wine.color || wine.region || wine.winery
      ? catalogHrefColorWithParents(wine.color, wine.region, wine.winery)
      : null

  return (
    <main className="page wine-detail">
      <div className="wine-detail__hero">
        <div className="wine-detail__left">
          <h1>{wine.name}</h1>
          {wine.winery && wineryHref && (
            <p className="wine-detail__winery">
              <a
                className="wine-detail__link wine-detail__link--value"
                href={wineryHref}
                target="_blank"
                rel="noopener noreferrer"
              >
                {wine.winery}
              </a>
            </p>
          )}
          <div className="wine-detail__stack">
            <Field
              label="Регион"
              value={wine.region}
              valueHref={regionValueHref}
              labelHref={regionLabelHref}
            />
            <Field
              label="Сорт винограда"
              value={wine.grape_variety}
              valueHref={grapeValueHref}
              labelHref={grapeLabelHref}
            />
            <Field
              label="Категория"
              value={wine.category}
              valueHref={categoryValueHref}
              labelHref={categoryLabelHref}
            />
            <Field
              label="Цвет"
              value={wine.color}
              valueHref={colorValueHref}
              labelHref={colorLabelHref}
            />
          </div>
        </div>

        <div className="wine-detail__photo">
          {wine.photo_url && !imgBroken ? (
            <img
              src={wine.photo_url}
              alt={wine.name}
              onError={() => setImgBroken(true)}
            />
          ) : (
            <div className="wine-detail__placeholder" />
          )}
        </div>

        <div className="wine-detail__right">
          <div className="wine-detail__card">
            <h2 className="wine-detail__card-title">
              <span>О вине</span>
              <span className="wine-detail__card-id">id {wine.id}</span>
            </h2>
            <Field label="Название" value={wine.name} />
            <Field
              label="Винодельня"
              value={wine.winery}
              valueHref={wineryHref}
            />
            <Field
              label="Регион"
              value={wine.region}
              valueHref={regionValueHref}
              labelHref={regionLabelHref}
            />
            <Field
              label="Сорт"
              value={wine.grape_variety}
              valueHref={grapeValueHref}
              labelHref={grapeLabelHref}
            />
            <Field
              label="Категория"
              value={wine.category}
              valueHref={categoryValueHref}
              labelHref={categoryLabelHref}
            />
            <Field
              label="Цвет"
              value={wine.color}
              valueHref={colorValueHref}
              labelHref={colorLabelHref}
            />
          </div>
        </div>
      </div>

      {(wine.label_url || wine.label || wine.label_ocr) && (
        <section className="wine-detail__label">
          <h2>Этикетка</h2>
          <div className="wine-detail__label-grid">
            {wine.label_url && !labelBroken ? (
              <img
                src={wine.label_url}
                alt={`Этикетка: ${wine.name}`}
                onError={() => setLabelBroken(true)}
              />
            ) : null}
            {wine.label_ocr != null ? (
              <pre className="wine-detail__label-ocr">
                {JSON.stringify(wine.label_ocr, null, 2)}
              </pre>
            ) : wine.label ? (
              <pre>{wine.label}</pre>
            ) : null}
          </div>
        </section>
      )}

      {wine.description && (
        <section className="wine-detail__desc">
          <h2>Описание</h2>
          <p>{wine.description}</p>
        </section>
      )}
    </main>
  )
}
