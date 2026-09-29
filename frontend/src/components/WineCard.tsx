import { Fragment, type ReactNode, useState } from 'react'
import { Link } from 'react-router-dom'
import type { Wine } from '../api/client'
import './WineCard.css'

type Props = { wine: Wine; highlightQuery?: string }

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Catalog grid: prefer server thumbs at /media/t/... (webp).
 * Full-size stays at /media/... for detail pages / fallback.
 * `v=2` busts browser cache after alpha-preserving thumb rebuild.
 */
export function catalogThumbUrl(url: string | null | undefined): string | null {
  if (!url) return null
  if (!url.startsWith('/media/')) return url
  if (url.startsWith('/media/t/')) return url
  const rest = url.slice('/media/'.length)
  const q = rest.indexOf('?')
  const path = q >= 0 ? rest.slice(0, q) : rest
  const stemDot = path.lastIndexOf('.')
  const stem = stemDot > 0 ? path.slice(0, stemDot) : path
  return `/media/t/${stem}.webp?v=2`
}

/** Bold matches only at word starts (same rule as backend). */
export function highlightWordStarts(text: string, query: string): ReactNode {
  const q = query.trim()
  if (!q || !text) return text

  const re = new RegExp(`(^|[^0-9A-Za-zА-Яа-яЁё])(${escapeRegExp(q)})`, 'gi')
  const nodes: ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  let key = 0

  while ((match = re.exec(text)) !== null) {
    const boundary = match[1] ?? ''
    const hit = match[2] ?? ''
    const start = match.index
    if (start > last) {
      nodes.push(text.slice(last, start))
    }
    if (boundary) {
      nodes.push(boundary)
    }
    nodes.push(<strong key={key++}>{hit}</strong>)
    last = start + boundary.length + hit.length
  }

  if (last === 0) return text
  if (last < text.length) nodes.push(text.slice(last))
  return <Fragment>{nodes}</Fragment>
}

function fieldMatches(text: string | null | undefined, query: string): boolean {
  const q = query.trim()
  if (!q || !text) return false
  const re = new RegExp(`(^|[^0-9A-Za-zА-Яа-яЁё])${escapeRegExp(q)}`, 'i')
  return re.test(text)
}

function CatalogImg({
  fullUrl,
  alt,
  onBroken,
}: {
  fullUrl: string
  alt: string
  onBroken: () => void
}) {
  const thumb = catalogThumbUrl(fullUrl)
  const [src, setSrc] = useState(thumb || fullUrl)
  return (
    <img
      src={src}
      alt={alt}
      loading="lazy"
      decoding="async"
      onError={() => {
        if (src !== fullUrl) {
          setSrc(fullUrl)
          return
        }
        onBroken()
      }}
    />
  )
}

export function WineCard({ wine, highlightQuery = '' }: Props) {
  const slug = wine.slug || String(wine.id)
  const [hovered, setHovered] = useState(false)
  const [bottleBroken, setBottleBroken] = useState(false)
  const [labelBroken, setLabelBroken] = useState(false)
  const hasPopup = Boolean(wine.label)
  const showBottle = Boolean(wine.photo_url && !bottleBroken)
  const showLabel = Boolean(wine.label_url && !labelBroken)
  const q = highlightQuery.trim()

  const extraFields = (
    [
      ['Категория', wine.category],
      ['Цвет', wine.color],
      ['Сорт', wine.grape_variety],
    ] as const
  ).filter(([, value]) => q && fieldMatches(value, q))

  return (
    <Link
      to={`/wines/${slug}`}
      className="wine-card"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="wine-card__media">
        <div className="wine-card__half wine-card__half--bottle">
          {showBottle ? (
            <CatalogImg
              key={`b-${wine.photo_url}`}
              fullUrl={wine.photo_url!}
              alt={wine.name}
              onBroken={() => setBottleBroken(true)}
            />
          ) : (
            <div className="wine-card__placeholder wine-card__placeholder--bottle" />
          )}
        </div>
        <div className="wine-card__half wine-card__half--label">
          {showLabel ? (
            <CatalogImg
              key={`l-${wine.label_url}`}
              fullUrl={wine.label_url!}
              alt={`Этикетка: ${wine.name}`}
              onBroken={() => setLabelBroken(true)}
            />
          ) : (
            <div className="wine-card__placeholder wine-card__placeholder--label" />
          )}
        </div>
      </div>
      <div className="wine-card__meta">
        <h2>{highlightWordStarts(wine.name, q)}</h2>
        <p>{highlightWordStarts(wine.winery || '—', q)}</p>
        {extraFields.map(([label, value]) => (
          <p key={label} className="wine-card__extra">
            {label}: {highlightWordStarts(value || '', q)}
          </p>
        ))}
      </div>

      {hovered && hasPopup && (
        <div className="wine-card__popup" role="tooltip">
          <pre className="wine-card__label-text">{wine.label}</pre>
        </div>
      )}
    </Link>
  )
}
