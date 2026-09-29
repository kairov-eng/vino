import { useState } from 'react'

/**
 * Thumbs are for production (nginx /media/t/...).
 * Local `npm run dev` skips thumbs by default — no 404 storm.
 *
 * Override:
 *   VITE_USE_MEDIA_THUMBS=1  force thumbs in dev
 *   VITE_USE_MEDIA_THUMBS=0  disable thumbs even in prod build
 */
export function useMediaThumbsEnabled(): boolean {
  const flag = String(import.meta.env.VITE_USE_MEDIA_THUMBS ?? '').trim().toLowerCase()
  if (flag === '0' || flag === 'false' || flag === 'off') return false
  if (flag === '1' || flag === 'true' || flag === 'on') return true
  return Boolean(import.meta.env.PROD)
}

/**
 * Prefer server catalog thumbs at /media/t/... (webp with alpha).
 * Full-size stays at /media/... for detail / fallback.
 * `v=2` busts browser cache after alpha-preserving thumb rebuild.
 */
export function catalogThumbUrl(url: string | null | undefined): string | null {
  if (!url) return null
  if (!useMediaThumbsEnabled()) return url
  if (!url.startsWith('/media/')) return url
  if (url.startsWith('/media/t/')) return url
  const rest = url.slice('/media/'.length)
  const q = rest.indexOf('?')
  const path = q >= 0 ? rest.slice(0, q) : rest
  const stemDot = path.lastIndexOf('.')
  const stem = stemDot > 0 ? path.slice(0, stemDot) : path
  return `/media/t/${stem}.webp?v=2`
}

type MediaThumbImgProps = {
  fullUrl: string
  alt: string
  className?: string
  loading?: 'lazy' | 'eager'
  onBroken?: () => void
}

/** Grid / card image: thumb first (prod), then full /media on error. */
export function MediaThumbImg({
  fullUrl,
  alt,
  className,
  loading = 'lazy',
  onBroken,
}: MediaThumbImgProps) {
  const thumb = catalogThumbUrl(fullUrl)
  const [src, setSrc] = useState(thumb || fullUrl)
  return (
    <img
      className={className}
      src={src}
      alt={alt}
      loading={loading}
      decoding="async"
      onError={() => {
        if (src !== fullUrl) {
          setSrc(fullUrl)
          return
        }
        onBroken?.()
      }}
    />
  )
}
