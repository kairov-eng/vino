import type { WineFilters } from './api/client'

export const EMPTY_WINE_FILTERS: WineFilters = {
  category: [],
  color: [],
  region: [],
  grape_variety: [],
  winery: [],
  name: [],
}

const FILTER_KEYS: (keyof WineFilters)[] = [
  'category',
  'color',
  'region',
  'grape_variety',
  'winery',
  'name',
]

function one(v: string | null | undefined): string[] {
  const s = String(v || '').trim()
  return s ? [s] : []
}

/** Build `/wines?...` path for opening catalog with selected facets. */
export function winesCatalogPath(
  partial: Partial<Record<keyof WineFilters, string | string[] | null | undefined>>,
  q?: string | null,
): string {
  const qs = new URLSearchParams()
  for (const key of FILTER_KEYS) {
    const raw = partial[key]
    if (raw == null || raw === '') continue
    const vals = Array.isArray(raw) ? raw : [raw]
    for (const v of vals) {
      const s = String(v || '').trim()
      if (s) qs.append(key, s)
    }
  }
  const query = String(q || '').trim()
  if (query) qs.set('q', query)
  const s = qs.toString()
  return s ? `/wines?${s}` : '/wines'
}

export function wineFiltersFromSearchParams(sp: URLSearchParams): WineFilters {
  const next = { ...EMPTY_WINE_FILTERS }
  for (const key of FILTER_KEYS) {
    next[key] = sp.getAll(key).map((v) => v.trim()).filter(Boolean)
  }
  return next
}

export function wineFiltersToSearchParams(
  filters: WineFilters,
  q?: string | null,
): URLSearchParams {
  const qs = new URLSearchParams()
  for (const key of FILTER_KEYS) {
    for (const v of filters[key] || []) {
      const s = String(v || '').trim()
      if (s) qs.append(key, s)
    }
  }
  const query = String(q || '').trim()
  if (query) qs.set('q', query)
  return qs
}

export function searchParamsEqual(a: URLSearchParams, b: URLSearchParams): boolean {
  const ka = [...a.keys()].sort()
  const kb = [...b.keys()].sort()
  if (ka.join('\0') !== kb.join('\0')) return false
  for (const k of ka) {
    const va = a.getAll(k).slice().sort().join('\0')
    const vb = b.getAll(k).slice().sort().join('\0')
    if (va !== vb) return false
  }
  return true
}

/** Convenience builders matching wine-detail click semantics. */
export function catalogHrefWinery(winery: string | null | undefined) {
  return winesCatalogPath({ winery: one(winery) })
}

export function catalogHrefRegion(region: string | null | undefined) {
  return winesCatalogPath({ region: one(region) })
}

export function catalogHrefRegionWithWinery(
  region: string | null | undefined,
  winery: string | null | undefined,
) {
  return winesCatalogPath({ region: one(region), winery: one(winery) })
}

export function catalogHrefGrape(grape: string | null | undefined) {
  return winesCatalogPath({ grape_variety: one(grape) })
}

export function catalogHrefGrapeWithParents(
  grape: string | null | undefined,
  region: string | null | undefined,
  winery: string | null | undefined,
) {
  return winesCatalogPath({
    grape_variety: one(grape),
    region: one(region),
    winery: one(winery),
  })
}

export function catalogHrefCategory(category: string | null | undefined) {
  return winesCatalogPath({ category: one(category) })
}

export function catalogHrefCategoryWithParents(
  category: string | null | undefined,
  region: string | null | undefined,
  winery: string | null | undefined,
) {
  return winesCatalogPath({
    category: one(category),
    region: one(region),
    winery: one(winery),
  })
}

export function catalogHrefColor(color: string | null | undefined) {
  return winesCatalogPath({ color: one(color) })
}

export function catalogHrefColorWithParents(
  color: string | null | undefined,
  region: string | null | undefined,
  winery: string | null | undefined,
) {
  return winesCatalogPath({
    color: one(color),
    region: one(region),
    winery: one(winery),
  })
}
