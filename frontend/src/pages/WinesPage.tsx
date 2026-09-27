import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  fetchFilters,
  fetchWines,
  type FiltersResponse,
  type Wine,
} from '../api/client'
import {
  wineFiltersFromSearchParams,
  wineFiltersToSearchParams,
} from '../catalogFilters'
import { FilterSidebar, type FilterKey } from '../components/FilterSidebar'
import { WineCard } from '../components/WineCard'
import './WinesPage.css'

const EMPTY_FACETS: FiltersResponse = {
  category: [],
  color: [],
  region: [],
  grape_variety: [],
  winery: [],
  name: [],
}

export function WinesPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const filters = useMemo(
    () => wineFiltersFromSearchParams(searchParams),
    [searchParams],
  )
  const urlQuery = searchParams.get('q') || ''
  const [nameQuery, setNameQuery] = useState(urlQuery)
  const [debouncedQuery, setDebouncedQuery] = useState(urlQuery.trim())
  const [facets, setFacets] = useState<FiltersResponse>(EMPTY_FACETS)
  const [items, setItems] = useState<Wine[]>([])
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const sentinelRef = useRef<HTMLDivElement>(null)
  const requestId = useRef(0)

  // External URL change (new tab / back) → sync name input
  useEffect(() => {
    setNameQuery(urlQuery)
    setDebouncedQuery(urlQuery.trim())
  }, [urlQuery])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(nameQuery.trim())
    }, 300)
    return () => window.clearTimeout(timer)
  }, [nameQuery])

  // Persist debounced name query into URL (keep facet filters)
  useEffect(() => {
    if ((searchParams.get('q') || '') === debouncedQuery) return
    setSearchParams(wineFiltersToSearchParams(filters, debouncedQuery), {
      replace: true,
    })
  }, [debouncedQuery, filters, searchParams, setSearchParams])

  useEffect(() => {
    fetchFilters()
      .then(setFacets)
      .catch((err) => setError(err instanceof Error ? err.message : 'Ошибка фильтров'))
  }, [])

  const loadPage = useCallback(
    async (offset: number, replace: boolean) => {
      const id = ++requestId.current
      setLoading(true)
      setError(null)
      try {
        const data = await fetchWines(filters, offset, 24, debouncedQuery || undefined)
        if (id !== requestId.current) return
        setItems((prev) => (replace ? data.items : [...prev, ...data.items]))
        setTotal(data.total)
        setHasMore(data.has_more)
      } catch (err) {
        if (id !== requestId.current) return
        setError(err instanceof Error ? err.message : 'Ошибка загрузки')
      } finally {
        if (id === requestId.current) setLoading(false)
      }
    },
    [filters, debouncedQuery],
  )

  useEffect(() => {
    setItems([])
    loadPage(0, true)
  }, [loadPage])

  useEffect(() => {
    const node = sentinelRef.current
    if (!node) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && hasMore && !loading) {
          loadPage(items.length, false)
        }
      },
      { rootMargin: '400px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [hasMore, loading, items.length, loadPage])

  const onFilterChange = (key: FilterKey, values: string[]) => {
    const next = { ...filters, [key]: values }
    setSearchParams(wineFiltersToSearchParams(next, debouncedQuery), {
      replace: true,
    })
  }

  return (
    <main className="page wines-page">
      <div className="page__intro">
        <h1>Свои вина</h1>
        <p className="page__lead">Самый полный независимый каталог российских вин</p>
        <p className="wines-page__count">Найдено: {total}</p>
      </div>

      <div className="wines-layout">
        <FilterSidebar
          facets={facets}
          selected={filters}
          onChange={onFilterChange}
          nameQuery={nameQuery}
          onNameQueryChange={setNameQuery}
          mobileOpen={filtersOpen}
          onMobileToggle={() => setFiltersOpen((v) => !v)}
        />
        <section>
          {error && <p className="page-error">{error}</p>}
          <div className="wines-grid">
            {items.map((wine) => (
              <WineCard key={wine.id} wine={wine} highlightQuery={debouncedQuery} />
            ))}
          </div>
          {!loading && items.length === 0 && !error && (
            <p className="page-empty">По выбранным фильтрам ничего не найдено</p>
          )}
          <div ref={sentinelRef} className="wines-sentinel" />
          {loading && <p className="page-loading">Загрузка…</p>}
        </section>
      </div>
    </main>
  )
}
