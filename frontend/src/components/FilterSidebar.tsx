import { useEffect, useState } from 'react'
import type { FilterFacet } from '../api/client'
import './FilterSidebar.css'

export type FilterKey = 'category' | 'color' | 'region' | 'grape_variety' | 'winery' | 'name'

const LABELS: Record<FilterKey, string> = {
  category: 'Категория',
  color: 'Цвет',
  region: 'Регион',
  grape_variety: 'Сорт винограда',
  winery: 'Винодельня',
  name: 'Название',
}

type Props = {
  facets: Record<FilterKey, FilterFacet[]>
  selected: Record<FilterKey, string[]>
  onChange: (key: FilterKey, values: string[]) => void
  nameQuery: string
  onNameQueryChange: (value: string) => void
  mobileOpen?: boolean
  onMobileToggle?: () => void
}

const PREVIEW = 8
const MOBILE_MQ = '(max-width: 900px)'

export function FilterSidebar({
  facets,
  selected,
  onChange,
  nameQuery,
  onNameQueryChange,
  mobileOpen = false,
  onMobileToggle,
}: Props) {
  const [isMobile, setIsMobile] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(MOBILE_MQ).matches,
  )
  const [open, setOpen] = useState<Record<FilterKey, boolean>>({
    category: true,
    color: true,
    region: true,
    grape_variety: true,
    winery: true,
    name: false,
  })
  const [query, setQuery] = useState<Record<FilterKey, string>>({
    category: '',
    color: '',
    region: '',
    grape_variety: '',
    winery: '',
    name: '',
  })
  const [expanded, setExpanded] = useState<Partial<Record<FilterKey, boolean>>>({})

  useEffect(() => {
    const mq = window.matchMedia(MOBILE_MQ)
    const onChangeMq = () => setIsMobile(mq.matches)
    onChangeMq()
    mq.addEventListener('change', onChangeMq)
    return () => mq.removeEventListener('change', onChangeMq)
  }, [])

  const toggle = (key: FilterKey, value: string) => {
    const current = selected[key]
    const next = current.includes(value)
      ? current.filter((v) => v !== value)
      : [...current, value]
    onChange(key, next)
  }

  const collapsed = isMobile && !mobileOpen

  return (
    <aside className={`filters ${collapsed ? 'filters--collapsed' : ''}`}>
      {isMobile && (
        <button
          type="button"
          className="filters__mobile-toggle"
          onClick={onMobileToggle}
          aria-expanded={mobileOpen}
        >
          {mobileOpen ? 'Фильтры <<' : 'Фильтры >>'}
        </button>
      )}

      {(!isMobile || mobileOpen) && (
        <>
          <div className="filters__top-search">
            <label htmlFor="wine-name-search">поиск</label>
            <input
              id="wine-name-search"
              className="filters__search filters__search--main"
              type="search"
              placeholder="вино, винодельня, сорт…"
              value={nameQuery}
              onChange={(e) => onNameQueryChange(e.target.value)}
            />
          </div>
          {(Object.keys(LABELS) as FilterKey[]).map((key) => {
            if (key === 'name') return null
            const q = query[key].trim().toLowerCase()
            const all = facets[key] || []
            const filtered = q
              ? all.filter((f) => f.value.toLowerCase().includes(q))
              : all
            const visible = expanded[key] ? filtered : filtered.slice(0, PREVIEW)

            return (
              <section key={key} className="filters__group">
                <button
                  type="button"
                  className="filters__title"
                  onClick={() => setOpen((s) => ({ ...s, [key]: !s[key] }))}
                >
                  <span>{LABELS[key]}</span>
                  <span aria-hidden>{open[key] ? '▴' : '▾'}</span>
                </button>
                {open[key] && (
                  <div className="filters__body">
                    <input
                      className="filters__search"
                      placeholder="Поиск"
                      value={query[key]}
                      onChange={(e) => setQuery((s) => ({ ...s, [key]: e.target.value }))}
                    />
                    <ul>
                      {visible.map((facet) => (
                        <li key={facet.value}>
                          <label>
                            <input
                              type="checkbox"
                              checked={selected[key].includes(facet.value)}
                              onChange={() => toggle(key, facet.value)}
                            />
                            <span>{facet.value}</span>
                            <em>{facet.count}</em>
                          </label>
                        </li>
                      ))}
                    </ul>
                    {filtered.length > PREVIEW && (
                      <button
                        type="button"
                        className="filters__more"
                        onClick={() =>
                          setExpanded((s) => ({ ...s, [key]: !s[key] }))
                        }
                      >
                        {expanded[key] ? 'Свернуть' : `Все ${LABELS[key].toLowerCase()}`}
                      </button>
                    )}
                  </div>
                )}
              </section>
            )
          })}
        </>
      )}
    </aside>
  )
}
