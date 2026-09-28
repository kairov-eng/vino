import { useCallback, useEffect, useState, type MouseEvent } from 'react'
import { Link, NavLink, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { fetchFindWineResult, fetchPipelineSettings, fetchScanHistory } from '../api/client'
import { useSiteAuth } from '../auth/SiteAuthContext'
import './Header.css'

const links = [
  { to: '/', label: 'Сканер вина', end: true },
  { to: '/wines', label: 'Каталог вин' },
  { to: '/history', label: 'История сканирования' },
]

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden>
      <path
        fill="currentColor"
        d="M10.5 3a7.5 7.5 0 0 1 5.9 12.1l4.25 4.25a1 1 0 0 1-1.4 1.4l-4.26-4.24A7.5 7.5 0 1 1 10.5 3Zm0 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Z"
      />
    </svg>
  )
}

export function Header() {
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { isAdmin } = useSiteAuth()
  const isScanner = location.pathname === '/'
  const rawScanId = isScanner ? searchParams.get('scanid') : null
  const scanId = rawScanId != null ? Number(rawScanId) : NaN
  const hasCurrent = Number.isFinite(scanId) && scanId > 0
  const [latestId, setLatestId] = useState<number | null>(null)
  const [nextBusy, setNextBusy] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [showSearchDetailsSetting, setShowSearchDetailsSetting] = useState(true)
  const showDetails = isAdmin && showSearchDetailsSetting

  useEffect(() => {
    setMenuOpen(false)
  }, [location.pathname, location.search])

  useEffect(() => {
    if (!isAdmin) {
      setShowSearchDetailsSetting(false)
      return
    }
    let cancelled = false
    fetchPipelineSettings()
      .then((s) => {
        if (!cancelled) {
          setShowSearchDetailsSetting(s.show_search_details !== false)
        }
      })
      .catch(() => {
        /* keep default true for admin */
      })
    const onSettings = (e: Event) => {
      const detail = (e as CustomEvent<{ show_search_details?: boolean }>).detail
      if (!detail || typeof detail.show_search_details === 'undefined') return
      setShowSearchDetailsSetting(detail.show_search_details !== false)
    }
    window.addEventListener('vino:pipeline-settings', onSettings)
    return () => {
      cancelled = true
      window.removeEventListener('vino:pipeline-settings', onSettings)
    }
  }, [isAdmin])

  useEffect(() => {
    if (!menuOpen) return
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => {
      document.body.style.overflow = prev
      window.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  useEffect(() => {
    if (!isScanner || !showDetails) {
      setLatestId(null)
      return
    }
    let cancelled = false
    fetchScanHistory({ limit: 1, sort: 'id', order: 'desc' })
      .then((data) => {
        if (cancelled) return
        const id = data.items[0]?.id
        setLatestId(typeof id === 'number' && id > 0 ? id : null)
      })
      .catch(() => {
        if (!cancelled) setLatestId(null)
      })
    return () => {
      cancelled = true
    }
  }, [isScanner, showDetails, hasCurrent, location.key])

  const prevId = hasCurrent ? scanId - 1 : latestId
  const nextId = hasCurrent ? scanId + 1 : null
  const canPrev = prevId != null && prevId >= 1
  const atLatest = latestId != null && hasCurrent && scanId >= latestId
  const canNext = Boolean(hasCurrent && nextId != null && !atLatest && !nextBusy)

  const goNext = useCallback(
    async (e: MouseEvent) => {
      e.preventDefault()
      if (!hasCurrent || nextId == null || nextBusy) return
      setNextBusy(true)
      try {
        await fetchFindWineResult(nextId)
        navigate(`/?scanid=${nextId}`)
      } catch {
        setLatestId((prev) => (prev == null || scanId >= prev ? scanId : prev))
      } finally {
        setNextBusy(false)
      }
    },
    [hasCurrent, nextId, nextBusy, navigate, scanId],
  )

  return (
    <header className="site-header">
      <div className="site-header__inner">
        <NavLink to="/" className="site-logo" end>
          <img src="/svoe-vino-logo.svg" alt="Своё Вино" width={40} height={40} />
          <span>
            <strong>Своё Вино</strong>
            <small className="site-logo__tagline-desktop">каталог и сканер</small>
            <small className="site-logo__tagline-mobile">От Россельхозбанка</small>
          </span>
        </NavLink>

        {isScanner && showDetails && (
          <nav className="site-scan-nav" aria-label="Навигация по сравнениям">
            {canPrev ? (
              <Link
                to={`/?scanid=${prevId}`}
                className="site-scan-nav__link"
                title={
                  hasCurrent
                    ? `Сравнение #${prevId}`
                    : `Последнее сравнение #${prevId}`
                }
              >
                {'<< предыдущий поиск'}
              </Link>
            ) : (
              <span className="site-scan-nav__link is-disabled" aria-disabled="true">
                {'<< предыдущий поиск'}
              </span>
            )}
            {canNext ? (
              <a
                href={`/?scanid=${nextId}`}
                className="site-scan-nav__link"
                title={`Сравнение #${nextId}`}
                onClick={goNext}
              >
                {'следующий поиск >>'}
              </a>
            ) : (
              <span
                className="site-scan-nav__link is-disabled"
                aria-disabled="true"
                title={atLatest ? 'Это последнее сравнение' : undefined}
              >
                {'следующий поиск >>'}
              </span>
            )}
          </nav>
        )}

        <nav className="site-nav site-nav--desktop">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) => (isActive ? 'is-active' : undefined)}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="site-header__mobile-actions">
          <Link to="/wines" className="site-header__search" aria-label="Каталог вин">
            <SearchIcon />
          </Link>
          <button
            type="button"
            className={`site-burger${menuOpen ? ' is-open' : ''}`}
            aria-label={menuOpen ? 'Закрыть меню' : 'Открыть меню'}
            aria-expanded={menuOpen}
            aria-controls="site-mobile-menu"
            onClick={() => setMenuOpen((v) => !v)}
          >
            <span />
            <span />
            <span />
          </button>
        </div>
      </div>

      {menuOpen && (
        <div
          className="site-menu-overlay"
          onClick={() => setMenuOpen(false)}
          aria-hidden
        />
      )}

      <nav
        id="site-mobile-menu"
        className={`site-nav-drawer${menuOpen ? ' is-open' : ''}`}
        aria-hidden={!menuOpen}
      >
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            className={({ isActive }) => (isActive ? 'is-active' : undefined)}
            onClick={() => setMenuOpen(false)}
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}
