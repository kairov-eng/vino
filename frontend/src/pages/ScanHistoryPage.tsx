import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { Link, useNavigate } from 'react-router-dom'
import {
  fetchScanHistory,
  fetchScanHistoryReport,
  updateScanHistoryEval,
  updateFindwineManualWine,
  type ScanHistoryItem,
  type ScanHistorySort,
  type ScanHistoryXgbTopItem,
} from '../api/client'
import {
  listSavedReports,
  saveReport,
} from './scanEvalReportStorage'
import './ScanHistoryPage.css'

type SortState = { sort: ScanHistorySort; order: 'asc' | 'desc' }

const MATCH_STOP = new Set([
  'и',
  'в',
  'на',
  'по',
  'из',
  'для',
  'the',
  'and',
  'of',
  'с',
  'к',
  'от',
  'до',
  'или',
  'wine',
  'вино',
])

function normalizeCompareText(raw: string): string {
  let t = String(raw || '')
    .replace(/\u00a0/g, ' ')
    .replace(/\t/g, ' ')
    .replace(/\r/g, '\n')
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/\n/g, ' ')
  while (t.includes('  ')) {
    t = t.replace(/  /g, ' ')
  }
  return t.trim()
}

function normalizeMatchToken(raw: string): string {
  return normalizeCompareText(raw).replace(/[^a-zа-я0-9]+/gi, '')
}

function sharedMatchTokens(a: string, b: string): Set<string> {
  const tok = (s: string) => {
    const out = new Set<string>()
    for (const m of normalizeCompareText(s).match(/[a-zа-я0-9]{3,}/g) || []) {
      const n = normalizeMatchToken(m)
      if (n.length >= 3 && !MATCH_STOP.has(n)) out.add(n)
    }
    return out
  }
  const A = tok(a)
  const B = tok(b)
  const inter = new Set<string>()
  for (const t of A) if (B.has(t)) inter.add(t)
  return inter
}

function highlightMatchedText(text: string, matched: Set<string>): ReactNode {
  if (!text) return '—'
  if (!matched.size) return text
  const parts = text.split(/([A-Za-zА-Яа-яЁё0-9]+)/)
  return parts.map((part, i) => {
    const n = normalizeMatchToken(part)
    if (n && matched.has(n)) {
      return (
        <mark key={i} className="history-ocr__match">
          {part}
        </mark>
      )
    }
    return <span key={i}>{part}</span>
  })
}

function formatScore01(v: number | string | null | undefined): string {
  if (v == null || v === '') return '—'
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return '—'
  return n.toFixed(2)
}

function placeNearAnchor(el: HTMLElement): { top: number; left: number } {
  const rect = el.getBoundingClientRect()
  const gap = 12
  const popupW = Math.min(window.innerWidth * 0.72, 920)
  const popupH = Math.min(window.innerHeight * 0.78, 640)

  let left = rect.right + gap
  if (left + popupW > window.innerWidth - 8) {
    left = rect.left - popupW - gap
  }
  if (left < 8) left = 8

  let top = rect.top
  if (top + popupH > window.innerHeight - 8) {
    top = window.innerHeight - popupH - 8
  }
  if (top < 8) top = 8
  return { top, left }
}

function HistoryWineHoverPopup({
  wine,
  pos,
  ocrQueryText,
  onMouseEnter,
  onMouseLeave,
}: {
  wine: ScanHistoryXgbTopItem
  pos: { top: number; left: number }
  ocrQueryText?: string
  onMouseEnter?: () => void
  onMouseLeave?: () => void
}) {
  const [labelBroken, setLabelBroken] = useState(false)
  const [bottleBroken, setBottleBroken] = useState(false)
  const catalogLabel = String(wine.label || '').trim()
  const queryOcr = String(ocrQueryText || '').trim()
  const matchedTokens = useMemo(
    () => sharedMatchTokens(queryOcr, catalogLabel),
    [queryOcr, catalogLabel],
  )

  return createPortal(
    <div
      className="history-cand-popup"
      role="dialog"
      aria-label={`Вино ${wine.id}`}
      style={{ top: pos.top, left: pos.left }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="history-cand-popup__inner">
        <div className="history-cand-popup__left">
          <div className="history-cand-popup__photos">
            <div className="history-cand-popup__photo">
              <span>Бутылка</span>
              {wine.photo_url && !bottleBroken ? (
                <img
                  src={wine.photo_url}
                  alt="Бутылка"
                  onError={() => setBottleBroken(true)}
                />
              ) : (
                <div className="history-cand-popup__placeholder" />
              )}
            </div>
            <div className="history-cand-popup__photo">
              <span>Этикетка</span>
              {wine.label_url && !labelBroken ? (
                <img
                  src={wine.label_url}
                  alt="Этикетка"
                  onError={() => setLabelBroken(true)}
                />
              ) : (
                <div className="history-cand-popup__placeholder" />
              )}
            </div>
          </div>
          <div className="history-cand-popup__texts">
            <div className="history-cand-popup__label-cols">
              <div className="history-cand-popup__label-text">
                <strong>OCR искомого</strong>
                <pre>{highlightMatchedText(queryOcr, matchedTokens)}</pre>
              </div>
              <div className="history-cand-popup__label-text">
                <strong>Текст этикетки каталога</strong>
                <pre>{highlightMatchedText(catalogLabel, matchedTokens)}</pre>
              </div>
            </div>
          </div>
        </div>
        <div className="history-cand-popup__info">
          <p>
            <strong>ID:</strong> {wine.id}
          </p>
          <p>
            <strong>Название:</strong> {wine.name || '—'}
          </p>
          <p>
            <strong>Винодельня:</strong> {wine.winery || '—'}
          </p>
          <div className="history-cand-popup__scores">
            <strong>Скоры</strong>
            <ul>
              <li>
                <span>cos</span>
                <em>{formatScore01(wine.cosine)}</em>
              </li>
              {wine.xgb_score != null && (
                <li>
                  <span>XGB</span>
                  <em>{formatScore01(wine.xgb_score)}</em>
                </li>
              )}
              {wine.xgb_fin != null && (
                <li>
                  <span>Final score</span>
                  <em>{formatScore01(wine.xgb_fin)}</em>
                </li>
              )}
              {wine.fin2 != null && (
                <li>
                  <span>fin2</span>
                  <em>{formatScore01(wine.fin2)}</em>
                </li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

function HistoryXgbTopCard({
  wine,
  ocrQueryText,
  onOpenWine,
  isManualMatch = false,
  onManualMatchChange,
}: {
  wine: ScanHistoryXgbTopItem
  ocrQueryText?: string
  onOpenWine: (slug: string | null, id: number) => void
  isManualMatch?: boolean
  onManualMatchChange?: (wineId: number, checked: boolean) => void
}) {
  const mediaRef = useRef<HTMLButtonElement>(null)
  const [hovered, setHovered] = useState(false)
  const [popupPos, setPopupPos] = useState<{ top: number; left: number } | null>(
    null,
  )
  const [imgBroken, setImgBroken] = useState(false)
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const thumb = wine.label_url || wine.photo_url

  const clearLeaveTimer = useCallback(() => {
    if (leaveTimer.current) {
      clearTimeout(leaveTimer.current)
      leaveTimer.current = null
    }
  }, [])

  const openHover = useCallback(() => {
    clearLeaveTimer()
    setHovered(true)
  }, [clearLeaveTimer])

  const closeHoverSoon = useCallback(() => {
    clearLeaveTimer()
    leaveTimer.current = setTimeout(() => setHovered(false), 200)
  }, [clearLeaveTimer])

  useEffect(() => () => clearLeaveTimer(), [clearLeaveTimer])

  const placePopup = useCallback(() => {
    const el = mediaRef.current
    if (!el) return
    setPopupPos(placeNearAnchor(el))
  }, [])

  useLayoutEffect(() => {
    if (!hovered) {
      setPopupPos(null)
      return
    }
    placePopup()
    const onMove = () => placePopup()
    window.addEventListener('scroll', onMove, true)
    window.addEventListener('resize', onMove)
    return () => {
      window.removeEventListener('scroll', onMove, true)
      window.removeEventListener('resize', onMove)
    }
  }, [hovered, placePopup])

  const scoreLabel =
    wine.xgb_fin != null
      ? 'Final score'
      : wine.fin2 != null
        ? 'Final score'
        : wine.cosine != null
          ? 'cos'
          : null
  const scoreValue =
    wine.xgb_fin != null
      ? wine.xgb_fin
      : wine.fin2 != null
        ? wine.fin2
        : wine.cosine

  return (
    <>
      <div
        className={`history-xgb-card${isManualMatch ? ' is-manual' : ''}`}
      >
        <button
          ref={mediaRef}
          type="button"
          className="history-xgb-card__media"
          title={wine.name || undefined}
          onMouseEnter={openHover}
          onMouseLeave={closeHoverSoon}
          onClick={(e) => {
            e.stopPropagation()
            onOpenWine(wine.slug, wine.id)
          }}
        >
          {thumb && !imgBroken ? (
            <img
              src={thumb}
              alt={wine.name || ''}
              loading="lazy"
              onError={() => setImgBroken(true)}
            />
          ) : (
            <div className="history-xgb-card__placeholder" />
          )}
        </button>
        <div className="history-xgb-card__meta">
          {scoreLabel != null && scoreValue != null && (
            <p className="history-xgb-card__score">
              {scoreLabel} <em>{formatScore01(scoreValue)}</em>
            </p>
          )}
          <p className="history-xgb-card__id">id {wine.id}</p>
          {onManualMatchChange && (
            <label
              className="history-xgb-card__manual"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={isManualMatch}
                onChange={(e) => {
                  e.stopPropagation()
                  onManualMatchChange(wine.id, e.target.checked)
                }}
              />
              Это вино
            </label>
          )}
        </div>
      </div>
      {hovered && popupPos && (
        <HistoryWineHoverPopup
          wine={wine}
          pos={popupPos}
          ocrQueryText={ocrQueryText}
          onMouseEnter={openHover}
          onMouseLeave={closeHoverSoon}
        />
      )}
    </>
  )
}

function HistoryScores({
  scores,
  scanId,
  totalMs,
  wineId,
  wineSlug,
  falsePositive,
  falseNegative,
  onEvalChange,
}: {
  scores: ScanHistoryItem['scores']
  scanId: number
  totalMs: number | null | undefined
  wineId: number | null | undefined
  wineSlug: string | null | undefined
  falsePositive: number
  falseNegative: number
  onEvalChange: (next: { false_positive: number; false_negative: number }) => void
}) {
  const meta = `#${scanId}${
    totalMs != null && Number.isFinite(totalMs) ? ` · ${Math.round(totalMs)} мс` : ''
  }`
  const found = wineId != null
  const wineLine = found
    ? `id ${wineId}${wineSlug ? ` · ${wineSlug}` : ''}`
    : null
  const fp = falsePositive ? 1 : 0
  const fn = falseNegative ? 1 : 0

  const lines: string[] = []
  if (scores) {
    if (scores.cos != null) {
      lines.push(`cos ${formatScore01(scores.cos)}`)
    }
    {
      const parts: string[] = []
      if (scores.txt1 != null) parts.push(`txt1 ${formatScore01(scores.txt1)}`)
      if (scores.fin1 != null) parts.push(`fin1 ${formatScore01(scores.fin1)}`)
      if (parts.length) lines.push(parts.join(' '))
    }
    {
      const parts: string[] = []
      if (scores.txt2 != null) parts.push(`txt2 ${formatScore01(scores.txt2)}`)
      if (scores.fin2 != null) parts.push(`fin2 ${formatScore01(scores.fin2)}`)
      if (parts.length) lines.push(parts.join(' '))
    }
    {
      const parts: string[] = []
      if (scores.xgb != null) parts.push(`XGB ${formatScore01(scores.xgb)}`)
      if (scores.xgb_fin != null)
        parts.push(`Final score ${formatScore01(scores.xgb_fin)}`)
      if (parts.length) lines.push(parts.join(' '))
    }
    {
      const parts: string[] = []
      if (scores.crenc != null) parts.push(`CrEnc ${formatScore01(scores.crenc)}`)
      if (scores.crenc_fin != null) {
        parts.push(`Final score ${formatScore01(scores.crenc_fin)}`)
      }
      if (parts.length) lines.push(parts.join(' '))
    }
    if (typeof scores.channels_line === 'string' && scores.channels_line) {
      // Уже собрано на бэке без нулевых Score V/Y/…
      const cleaned = scores.channels_line
        .split(/\s*·\s*/)
        .filter((bit) => {
          const m = bit.match(/^Score\s+[A-Z]\s+([\d.]+)$/i)
          if (!m) return true
          const n = Number(m[1])
          return Number.isFinite(n) && Math.abs(n) > 1e-9
        })
        .join(' · ')
      if (cleaned) lines.push(cleaned)
    } else {
      const ch: string[] = []
      for (const letter of ['V', 'Y', 'G', 'O', 'K', 'Q']) {
        const s = scores[`score_${letter}`]
        const n = typeof s === 'number' ? s : Number(s)
        if (s != null && Number.isFinite(n) && Math.abs(n) > 1e-9) {
          ch.push(`Score ${letter} ${formatScore01(n)}`)
        }
      }
      if (ch.length) lines.push(ch.join(' '))
    }
  }

  return (
    <div className="history-scores">
      <div className="history-scores__meta">{meta}</div>
      {lines.length ? (
        lines.map((line) => (
          <div
            key={line}
            className={
              line.startsWith('XGB ') || line.startsWith('CrEnc ')
                ? line.startsWith('XGB ')
                  ? 'history-scores__xgb'
                  : 'history-scores__crenc'
                : undefined
            }
          >
            {line}
          </div>
        ))
      ) : (
        <div>—</div>
      )}
      {wineLine && <div className="history-scores__wine">{wineLine}</div>}
      <div
        className="history-scores__eval"
        onClick={(e) => e.stopPropagation()}
      >
        <label className="history-scores__check">
          <input
            type="checkbox"
            checked={fp === 1}
            onChange={() => {
              if (fp === 1) onEvalChange({ false_positive: 0, false_negative: 0 })
              else onEvalChange({ false_positive: 1, false_negative: 0 })
            }}
          />
          False positive
        </label>
        <label className="history-scores__check">
          <input
            type="checkbox"
            checked={fn === 1}
            onChange={() => {
              if (fn === 1) onEvalChange({ false_positive: 0, false_negative: 0 })
              else onEvalChange({ false_positive: 0, false_negative: 1 })
            }}
          />
          False negative
        </label>
      </div>
    </div>
  )
}

function HistoryOcrCompare({
  queryText,
  catalogText,
}: {
  queryText: string
  catalogText: string
}) {
  const matched = useMemo(
    () => sharedMatchTokens(queryText, catalogText),
    [queryText, catalogText],
  )
  return (
    <div className="history-ocr">
      <div className="history-ocr__cell">
        <strong>OCR искомого</strong>
        <pre>{highlightMatchedText(queryText, matched)}</pre>
      </div>
      <div className="history-ocr__cell">
        <strong>Этикетка найденного</strong>
        <pre>{highlightMatchedText(catalogText, matched)}</pre>
      </div>
    </div>
  )
}

export function ScanHistoryPage() {
  const navigate = useNavigate()
  const [items, setItems] = useState<ScanHistoryItem[]>([])
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sortState, setSortState] = useState<SortState>({
    sort: 'id',
    order: 'desc',
  })
  const [filterPositive, setFilterPositive] = useState(false)
  const [filterNegative, setFilterNegative] = useState(false)
  const [filterFp, setFilterFp] = useState(false)
  const [filterFn, setFilterFn] = useState(false)
  const [scanIdInput, setScanIdInput] = useState('')
  const [wineIdInput, setWineIdInput] = useState('')
  const [wineQFilter, setWineQFilter] = useState<string | null>(null)
  const [highlightScanId, setHighlightScanId] = useState<number | null>(null)
  const [reportOpen, setReportOpen] = useState(false)
  const [reportFrom, setReportFrom] = useState('')
  const [reportTo, setReportTo] = useState('')
  const [reportBusy, setReportBusy] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)
  const [savedReports, setSavedReports] = useState(() => listSavedReports())
  const sentinelRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef(false)
  const offsetRef = useRef(0)
  const hasMoreRef = useRef(true)
  const sortRef = useRef(sortState)
  const filtersRef = useRef({
    flags: [] as Array<'positive' | 'negative' | 'fp' | 'fn'>,
    wine_q: null as string | null,
  })
  const limit = 40

  sortRef.current = sortState
  const activeFlags = useMemo(() => {
    const flags: Array<'positive' | 'negative' | 'fp' | 'fn'> = []
    if (filterPositive) flags.push('positive')
    if (filterNegative) flags.push('negative')
    if (filterFp) flags.push('fp')
    if (filterFn) flags.push('fn')
    return flags
  }, [filterPositive, filterNegative, filterFp, filterFn])
  filtersRef.current = {
    flags: activeFlags,
    wine_q: wineQFilter,
  }

  const loadMore = useCallback(async (reset = false, aroundId?: number | null) => {
    if (loadingRef.current) return [] as ScanHistoryItem[]
    if (!reset && aroundId == null && !hasMoreRef.current) {
      return [] as ScanHistoryItem[]
    }
    loadingRef.current = true
    setBusy(true)
    setError(null)
    const nextOffset = reset || aroundId != null ? 0 : offsetRef.current
    const { sort, order } = sortRef.current
    const { flags, wine_q } = filtersRef.current
    try {
      const data = await fetchScanHistory({
        offset: nextOffset,
        limit,
        sort,
        order,
        flags,
        wine_q,
        around_id: aroundId ?? null,
      })
      setTotal(data.total)
      hasMoreRef.current = data.has_more
      setHasMore(data.has_more)
      offsetRef.current = data.offset + data.items.length
      setItems((prev) =>
        reset || aroundId != null ? data.items : [...prev, ...data.items],
      )
      return data.items
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка загрузки')
      return [] as ScanHistoryItem[]
    } finally {
      loadingRef.current = false
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    offsetRef.current = 0
    hasMoreRef.current = true
    setItems([])
    setHasMore(true)
    setHighlightScanId(null)
    void loadMore(true)
  }, [
    sortState.sort,
    sortState.order,
    filterPositive,
    filterNegative,
    filterFp,
    filterFn,
    wineQFilter,
    loadMore,
  ])

  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          void loadMore(false)
        }
      },
      { rootMargin: '240px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [loadMore])

  const highlightScrolledRef = useRef<number | null>(null)

  useEffect(() => {
    if (highlightScanId == null) {
      highlightScrolledRef.current = null
      return
    }
    // Scroll to the row once per highlight; do not re-jump when infinite
    // scroll appends more items (that was yanking the page back up).
    if (highlightScrolledRef.current === highlightScanId) return
    const row = document.querySelector(
      `tr[data-scan-id="${highlightScanId}"]`,
    )
    if (!row) return
    highlightScrolledRef.current = highlightScanId
    const t = window.setTimeout(() => {
      row.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 80)
    return () => window.clearTimeout(t)
  }, [highlightScanId, items])

  const toggleSort = (key: ScanHistorySort) => {
    setSortState((prev) =>
      prev.sort === key
        ? { sort: key, order: prev.order === 'asc' ? 'desc' : 'asc' }
        : { sort: key, order: 'desc' },
    )
  }

  const openSearch = (id: number) => {
    window.open(`/?scanid=${id}`, '_blank', 'noopener,noreferrer')
  }

  const openWine = (slug: string | null | undefined, id: number | null) => {
    if (!slug && id == null) return
    const path = `/wines/${encodeURIComponent(slug || String(id))}`
    window.open(path, '_blank', 'noopener,noreferrer')
  }

  const applyWineIdFilter = () => {
    const raw = wineIdInput.trim()
    if (!raw) {
      setWineQFilter(null)
      return
    }
    setWineQFilter(raw)
  }

  const jumpToScanId = async () => {
    const raw = scanIdInput.trim()
    if (!raw) return
    const n = Number(raw)
    if (!Number.isFinite(n) || n <= 0) {
      setError('Укажите корректный id поиска')
      return
    }
    const id = Math.trunc(n)
    const existing = items.find((it) => it.id === id)
    if (existing) {
      setHighlightScanId(id)
      return
    }
    const loaded = await loadMore(true, id)
    if (!loaded.length || !loaded.some((it) => it.id === id)) {
      setError(`Скан #${id} не найден (или не проходит текущие фильтры)`)
      setHighlightScanId(null)
      return
    }
    setHighlightScanId(id)
  }

  const setEval = async (
    scanId: number,
    next: { false_positive: number; false_negative: number },
  ) => {
    const fp = next.false_positive ? 1 : 0
    const fn = next.false_negative ? 1 : 0
    // Optimistic mutual-exclusive update
    setItems((prev) =>
      prev.map((it) =>
        it.id === scanId
          ? { ...it, false_positive: fp, false_negative: fn }
          : it,
      ),
    )
    try {
      const saved = await updateScanHistoryEval(scanId, {
        false_positive: fp,
        false_negative: fn,
      })
      setItems((prev) =>
        prev.map((it) =>
          it.id === scanId
            ? {
                ...it,
                false_positive: saved.false_positive ? 1 : 0,
                false_negative: saved.false_negative ? 1 : 0,
              }
            : it,
        ),
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ошибка сохранения Eval')
      // reload page slice would be heavy; leave optimistic and show error
    }
  }

  const setManualWine = async (
    scanId: number,
    wineId: number,
    checked: boolean,
  ) => {
    const nextId = checked ? wineId : null
    const prevId =
      items.find((it) => it.id === scanId)?.manual_wines_id ?? null
    setItems((prev) =>
      prev.map((it) =>
        it.id === scanId ? { ...it, manual_wines_id: nextId } : it,
      ),
    )
    try {
      const saved = await updateFindwineManualWine(scanId, nextId)
      setItems((prev) =>
        prev.map((it) =>
          it.id === scanId
            ? {
                ...it,
                manual_wines_id: saved.manual_wines_id ?? null,
                ...(checked
                  ? {
                      false_positive: saved.false_positive ? 1 : 0,
                      false_negative: saved.false_negative ? 1 : 0,
                    }
                  : {}),
              }
            : it,
        ),
      )
    } catch (e) {
      setItems((prev) =>
        prev.map((it) =>
          it.id === scanId ? { ...it, manual_wines_id: prevId } : it,
        ),
      )
      setError(
        e instanceof Error ? e.message : 'Ошибка сохранения «Это вино»',
      )
    }
  }

  const sortMark = (key: ScanHistorySort) =>
    sortState.sort === key ? (sortState.order === 'asc' ? ' ↑' : ' ↓') : ''

  const openReportDialog = () => {
    const ids = items.map((it) => it.id)
    if (!reportFrom && ids.length) {
      setReportFrom(String(Math.min(...ids)))
    }
    if (!reportTo && ids.length) {
      setReportTo(String(Math.max(...ids)))
    }
    setReportError(null)
    setSavedReports(listSavedReports())
    setReportOpen(true)
  }

  const buildReport = async () => {
    const from = Number(reportFrom)
    const to = Number(reportTo)
    if (!Number.isFinite(from) || !Number.isFinite(to) || from < 1 || to < 1) {
      setReportError('Укажите корректные id «с» и «по»')
      return
    }
    setReportBusy(true)
    setReportError(null)
    try {
      const data = await fetchScanHistoryReport({ id_from: from, id_to: to })
      if (!data.counts.n) {
        setReportError('В диапазоне нет сканов')
        return
      }
      const saved = saveReport(data)
      setSavedReports(listSavedReports())
      setReportOpen(false)
      navigate(`/history/report/${saved.key}`)
    } catch (e) {
      setReportError(e instanceof Error ? e.message : 'Ошибка отчёта')
    } finally {
      setReportBusy(false)
    }
  }

  return (
    <main className="page history-page">
      <div className="page__intro">
        <h1>История сканирования</h1>
        <p className="page__lead">
          Все запросы поиска ({total}). Новые записи подгружаются при прокрутке.
        </p>
      </div>

      {error && <p className="history-error">{error}</p>}

      <div className="history-filters">
        <div className="history-filters__flags" role="group" aria-label="Тип результата">
          <label className="history-filters__check">
            <input
              type="checkbox"
              checked={filterPositive}
              onChange={(e) => setFilterPositive(e.target.checked)}
            />
            Positive
          </label>
          <label className="history-filters__check">
            <input
              type="checkbox"
              checked={filterNegative}
              onChange={(e) => setFilterNegative(e.target.checked)}
            />
            Negative
          </label>
          <label className="history-filters__check">
            <input
              type="checkbox"
              checked={filterFp}
              onChange={(e) => setFilterFp(e.target.checked)}
            />
            False positive
          </label>
          <label className="history-filters__check">
            <input
              type="checkbox"
              checked={filterFn}
              onChange={(e) => setFilterFn(e.target.checked)}
            />
            False negative
          </label>
          <button
            type="button"
            className="history-filters__btn history-filters__btn--report"
            onClick={openReportDialog}
          >
            Отчёт
          </button>
        </div>
        <div className="history-filters__ids">
          <label className="history-filters__field">
            <span>id поиска</span>
            <input
              type="number"
              inputMode="numeric"
              min={1}
              placeholder="например 818"
              value={scanIdInput}
              onChange={(e) => setScanIdInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  void jumpToScanId()
                }
              }}
            />
            <button type="button" className="history-filters__btn" onClick={() => void jumpToScanId()}>
              Перейти
            </button>
          </label>
          <label className="history-filters__field">
            <span>wine id / slug</span>
            <input
              type="text"
              inputMode="search"
              autoComplete="off"
              placeholder="id или часть slug"
              value={wineIdInput}
              onChange={(e) => setWineIdInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  applyWineIdFilter()
                }
              }}
            />
            <button type="button" className="history-filters__btn" onClick={applyWineIdFilter}>
              Фильтр
            </button>
            {wineQFilter != null && (
              <button
                type="button"
                className="history-filters__btn history-filters__btn--ghost"
                onClick={() => {
                  setWineQFilter(null)
                  setWineIdInput('')
                }}
              >
                Сброс
              </button>
            )}
          </label>
        </div>
      </div>

      <div className="history-table-wrap">
        <table className="history-table">
          <thead>
            <tr>
              <th className="col-photo">Искомая этикетка</th>
              <th className="col-ocr">OCR</th>
              <th className="col-photo">Найденная этикетка</th>
              <th className="col-score">
                <button
                  type="button"
                  className="sort-btn"
                  onClick={() => toggleSort('confidence')}
                >
                  Score{sortMark('confidence')}
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => {
              const noMatch = row.matched_wine_id == null
              const topCands = row.xgb_top || []
              return (
              <tr
                key={row.id}
                data-scan-id={row.id}
                className={`history-row${
                  highlightScanId === row.id ? ' is-highlight' : ''
                }`}
                onClick={() => openSearch(row.id)}
              >
                <td className="col-photo">
                  {row.query_photo_url ? (
                    <img src={row.query_photo_url} alt="" className="history-thumb" />
                  ) : (
                    <div className="history-thumb history-thumb--empty" />
                  )}
                </td>
                {noMatch ? (
                  <td
                    className="col-nomatch"
                    colSpan={2}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="history-nomatch">
                      <div className="history-nomatch__ocr">
                        <strong>OCR искомого</strong>
                        <pre>
                          {String(row.query_ocr_text || '').trim() || '—'}
                        </pre>
                      </div>
                      <div className="history-nomatch__cands">
                        {topCands.length > 0 ? (
                          <div className="history-xgb-top history-xgb-top--7">
                            {topCands.map((w) => (
                              <HistoryXgbTopCard
                                key={w.id}
                                wine={w}
                                ocrQueryText={row.query_ocr_text || undefined}
                                onOpenWine={openWine}
                                isManualMatch={
                                  row.manual_wines_id != null &&
                                  row.manual_wines_id === w.id
                                }
                                onManualMatchChange={(wineId, checked) =>
                                  void setManualWine(row.id, wineId, checked)
                                }
                              />
                            ))}
                          </div>
                        ) : (
                          <div className="history-thumb history-thumb--empty" />
                        )}
                      </div>
                    </div>
                  </td>
                ) : (
                  <>
                    <td className="col-ocr" onClick={(e) => e.stopPropagation()}>
                      <HistoryOcrCompare
                        queryText={String(row.query_ocr_text || '').trim()}
                        catalogText={String(row.matched_wine_label || '').trim()}
                      />
                    </td>
                    <td
                      className="col-photo col-found"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {row.matched_wine_photo_url ? (
                        <button
                          type="button"
                          className="history-thumb-btn"
                          title={row.matched_wine_name || undefined}
                          onClick={(e) => {
                            e.stopPropagation()
                            openWine(row.matched_wine_slug, row.matched_wine_id)
                          }}
                        >
                          <img
                            src={row.matched_wine_photo_url}
                            alt={row.matched_wine_name || ''}
                            className="history-thumb"
                          />
                        </button>
                      ) : (
                        <div className="history-thumb history-thumb--empty" />
                      )}
                    </td>
                  </>
                )}
                <td
                  className="col-score"
                  onClick={(e) => e.stopPropagation()}
                >
                  <HistoryScores
                    scores={row.scores}
                    scanId={row.id}
                    totalMs={row.total_ms}
                    wineId={row.matched_wine_id}
                    wineSlug={row.matched_wine_slug}
                    falsePositive={row.false_positive ? 1 : 0}
                    falseNegative={row.false_negative ? 1 : 0}
                    onEvalChange={(next) => setEval(row.id, next)}
                  />
                </td>
              </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div ref={sentinelRef} className="history-sentinel">
        {busy ? 'Загрузка…' : hasMore ? '' : items.length ? 'Все записи загружены' : 'Нет записей'}
      </div>

      {reportOpen &&
        createPortal(
          <div
            className="history-report-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="history-report-title"
            onClick={() => !reportBusy && setReportOpen(false)}
          >
            <div
              className="history-report-modal__panel"
              onClick={(e) => e.stopPropagation()}
            >
              <h2 id="history-report-title">Отчёт по диапазону id</h2>
              <p className="history-report-modal__hint">
                Inclusive: id с … по …. Класс учитывает manual wine: matched≠manual →
                FP; нет matched + есть manual → FN.
              </p>
              <div className="history-report-modal__fields">
                <label>
                  <span>id с</span>
                  <input
                    type="number"
                    inputMode="numeric"
                    min={1}
                    value={reportFrom}
                    onChange={(e) => setReportFrom(e.target.value)}
                    disabled={reportBusy}
                  />
                </label>
                <label>
                  <span>id по</span>
                  <input
                    type="number"
                    inputMode="numeric"
                    min={1}
                    value={reportTo}
                    onChange={(e) => setReportTo(e.target.value)}
                    disabled={reportBusy}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        void buildReport()
                      }
                    }}
                  />
                </label>
              </div>
              {reportError && (
                <p className="history-error">{reportError}</p>
              )}
              <div className="history-report-modal__actions">
                <button
                  type="button"
                  className="history-filters__btn history-filters__btn--ghost"
                  disabled={reportBusy}
                  onClick={() => setReportOpen(false)}
                >
                  Отмена
                </button>
                <button
                  type="button"
                  className="history-filters__btn history-filters__btn--report"
                  disabled={reportBusy}
                  onClick={() => void buildReport()}
                >
                  {reportBusy ? 'Считаем…' : 'Построить'}
                </button>
              </div>
              {savedReports.length > 0 && (
                <div className="history-report-modal__saved">
                  <strong>Сохранённые в браузере</strong>
                  <ul>
                    {savedReports.slice(0, 12).map((r) => (
                      <li key={r.key}>
                        <Link
                          to={`/history/report/${r.key}`}
                          onClick={() => setReportOpen(false)}
                        >
                          {r.name}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>,
          document.body,
        )}
    </main>
  )
}
