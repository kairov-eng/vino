import type { CSSProperties } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  deleteSavedReport,
  getSavedReport,
  listSavedReports,
  type SavedScanEvalReport,
} from './scanEvalReportStorage'
import './ScanHistoryReportPage.css'

function pct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(1)}%`
}

function reasonLabel(reason: string): string {
  switch (reason) {
    case 'manual_differs_from_matched':
      return 'matched ≠ manual'
    case 'manual_without_matched':
      return 'нет matched, есть manual'
    case 'checkbox_fp':
      return 'checkbox FP'
    case 'checkbox_fn':
      return 'checkbox FN'
    default:
      return reason
  }
}

function fmt01(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return v.toFixed(3)
}

function scoreDistTitle(d: {
  metric: string
  polarity: string
}): string {
  const m = d.metric === 'xgb' ? 'XGB' : 'Cos'
  const p = d.polarity === 'plus' ? '+' : '−'
  return `${m}${p}`
}

function ScoreDistTables({
  dists,
}: {
  dists: NonNullable<SavedScanEvalReport['data']['score_dists']>
}) {
  if (!dists.length) return null
  const order = [
    ['cosine', 'plus'],
    ['cosine', 'minus'],
    ['xgb', 'plus'],
    ['xgb', 'minus'],
  ] as const
  const sorted = order
    .map(([m, p]) => dists.find((d) => d.metric === m && d.polarity === p))
    .filter(Boolean) as typeof dists

  return (
    <section className="eval-report__section">
      <h2>Распределения Cos / XGB</h2>
      <p className="eval-report__dist-hint">
        + целевое вино (manual, иначе matched без FP). − остальные из пула
        кандидатов (~20), включая ошибочный FP-финалист.
      </p>
      <div className="eval-report__dist-grid">
        {sorted.map((d) => (
          <div key={`${d.metric}-${d.polarity}`} className="eval-report__dist-card">
            <h3>
              {scoreDistTitle(d)}{' '}
              <span className="eval-report__dist-n">n={d.n}</span>
            </h3>
            {d.n === 0 ? (
              <p className="eval-report__empty">Нет данных</p>
            ) : (
              <>
                <table className="eval-report__dist-table">
                  <thead>
                    <tr>
                      <th>Диапазон</th>
                      <th>n</th>
                      <th>%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.bins.map((b) => (
                      <tr key={b.range} className={b.n ? undefined : 'is-zero'}>
                        <td>{b.range}</td>
                        <td>{b.n}</td>
                        <td>{b.pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="eval-report__dist-stats">
                  min {fmt01(d.min)} · median {fmt01(d.median)} · mean{' '}
                  {fmt01(d.mean)} · max {fmt01(d.max)}
                </p>
              </>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function fmtSec(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return `${v.toFixed(2)} с`
}

function TimingHistogram({
  dist,
}: {
  dist: NonNullable<SavedScanEvalReport['data']['timing_dist']>
}) {
  if (!dist.n || !dist.bins.length) {
    return <p className="eval-report__empty">Нет данных о времени</p>
  }
  const bins = dist.bins.filter((b) => b.n > 0)
  if (!bins.length) {
    return <p className="eval-report__empty">Нет данных о времени</p>
  }
  const maxN = Math.max(1, ...bins.map((b) => b.n))
  return (
    <>
      <p className="eval-report__timing-summary">
        среднее {fmtSec(dist.mean_sec)} · медиана {fmtSec(dist.median_sec)}
        <span className="eval-report__dist-n"> · n={dist.n}</span>
      </p>
      <div
        className="eval-report__timing-chart"
        role="img"
        aria-label="Гистограмма времени выполнения запросов"
      >
        <div
          className="eval-report__timing-bars"
          style={
            { ['--timing-cols']: String(bins.length) } as CSSProperties
          }
        >
          {bins.map((b) => (
            <div key={b.seconds} className="eval-report__timing-col">
              <div className="eval-report__timing-top">
                <span className="eval-report__timing-n">{b.n}</span>
                <span className="eval-report__timing-pct">
                  {b.pct.toFixed(1)}%
                </span>
              </div>
              <div className="eval-report__timing-track">
                <div
                  className="eval-report__timing-fill"
                  style={{ height: `${(b.n / maxN) * 100}%` }}
                />
              </div>
              <span className="eval-report__timing-x">{b.seconds}</span>
            </div>
          ))}
        </div>
        <div className="eval-report__timing-axis">
          <span>сек</span>
        </div>
      </div>
    </>
  )
}

function ClassBars({
  TP,
  FP,
  TN,
  FN,
}: {
  TP: number
  FP: number
  TN: number
  FN: number
}) {
  const n = Math.max(1, TP + FP + TN + FN)
  const rows = [
    { key: 'TP', label: 'Positive (TP)', value: TP, cls: 'is-tp' },
    { key: 'FP', label: 'False positive', value: FP, cls: 'is-fp' },
    { key: 'TN', label: 'Negative (TN)', value: TN, cls: 'is-tn' },
    { key: 'FN', label: 'False negative', value: FN, cls: 'is-fn' },
  ]
  return (
    <div className="eval-report__bars" role="img" aria-label="Распределение классов">
      {rows.map((r) => (
        <div key={r.key} className="eval-report__bar-row">
          <span className="eval-report__bar-label">{r.label}</span>
          <div className="eval-report__bar-track">
            <div
              className={`eval-report__bar-fill ${r.cls}`}
              style={{ width: `${(r.value / n) * 100}%` }}
            />
          </div>
          <span className="eval-report__bar-value">{r.value}</span>
        </div>
      ))}
    </div>
  )
}

function PieChart({
  TP,
  FP,
  TN,
  FN,
}: {
  TP: number
  FP: number
  TN: number
  FN: number
}) {
  const parts = [
    { v: TP, color: '#2f7d4a' },
    { v: FP, color: '#c00c1a' },
    { v: TN, color: '#8a7468' },
    { v: FN, color: '#c47a12' },
  ]
  const total = parts.reduce((s, p) => s + p.v, 0) || 1
  let acc = 0
  const stops = parts
    .map((p) => {
      const start = (acc / total) * 360
      acc += p.v
      const end = (acc / total) * 360
      return `${p.color} ${start}deg ${end}deg`
    })
    .join(', ')
  return (
    <div className="eval-report__pie-wrap">
      <div
        className="eval-report__pie"
        style={{ background: `conic-gradient(${stops})` }}
        aria-hidden
      />
      <ul className="eval-report__pie-legend">
        <li>
          <i className="is-tp" /> TP {TP}
        </li>
        <li>
          <i className="is-fp" /> FP {FP}
        </li>
        <li>
          <i className="is-tn" /> TN {TN}
        </li>
        <li>
          <i className="is-fn" /> FN {FN}
        </li>
      </ul>
    </div>
  )
}

function ErrorTable({
  title,
  items,
  tone,
}: {
  title: string
  items: SavedScanEvalReport['data']['false_positives']
  tone: 'fp' | 'fn'
}) {
  if (!items.length) {
    return (
      <section className="eval-report__section">
        <h2>{title}</h2>
        <p className="eval-report__empty">Нет записей</p>
      </section>
    )
  }
  return (
    <section className="eval-report__section">
      <h2>
        {title} ({items.length})
      </h2>
      <div className="eval-report__table-wrap">
        <table className={`eval-report__table is-${tone}`}>
          <thead>
            <tr>
              <th>Скан</th>
              <th>matched</th>
              <th>manual</th>
              <th>причина</th>
              <th>fp/fn</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.id}>
                <td>
                  <Link to={`/?scanid=${row.id}`}>{row.id}</Link>
                </td>
                <td>{row.matched_wine_id ?? '—'}</td>
                <td>{row.manual_wines_id ?? '—'}</td>
                <td>{reasonLabel(row.reason)}</td>
                <td>
                  {row.hist_fp}/{row.hist_fn}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export function ScanHistoryReportPage() {
  const { reportKey } = useParams<{ reportKey: string }>()
  const navigate = useNavigate()
  const report = reportKey ? getSavedReport(reportKey) : null
  const saved = listSavedReports()

  if (!report) {
    return (
      <main className="page eval-report-page">
        <div className="page__intro">
          <h1>Отчёт не найден</h1>
          <p className="page__lead">
            Отчёт хранится только в этом браузере (localStorage). Постройте новый
            на странице истории.
          </p>
        </div>
        <p>
          <Link to="/history">← К истории</Link>
        </p>
        {saved.length > 0 && (
          <section className="eval-report__section">
            <h2>Сохранённые отчёты</h2>
            <ul className="eval-report__saved">
              {saved.map((r) => (
                <li key={r.key}>
                  <Link to={`/history/report/${r.key}`}>{r.name}</Link>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
    )
  }

  const { data, name } = report
  const { counts, metrics } = data

  return (
    <main className="page eval-report-page">
      <div className="page__intro">
        <p className="eval-report__back">
          <Link to="/history">← История</Link>
        </p>
        <h1>{name}</h1>
        <p className="page__lead">
          Диапазон запроса id {data.id_from}–{data.id_to}
          {counts.min_id != null && counts.max_id != null
            ? ` · в выборке ${counts.min_id}–${counts.max_id}`
            : ''}{' '}
          · N={counts.n} · сохранено{' '}
          {new Date(report.createdAt).toLocaleString('ru-RU')}
        </p>
      </div>

      <div className="eval-report__stats">
        <div className="eval-report__stat is-tp">
          <strong>{counts.TP}</strong>
          <span>Positive (TP)</span>
        </div>
        <div className="eval-report__stat is-fp">
          <strong>{counts.FP}</strong>
          <span>False positive</span>
        </div>
        <div className="eval-report__stat is-tn">
          <strong>{counts.TN}</strong>
          <span>Negative (TN)</span>
        </div>
        <div className="eval-report__stat is-fn">
          <strong>{counts.FN}</strong>
          <span>False negative</span>
        </div>
      </div>

      <div className="eval-report__grid">
        <section className="eval-report__card">
          <h2>Метрики</h2>
          <table className="eval-report__metrics">
            <tbody>
              <tr>
                <th>F1</th>
                <td>{pct(metrics.f1)}</td>
              </tr>
              <tr>
                <th>Precision</th>
                <td>{pct(metrics.precision)}</td>
              </tr>
              <tr>
                <th>Recall</th>
                <td>{pct(metrics.recall)}</td>
              </tr>
              <tr>
                <th>Accuracy</th>
                <td>{pct(metrics.accuracy)}</td>
              </tr>
              <tr>
                <th>Specificity</th>
                <td>{pct(metrics.specificity)}</td>
              </tr>
              <tr>
                <th>NPV</th>
                <td>{pct(metrics.npv)}</td>
              </tr>
              <tr>
                <th>FPR</th>
                <td>{pct(metrics.fpr)}</td>
              </tr>
              <tr>
                <th>FNR</th>
                <td>{pct(metrics.fnr)}</td>
              </tr>
              <tr>
                <th>Match rate</th>
                <td>{pct(metrics.match_rate)}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="eval-report__card">
          <h2>График классов</h2>
          <PieChart
            TP={counts.TP}
            FP={counts.FP}
            TN={counts.TN}
            FN={counts.FN}
          />
          <ClassBars
            TP={counts.TP}
            FP={counts.FP}
            TN={counts.TN}
            FN={counts.FN}
          />
        </section>
      </div>

      <div className="eval-report__grid eval-report__grid--timing">
        <section className="eval-report__card">
          <h2>Время выполнения</h2>
          {data.timing_dist ? (
            <TimingHistogram dist={data.timing_dist} />
          ) : (
            <p className="eval-report__empty">
              В этом отчёте нет статистики времени — постройте отчёт заново
              кнопкой «Отчёт» в истории.
            </p>
          )}
        </section>

        <section className="eval-report__card eval-report__rules">
          <h2>Правила класса</h2>
          <ol>
            {(data.rules || []).map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
        </section>
      </div>

      {data.score_dists && data.score_dists.length > 0 ? (
        <ScoreDistTables dists={data.score_dists} />
      ) : (
        <section className="eval-report__section">
          <h2>Распределения Cos / XGB</h2>
          <p className="eval-report__empty">
            В этом отчёте нет score_dists — постройте отчёт заново кнопкой
            «Отчёт» в истории.
          </p>
        </section>
      )}

      <ErrorTable
        title="False positive"
        items={data.false_positives}
        tone="fp"
      />
      <ErrorTable
        title="False negative"
        items={data.false_negatives}
        tone="fn"
      />

      {saved.length > 1 && (
        <section className="eval-report__section">
          <h2>Другие сохранённые отчёты</h2>
          <ul className="eval-report__saved">
            {saved
              .filter((r) => r.key !== report.key)
              .map((r) => (
                <li key={r.key}>
                  <Link to={`/history/report/${r.key}`}>{r.name}</Link>
                  <button
                    type="button"
                    className="eval-report__del"
                    onClick={() => {
                      deleteSavedReport(r.key)
                      navigate(`/history/report/${report.key}`, { replace: true })
                    }}
                  >
                    удалить
                  </button>
                </li>
              ))}
          </ul>
        </section>
      )}
    </main>
  )
}
