import type { ScanHistoryReportResponse } from '../api/client'

const STORAGE_KEY = 'vino:scan-eval-reports:v1'

export type SavedScanEvalReport = {
  key: string
  name: string
  createdAt: string
  idFrom: number
  idTo: number
  data: ScanHistoryReportResponse
}

function readAll(): SavedScanEvalReport[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as SavedScanEvalReport[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function writeAll(items: SavedScanEvalReport[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
}

export function buildReportName(data: ScanHistoryReportResponse): string {
  const { counts, id_from, id_to } = data
  const lo = counts.min_id ?? id_from
  const hi = counts.max_id ?? id_to
  return `id ${lo}–${hi} · TP${counts.TP} FP${counts.FP} TN${counts.TN} FN${counts.FN}`
}

export function listSavedReports(): SavedScanEvalReport[] {
  return readAll().sort((a, b) => b.createdAt.localeCompare(a.createdAt))
}

export function getSavedReport(key: string): SavedScanEvalReport | null {
  return readAll().find((r) => r.key === key) ?? null
}

export function saveReport(
  data: ScanHistoryReportResponse,
): SavedScanEvalReport {
  const item: SavedScanEvalReport = {
    key: `r-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: buildReportName(data),
    createdAt: new Date().toISOString(),
    idFrom: data.id_from,
    idTo: data.id_to,
    data,
  }
  const next = [item, ...readAll().filter((r) => r.name !== item.name)]
  writeAll(next.slice(0, 40))
  return item
}

export function deleteSavedReport(key: string) {
  writeAll(readAll().filter((r) => r.key !== key))
}
