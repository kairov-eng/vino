import type { FindWineResult } from './api/client'

const SCAN_CACHE_KEY = 'vino_scan_cache_v1'

export type ScanSessionCache = {
  id: number
  result: FindWineResult
  status: string | null
  fileName: string | null
  savedAt: number
}

/** Survives SPA remounts without a full document reload. */
let memoryCache: ScanSessionCache | null = null

export function readScanSessionCache(id: number): ScanSessionCache | null {
  if (memoryCache?.id === id && memoryCache.result) return memoryCache
  if (typeof sessionStorage === 'undefined') return null
  try {
    const raw = sessionStorage.getItem(SCAN_CACHE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as ScanSessionCache
    if (!parsed || parsed.id !== id || !parsed.result) return null
    memoryCache = parsed
    return parsed
  } catch {
    return null
  }
}

export function writeScanSessionCache(entry: ScanSessionCache) {
  memoryCache = entry
  if (typeof sessionStorage === 'undefined') return
  try {
    sessionStorage.setItem(SCAN_CACHE_KEY, JSON.stringify(entry))
  } catch {
    /* quota / private mode */
  }
}

export function clearScanSessionCache() {
  memoryCache = null
  if (typeof sessionStorage === 'undefined') return
  try {
    sessionStorage.removeItem(SCAN_CACHE_KEY)
  } catch {
    /* ignore */
  }
}
