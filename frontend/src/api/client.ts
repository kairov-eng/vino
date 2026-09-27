export type Wine = {
  id: number
  wineries_id: number | null
  name: string
  category: string | null
  color: string | null
  region: string | null
  grape_variety: string | null
  description: string | null
  winery: string | null
  slug: string | null
  photo_name: string | null
  photo_url: string | null
  crop: number | null
  label: string | null
  label_url: string | null
  label_ocr?: Record<string, unknown> | null
}

export type FilterFacet = { value: string; count: number }

export type FiltersResponse = {
  category: FilterFacet[]
  color: FilterFacet[]
  region: FilterFacet[]
  grape_variety: FilterFacet[]
  winery: FilterFacet[]
  name: FilterFacet[]
}

export type WineListResponse = {
  items: Wine[]
  total: number
  offset: number
  limit: number
  has_more: boolean
}

export type WineFilters = {
  category: string[]
  color: string[]
  region: string[]
  grape_variety: string[]
  winery: string[]
  name: string[]
}

const API_BASE = import.meta.env.VITE_API_URL || ''

export type SiteAuthResponse = {
  ok: boolean
  role: 'admin' | 'user' | null
  enabled: boolean
}

export async function checkSiteAuth(
  password?: string | null,
): Promise<SiteAuthResponse> {
  const headers: Record<string, string> = {}
  const p = String(password || '').trim()
  if (p) headers['X-Site-Password'] = p
  const res = await fetch(`${API_BASE}/api/site-auth`, {
    method: 'GET',
    credentials: 'include',
    headers,
  })
  if (!res.ok) {
    return { ok: false, role: null, enabled: true }
  }
  return res.json()
}

export async function loginSiteAuth(password: string): Promise<SiteAuthResponse> {
  const res = await fetch(`${API_BASE}/api/site-auth`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })
  if (!res.ok) {
    return { ok: false, role: null, enabled: true }
  }
  return res.json()
}

function buildQuery(params: Record<string, string | number | string[] | undefined>) {
  const qs = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === '') return
    if (Array.isArray(value)) {
      value.forEach((v) => qs.append(key, v))
    } else {
      qs.set(key, String(value))
    }
  })
  return qs.toString()
}

export async function fetchFilters(): Promise<FiltersResponse> {
  const res = await fetch(`${API_BASE}/api/wines/filters`)
  if (!res.ok) throw new Error('Не удалось загрузить фильтры')
  return res.json()
}

export async function fetchWines(
  filters: WineFilters,
  offset: number,
  limit = 24,
  q?: string,
): Promise<WineListResponse> {
  const query = buildQuery({ ...filters, offset, limit, q })
  const res = await fetch(`${API_BASE}/api/wines?${query}`)
  if (!res.ok) throw new Error('Не удалось загрузить вина')
  return res.json()
}

export async function fetchWine(slug: string): Promise<Wine> {
  const res = await fetch(`${API_BASE}/api/wines/${encodeURIComponent(slug)}`)
  if (!res.ok) throw new Error('Вино не найдено')
  return res.json()
}

export async function previewImage(file: File): Promise<string> {
  const body = new FormData()
  body.append('file', file)
  const res = await fetch(`${API_BASE}/api/preview-image`, {
    method: 'POST',
    body,
  })
  if (!res.ok) {
    let detail = 'Не удалось построить превью'
    try {
      const payload = await res.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

function canBrowserPreview(file: File): boolean {
  const name = file.name.toLowerCase()
  if (/\.(heic|heif|tif|tiff|jp2|j2k|jpx)$/i.test(name)) return false
  const t = (file.type || '').toLowerCase()
  if (!t) return false
  if (t === 'image/heic' || t === 'image/heif' || t === 'image/tiff') return false
  return t.startsWith('image/')
}

export async function makePreviewUrl(file: File): Promise<string> {
  if (canBrowserPreview(file)) {
    const url = URL.createObjectURL(file)
    const ok = await new Promise<boolean>((resolve) => {
      const img = new Image()
      img.onload = () => resolve(true)
      img.onerror = () => resolve(false)
      img.src = url
    })
    if (ok) return url
    URL.revokeObjectURL(url)
  }
  return previewImage(file)
}

export type FindWineCandidate = {
  id: number
  cosine_similarity: number
  name: string | null
  winery: string | null
  slug: string | null
  label: string | null
  photo_url: string | null
  label_url: string | null
  geometry_ok?: boolean | null
  geometry_inliers?: number | null
  geometry_matches?: number | null
  geometry_inlier_ratio?: number | null
  geometry_score?: number | null
  xgb_score?: number | null
  xgb_fin?: number | null
  /** XGB_fin без hard reject (для UI серым при reject) */
  xgb_fin_pre_reject?: number | null
  xgb_suitable?: boolean | null
  xgb_fin_reason?: string | null
  exclusive_rejected?: boolean | null
  label_text_hard_reject?: string | null
  crenc_score?: number | null
  crenc_fin?: number | null
  crenc_suitable?: boolean | null
  crenc_fin_reason?: string | null
  /** OpenAI сравнение текста — вероятность 0…1 */
  llm_txt?: number | null
  /** Bhattacharyya HSV: 0 одна гамма, 1 гаммы не пересекаются */
  hsv?: number | null
  /** CIEDE2000 доминантных цветов query↔candidate (меньше = ближе) */
  color_delta?: number | null
  /** Final score per OCR (selected text method) */
  final_by_ocr?: Record<string, number> | null
  final_ocr_primary?: string | null
}

export type FindWineResult = {
  search_photos_id: number | null
  crops_filename: string | null
  label_filename: string | null
  crops_url: string | null
  label_url: string | null
  algorithm_version: string | null
  algorithm_updated_at: string | null
  candidates_siglip2: FindWineCandidate[]
  candidates_dinov3: FindWineCandidate[]
  matched_wine_id?: number | null
  matched_wine_confidence?: number | null
  matched_wine_name?: string | null
  matched_wine_slug?: string | null
  matched_wine_photo_url?: string | null
  /** Ручная отметка «Соответствие» */
  manual_wines_id?: number | null
  status: {
    ok?: boolean
    error?: string | null
    search_photos_id?: number
    crops_filename?: string
    label_filename?: string
    algorithm_version?: string
    algorithm_updated_at?: string
    candidates_siglip2_ids?: number[]
    candidates_dinov3_ids?: number[]
    steps?: Record<string, any>
    timings_ms?: Record<string, number>
    analogs?: WineAnalogs
    [key: string]: unknown
  }
}

/** true — совпал, false — не совпал, 'partial' — соседний тип, null — у вина не указано */
export type AnalogMatchFlag = boolean | 'partial' | null

export type WineAnalogItem = {
  id: number
  name: string | null
  slug: string | null
  winery: string | null
  category: string | null
  type: string | null
  grapes: { name: string; matched: boolean }[]
  matched: Partial<Record<'winery' | 'grape' | 'category' | 'type', AnalogMatchFlag>>
  matched_count: number
  criteria_count: number
  score: number
  cosine: number | null
  label_url: string | null
}

export type WineAnalogs = {
  ok: boolean
  error?: string
  skipped?: boolean
  reason?: string
  source?: string
  ms?: number
  criteria: {
    winery: string[]
    category: string[]
    type: string | null
    grape: string[]
  }
  criteria_found: string[]
  items: WineAnalogItem[]
}

export async function findWine(file: File): Promise<FindWineResult> {
  const body = new FormData()
  body.append('file', file)
  let res: Response
  try {
    res = await fetch(`${API_BASE}/api/findwine`, { method: 'POST', body })
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    if (/failed to fetch|networkerror|load failed/i.test(msg)) {
      throw new Error(
        'Нет связи с API (бэкенд перезапускается или таймаут). Повторите поиск.',
      )
    }
    throw err instanceof Error ? err : new Error(msg)
  }
  if (!res.ok) {
    let detail = 'Ошибка поиска вина'
    try {
      const payload = await res.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export async function fetchFindWineResult(id: number): Promise<FindWineResult> {
  const url = `${API_BASE}/api/findwine/${id}`
  const attempt = async (): Promise<Response> => {
    try {
      return await fetch(url)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      if (/failed to fetch|networkerror|load failed/i.test(msg)) {
        throw new Error(
          'Нет связи с API (бэкенд перезапускается или таймаут). Обновите страницу.',
        )
      }
      throw err instanceof Error ? err : new Error(msg)
    }
  }
  let res: Response
  try {
    res = await attempt()
  } catch (err) {
    // Один повтор — типичный reload uvicorn между запросами
    await new Promise((r) => setTimeout(r, 800))
    res = await attempt()
  }
  if (!res.ok) {
    let detail = 'Результат поиска не найден'
    try {
      const payload = await res.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export type ScanHistorySort =
  | 'id'
  | 'created_at'
  | 'total_ms'
  | 'matched_wine_id'
  | 'confidence'

export type ScanHistoryXgbTopItem = {
  id: number
  name: string | null
  slug: string | null
  winery: string | null
  label: string | null
  label_url: string | null
  photo_url: string | null
  xgb_score: number | null
  xgb_fin: number | null
  cosine: number | null
}

export type ScanHistoryItem = {
  id: number
  created_at: string | null
  query_photo_url: string | null
  query_filename: string | null
  total_ms: number | null
  ocr_texts: Record<string, string | null>
  query_ocr_text?: string | null
  matched_wine_id: number | null
  matched_wine_confidence: number | null
  matched_wine_name: string | null
  matched_wine_slug: string | null
  matched_wine_photo_url: string | null
  matched_wine_label?: string | null
  /** Top-3 по XGB_fin, если финального вина нет */
  xgb_top?: ScanHistoryXgbTopItem[] | null
  scores?: Record<string, number | string | null> | null
  algorithm_version: string | null
  false_positive?: number
  false_negative?: number
}

export type ScanHistoryResponse = {
  items: ScanHistoryItem[]
  total: number
  offset: number
  limit: number
  has_more: boolean
}

export type ScanHistoryReportErrorItem = {
  id: number
  matched_wine_id: number | null
  manual_wines_id: number | null
  hist_fp: number
  hist_fn: number
  reason: string
  cls: string
}

export type ScanHistoryReportScoreDist = {
  metric: string
  polarity: string
  n: number
  bins: Array<{ range: string; n: number; pct: number }>
  min?: number | null
  max?: number | null
  mean?: number | null
  median?: number | null
}

export type ScanHistoryReportResponse = {
  id_from: number
  id_to: number
  counts: {
    TP: number
    FP: number
    TN: number
    FN: number
    n: number
    min_id: number | null
    max_id: number | null
  }
  metrics: {
    precision: number | null
    recall: number | null
    f1: number | null
    specificity: number | null
    npv: number | null
    accuracy: number | null
    fpr: number | null
    fnr: number | null
    match_rate: number | null
  }
  false_positives: ScanHistoryReportErrorItem[]
  false_negatives: ScanHistoryReportErrorItem[]
  rules: string[]
  score_dists?: ScanHistoryReportScoreDist[]
}

export async function fetchScanHistory(params: {
  offset?: number
  limit?: number
  sort?: ScanHistorySort
  order?: 'asc' | 'desc'
  flags?: Array<'positive' | 'negative' | 'fp' | 'fn'>
  wine_q?: string | null
  matched_wine_id?: number | null
  around_id?: number | null
}): Promise<ScanHistoryResponse> {
  const qs = buildQuery({
    offset: params.offset ?? 0,
    limit: params.limit ?? 40,
    sort: params.sort ?? 'id',
    order: params.order ?? 'desc',
    flag: params.flags?.length ? params.flags : undefined,
    wine_q:
      params.wine_q != null && String(params.wine_q).trim()
        ? String(params.wine_q).trim()
        : undefined,
    matched_wine_id:
      params.matched_wine_id != null && params.matched_wine_id > 0
        ? params.matched_wine_id
        : undefined,
    around_id:
      params.around_id != null && params.around_id > 0
        ? params.around_id
        : undefined,
  })
  const res = await fetch(`${API_BASE}/api/scan-history?${qs}`)
  if (!res.ok) throw new Error('Не удалось загрузить историю сканирования')
  return res.json()
}

export async function fetchScanHistoryReport(params: {
  id_from: number
  id_to: number
}): Promise<ScanHistoryReportResponse> {
  const qs = buildQuery({
    id_from: params.id_from,
    id_to: params.id_to,
  })
  const res = await fetch(`${API_BASE}/api/scan-history/report?${qs}`)
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(detail || 'Не удалось построить отчёт')
  }
  return res.json()
}

export async function updateScanHistoryEval(
  scanId: number,
  flags: { false_positive: number; false_negative: number },
): Promise<ScanHistoryItem> {
  const res = await fetch(`${API_BASE}/api/scan-history/${scanId}/eval`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      false_positive: flags.false_positive ? 1 : 0,
      false_negative: flags.false_negative ? 1 : 0,
    }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(detail || 'Не удалось сохранить Eval')
  }
  return res.json()
}

export async function updateFindwineManualWine(
  scanId: number,
  wineId: number | null,
): Promise<{ search_photos_id: number; manual_wines_id: number | null }> {
  const res = await fetch(
    `${API_BASE}/api/findwine/${scanId}/manual-wine`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wine_id: wineId }),
    },
  )
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(detail || 'Не удалось сохранить соответствие')
  }
  return res.json()
}

export type PipelineSettings = {
  ocr_preprocess: string[]
  ocr_preprocess_options: {
    id: string
    name: string
    title: string
    enabled: boolean
  }[]
  ocr_engines: string[]
  ocr_engine_options: {
    id: string
    name: string
    title: string
    enabled: boolean
    disabled?: boolean
  }[]
  text_weights: Record<string, number>
  text_weight_options: {
    key: string
    label: string
    env: string
    value: number
  }[]
  final_weights: Record<string, number>
  final_weight_options: {
    key: string
    label: string
    env: string
    value: number
  }[]
  use_dinov3: boolean
  geometry_siglip2: boolean
  geometry_dinov3: boolean
  use_gemini_ocr: boolean
  use_openai_ocr: boolean
  use_deepseek_ocr: boolean
  use_qwen_ocr: boolean
  use_yandex_ocr: boolean
  use_google_vision_ocr: boolean
  yolo_fallback_full_image: boolean
  use_yolo: boolean
  yolo_variant: 'label' | 'bottle_label'
  bottle_min_conf: number
  use_bottle_orient: boolean
  bottle_orient_min_deg: number
  label_detect_openai: boolean
  label_detect_gemini: boolean
  embedding_device: 'cpu' | 'gpu'
  embed_cpu_use_cache: boolean
  normalize_max_side: number
  normalize_max_side_options: number[]
  hf_start_timeout_sec: number
  text_match_methods: string[]
  text_match_options: {
    id: string
    label: string
    enabled: boolean
    match?: number
    similar?: number
  }[]
  text_match_thresholds: Record<string, { match: number; similar: number }>
  final_score_method: string
  final_score_method_options: { id: string; label: string }[]
  empty_ocr_cosine_threshold: number
  xgb_dead_max: number
  compute_hsv: boolean
  use_hsv_filter: boolean
  hsv_bhattacharyya_max: number
  hsv_ignore_high_cosine: boolean
  hsv_ignore_cosine_min: number
  use_hsv_hard_reject: boolean
  hsv_hard_reject_max: number
  compute_color_delta: boolean
  final_ocr: string
  final_ocr_options: { id: string; label: string }[]
  exclusive_use_translit: boolean
  exclusive_match_spaced: boolean
}

export async function fetchPipelineSettings(): Promise<PipelineSettings> {
  const res = await fetch(`${API_BASE}/api/settings`)
  if (!res.ok) throw new Error('Не удалось загрузить настройки')
  return res.json()
}

export async function updatePipelineSettings(patch: {
  ocr_preprocess?: string[]
  ocr_engines?: string[]
  text_weights?: Record<string, number>
  final_weights?: Record<string, number>
  use_dinov3?: boolean
  geometry_siglip2?: boolean
  geometry_dinov3?: boolean
  use_gemini_ocr?: boolean
  use_openai_ocr?: boolean
  use_deepseek_ocr?: boolean
  use_qwen_ocr?: boolean
  use_yandex_ocr?: boolean
  use_google_vision_ocr?: boolean
  yolo_fallback_full_image?: boolean
  use_yolo?: boolean
  yolo_variant?: 'label' | 'bottle_label'
  bottle_min_conf?: number
  use_bottle_orient?: boolean
  bottle_orient_min_deg?: number
  label_detect_openai?: boolean
  label_detect_gemini?: boolean
  embedding_device?: 'cpu' | 'gpu'
  embed_cpu_use_cache?: boolean
  normalize_max_side?: number
  hf_start_timeout_sec?: number
  text_match_methods?: string[]
  final_score_method?: string
  text_match_thresholds?: Record<string, { match?: number; similar?: number }>
  empty_ocr_cosine_threshold?: number
  xgb_dead_max?: number
  compute_hsv?: boolean
  use_hsv_filter?: boolean
  hsv_bhattacharyya_max?: number
  hsv_ignore_high_cosine?: boolean
  hsv_ignore_cosine_min?: number
  use_hsv_hard_reject?: boolean
  hsv_hard_reject_max?: number
  compute_color_delta?: boolean
  final_ocr?: string
  exclusive_use_translit?: boolean
  exclusive_match_spaced?: boolean
}): Promise<PipelineSettings> {
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || 'Не удалось сохранить настройки')
  }
  return res.json()
}
