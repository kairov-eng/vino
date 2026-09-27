import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { useSearchParams } from 'react-router-dom'
import {
  fetchFindWineResult,
  findWine,
  makePreviewUrl,
  updateScanHistoryEval,
  updateFindwineManualWine,
  type AnalogMatchFlag,
  type FindWineCandidate,
  type FindWineResult,
  type WineAnalogItem,
  type WineAnalogs,
} from '../api/client'
import { ScanSettingsGear, ScanSettingsPopup } from '../components/ScanSettings'
import './HomePage.css'

const TIMING_LABELS: Record<string, string> = {
  save: 'Сохранение',
  normalize: 'Нормализация',
  yolo: 'YOLO',
  orient: 'Ориентация бутылки',
  openai_label_detect: 'OpenAI рамка+OCR',
  gemini_label_detect: 'Gemini рамка+OCR',
  label_detect: 'LLM рамка',
  hf_endpoint_start: 'HF endpoint start',
  hf_endpoint_start_siglip2: 'HF start SigLIP2',
  hf_endpoint_start_dinov3: 'HF start DINOv3',
  hf_endpoint_start_qwen_ocr: 'HF start Qwen OCR',
  label_crop: 'Кроп этикетки',
  embed_siglip2: 'SigLIP2',
  embed_siglip2_gpu: 'SigLIP2 GPU',
  embed_siglip2_cpu: 'SigLIP2 CPU',
  embed_siglip2_local: 'SigLIP2 local',
  embed_dinov3: 'DINOv3',
  candidates: 'Поиск top-20',
  hsv: 'HSV',
  geometry: 'Геометрия (keypoints)',
  ocr_deskew: 'OCR deskew',
  ocr_A_preprocess: 'OCR A prep',
  ocr_B_preprocess: 'OCR B prep',
  ocr_C_preprocess: 'OCR C prep',
  ocr_D_preprocess: 'OCR D prep',
  ocr: 'OCR всего',
  ocr_wine_id: 'OCR→wine_id',
  ocr_gemini: 'Gemini OCR',
  ocr_openai: 'OpenAI OCR',
  ocr_deepseek: 'DeepSeek OCR',
  ocr_qwen: 'Qwen OCR',
  ocr_yandex: 'Yandex OCR',
  ocr_google_vision: 'Google Vision OCR',
  ocr_wait_after_geometry: 'Ожидание OCR',
  exclusive_lexicon: 'Exclusive lexicon',
  label_text_hard_reject: 'Label text reject',
  xgb_match: 'XGBoost match',
  crenc_match: 'CrossEncoder',
  crenc_match_loc: 'CrossEncoder loc',
  crenc_match_srv: 'CrossEncoder Srv',
  openai_txt_match: 'OpenAI сравнение текста',
  parallel_branch: 'Параллельная ветка',
  total: 'Всего',
}

const OCR_PREPROCESS_TITLES: Record<string, string> = {
  A: 'Original',
  B: 'Upscale + CLAHE',
  C: 'Gray + Sharpen',
  D: 'Adaptive Threshold',
}

const OCR_ENGINE_TITLES: Record<string, string> = {
  rapid: 'RapidOCR',
  easy: 'EasyOCR',
  surya: 'Surya',
  tess: 'Tesseract',
}

const LABEL_BOX_SOURCE_TITLES: Record<string, string> = {
  yolo: 'YOLO',
  openai: 'OpenAI',
  gemini: 'Gemini',
  llm: 'LLM',
  full_image: 'полное фото',
}

function labelBoxEngineTitle(source: string | null | undefined): string {
  const raw = (source || '').trim()
  if (!raw) return 'рамка'
  if (raw.includes('+')) {
    return raw
      .split('+')
      .map((p) => LABEL_BOX_SOURCE_TITLES[p.trim()] || p.trim())
      .filter(Boolean)
      .join('+')
  }
  return LABEL_BOX_SOURCE_TITLES[raw] || raw
}

/** Подпись первой картинки: каким движком получена рамка. */
function cropsFigcaption(result: FindWineResult): string {
  const steps = (result.status?.steps || {}) as Record<string, any>
  const crop = steps.label_crop || {}
  const source = String(
    crop.source || result.status?.label_box_source || '',
  )
  const fallback = Boolean(crop.fallback) || source === 'full_image'

  if (fallback) {
    const tried: string[] = []
    if (steps.yolo) tried.push('YOLO')
    const engines = steps.label_detect?.engines
    if (Array.isArray(engines)) {
      for (const e of engines) {
        const t = labelBoxEngineTitle(String(e))
        if (t && !tried.includes(t)) tried.push(t)
      }
    }
    const via = tried.length > 0 ? tried.join('+') : labelBoxEngineTitle(source)
    return `Рамка · ${via} → полное фото`
  }

  return `Рамка · ${labelBoxEngineTitle(source)}`
}

const OCR_LETTERS = ['A', 'B', 'C', 'D'] as const
type OcrLetter = (typeof OCR_LETTERS)[number]
type OcrLlmEngine =
  | 'gemini'
  | 'openai'
  | 'deepseek'
  | 'qwen'
  | 'yandex'
  | 'google_vision'
type OcrMarkId = OcrLetter | OcrLlmEngine

const OCR_MARK_COLORS: Record<OcrMarkId, string> = {
  A: '#1f9d6a',
  B: '#2f6fdb',
  C: '#d97706',
  D: '#7c3aed',
  gemini: '#0f766e',
  openai: '#7c3aed',
  deepseek: '#0369a1',
  qwen: '#b45309',
  yandex: '#dc2626',
  google_vision: '#4285F4',
}

const OCR_MARK_LABELS: Record<OcrMarkId, string> = {
  A: 'A',
  B: 'B',
  C: 'C',
  D: 'D',
  gemini: 'G',
  openai: 'O',
  deepseek: 'K',
  qwen: 'Q',
  yandex: 'Y',
  google_vision: 'V',
}

/** @deprecated — use OCR_MARK_COLORS */
const OCR_COLORS = OCR_MARK_COLORS

type OcrScoreChannel = OcrMarkId

type OcrGroupBest = {
  letter: OcrLetter
  title: string
  bestKey: string | null
  bestScore: number | null
  wineId: number | null
}

type OcrFinalBest = {
  mark: OcrMarkId
  wineId: number
  score: number
}

type OcrWineIdHelp = {
  best_variant_for_visual_support?: string
  best_visual_support_score?: number
  best_wine_id?: number | null
  best_final?: {
    id?: number
    score?: number
    score_01?: number
    cosine?: number | null
    final_score?: number
    name?: string
    text_from?: string
  } | null
  final_ranked?: Array<{
    id: number
    score?: number
    cosine?: number | null
    final_score?: number
    final_score2?: number
    name?: string
  }>
  weights?: Record<string, number>
  explain?: {
    formula?: string
    final_formula?: string
    notes?: string[]
    weights?: Record<string, number>
    final_weights?: Record<string, number>
    soft_tfidf?: {
      formula?: string
      final_formula?: string
      veto?: string
      exclusive?: string
      channels?: string[]
    }
  }
  ensemble?: {
    entities?: Record<string, unknown>
    best_visual?: { id: number; score: number; name?: string; parts?: Record<string, number> } | null
    best_final?: {
      id: number
      score?: number
      final_score?: number
      name?: string
    } | null
    top_wine_ids?: number[]
  }
  per_variant?: Record<
    string,
    {
      top_wine_ids?: number[]
      entities?: Record<string, unknown>
      best_visual?: {
        id: number
        score: number
        name?: string
        parts?: Record<string, number>
      } | null
      best_final?: {
        id: number
        score?: number
        final_score?: number
        name?: string
      } | null
      visual_support?: {
        id: number
        score: number
        final_score?: number
      }[]
      visual_top1_ocr_rank?: number | null
      visual_top1_support_score?: number | null
    }
  >
}

function ocrVariantTitle(key: string, name?: string): string {
  if (key === 'gemini') return name || 'Gemini · original'
  if (key === 'openai') return name || 'OpenAI · original'
  if (key === 'deepseek') return name || 'DeepSeek · original'
  if (key === 'qwen') return name || 'Qwen · original'
  if (key === 'yandex') return name || 'Yandex · original'
  if (key === 'google_vision') return name || 'Google Vision · original'
  if (name) return name
  const m = key.match(/^([A-D])_(rapid|easy|surya|tess)$/)
  if (m) {
    return `${m[1]} · ${OCR_PREPROCESS_TITLES[m[1]] || m[1]} · ${OCR_ENGINE_TITLES[m[2]] || m[2]}`
  }
  return OCR_PREPROCESS_TITLES[key] ? `${key} · ${OCR_PREPROCESS_TITLES[key]}` : key
}

/** Best OCR engine card per preprocess letter A–D (FinalScore). */
function collectOcrGroupBests(result: FindWineResult | null): OcrGroupBest[] {
  if (!result) return []
  const ocr = result.status?.steps?.ocr as
    | { variants?: Record<string, { alias_of?: string }> }
    | undefined
  const help = result.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined
  if (!ocr?.variants) return []

  const keys = Object.keys(ocr.variants)
    .filter((k) => {
      const v = ocr.variants?.[k]
      return v && !v.alias_of
    })
    .sort((a, b) => a.localeCompare(b))

  return OCR_LETTERS.map((letter) => {
    const groupKeys = keys.filter((k) => k === letter || k.startsWith(`${letter}_`))
    let bestKey: string | null = null
    let bestScore = -1
    let wineId: number | null = null
    for (const k of groupKeys) {
      const bf = help?.per_variant?.[k]?.best_final
      const sc = Number(bf?.final_score ?? -1)
      if (sc > bestScore && bf?.id != null) {
        bestScore = sc
        bestKey = k
        wineId = Number(bf.id) || null
      }
    }
    return {
      letter,
      title: OCR_PREPROCESS_TITLES[letter] || letter,
      bestKey: bestScore > 0 ? bestKey : null,
      bestScore: bestScore > 0 ? bestScore : null,
      wineId: bestScore > 0 ? wineId : null,
    }
  }).filter((g) => keys.some((k) => k === g.letter || k.startsWith(`${g.letter}_`)))
}

/**
 * Победители FinalScore (embedding+OCR) по каждому типу OCR —
 * для цветных рамок карточек.
 */
function collectOcrFinalBests(result: FindWineResult | null): OcrFinalBest[] {
  const help = result?.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined
  const per = help?.per_variant
  if (!per) return []
  const out: OcrFinalBest[] = []

  for (const g of collectOcrGroupBests(result)) {
    if (g.wineId && g.bestScore != null && g.bestScore > 0) {
      out.push({ mark: g.letter, wineId: g.wineId, score: g.bestScore })
    }
  }

  for (const eng of [
    'gemini',
    'openai',
    'deepseek',
    'qwen',
    'yandex',
    'google_vision',
  ] as const) {
    const bf = per[eng]?.best_final
    const sc = Number(bf?.final_score ?? 0)
    if (bf?.id != null && sc > 0) {
      out.push({ mark: eng, wineId: Number(bf.id), score: sc })
    }
  }
  return out
}

/** wine id → OCR marks that picked it as FinalScore best. */
function ocrBestWineMarks(bests: OcrFinalBest[]): Map<number, OcrMarkId[]> {
  const map = new Map<number, OcrMarkId[]>()
  for (const g of bests) {
    if (!g.wineId) continue
    const list = map.get(g.wineId) || []
    if (!list.includes(g.mark)) list.push(g.mark)
    map.set(g.wineId, list)
  }
  return map
}

/**
 * Ранги 1..5 по убыванию score выбранного final_score_method
 * (только при определённом победителе, score > 0).
 */
function collectFinalistRanks(
  result: FindWineResult | null,
): Map<number, number> {
  const ranks = new Map<number, number>()
  if (!result) return ranks
  const winnerRaw =
    result.status?.matched_wine_id ?? result.matched_wine_id ?? null
  const winnerId =
    typeof winnerRaw === 'number'
      ? winnerRaw
      : Number.isFinite(Number(winnerRaw))
        ? Number(winnerRaw)
        : null
  if (winnerId == null || winnerId <= 0) return ranks

  const matched = result.status?.matched_wine as
    | { source?: string }
    | undefined
  const source = String(matched?.source || '')
  const method = String(
    (result.status?.final_score_method as string | undefined) || 'fin1',
  )

  const byId = new Map<number, number>()
  const bump = (id: unknown, raw: unknown) => {
    const wid = Number(id)
    const sc = Number(raw)
    if (!Number.isFinite(wid) || wid <= 0 || !Number.isFinite(sc) || sc <= 0) {
      return
    }
    const prev = byId.get(wid)
    if (prev == null || sc > prev) byId.set(wid, sc)
  }

  if (source === 'empty_ocr_cosine') {
    for (const w of [
      ...(result.candidates_siglip2 || []),
      ...(result.candidates_dinov3 || []),
    ]) {
      bump(w.id, w.cosine_similarity)
    }
  } else if (method === 'fin1' || method === 'fin2') {
    const help = result.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined
    const scoreKey = method === 'fin2' ? 'final_score2' : 'final_score'
    for (const row of help?.final_ranked || []) {
      if (!row || typeof row !== 'object') continue
      bump(row.id, (row as Record<string, unknown>)[scoreKey])
    }
  } else if (method === 'xgb') {
    const stepById = result.status?.steps?.xgb_match?.by_id
    if (stepById && typeof stepById === 'object') {
      for (const [k, v] of Object.entries(stepById as Record<string, any>)) {
        bump(k, v?.xgb_fin)
      }
    }
    for (const w of [
      ...(result.candidates_siglip2 || []),
      ...(result.candidates_dinov3 || []),
    ]) {
      bump(w.id, w.xgb_fin)
    }
  } else if (method === 'crenc' || method === 'crenc_srv') {
    const stepById = result.status?.steps?.crenc_match?.by_id
    if (stepById && typeof stepById === 'object') {
      for (const [k, v] of Object.entries(stepById as Record<string, any>)) {
        bump(k, v?.crenc_fin)
      }
    }
    for (const w of [
      ...(result.candidates_siglip2 || []),
      ...(result.candidates_dinov3 || []),
    ]) {
      bump(w.id, w.crenc_fin)
    }
  } else {
    return ranks
  }

  const sorted = [...byId.entries()].sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1]
    return a[0] - b[0]
  })
  sorted.slice(0, 5).forEach(([id], i) => {
    ranks.set(id, i + 1)
  })
  return ranks
}

function ocrChannelFromEngineKey(key: string): OcrScoreChannel | null {
  if (key === 'A' || key === 'B' || key === 'C' || key === 'D') return key
  if (
    key === 'gemini' ||
    key === 'openai' ||
    key === 'deepseek' ||
    key === 'qwen' ||
    key === 'yandex' ||
    key === 'google_vision'
  ) {
    return key
  }
  const m = key.match(/^([A-D])(?:_|$)/)
  return m ? (m[1] as OcrLetter) : null
}

/** wine id → FinalScore 0..1 per OCR channel (выбранный text method). */
function collectOcrScoresByWine(
  result: FindWineResult | null,
): Map<number, Partial<Record<OcrScoreChannel, number>>> {
  const map = new Map<number, Partial<Record<OcrScoreChannel, number>>>()
  const to01 = (n: number): number => (Math.abs(n) > 1.5 ? n / 100 : n)

  const bump = (wid: number, channel: OcrScoreChannel, sc: number) => {
    if (!wid || !Number.isFinite(sc)) return
    const cur = map.get(wid) || {}
    const prev = cur[channel]
    const v = to01(sc)
    if (prev == null || v > prev) cur[channel] = v
    map.set(wid, cur)
  }

  const fbo = result?.status?.steps?.final_by_ocr as
    | {
        by_ocr?: Record<string, { by_id?: Record<string, number> }>
      }
    | undefined
  const byOcr = fbo?.by_ocr
  if (byOcr && typeof byOcr === 'object') {
    for (const [eng, pack] of Object.entries(byOcr)) {
      const ch = ocrChannelFromEngineKey(eng)
      if (!ch || !pack || typeof pack !== 'object') continue
      for (const [widS, raw] of Object.entries(pack.by_id || {})) {
        const sc = Number(raw)
        if (!Number.isFinite(sc)) continue
        bump(Number(widS), ch, sc)
      }
    }
  }

  for (const w of [
    ...(result?.candidates_siglip2 || []),
    ...(result?.candidates_dinov3 || []),
  ]) {
    const scores = w.final_by_ocr
    if (!scores || typeof scores !== 'object') continue
    for (const [eng, raw] of Object.entries(scores)) {
      const ch = ocrChannelFromEngineKey(eng)
      if (!ch) continue
      const sc = Number(raw)
      if (!Number.isFinite(sc)) continue
      bump(w.id, ch, sc)
    }
  }

  if (map.size > 0) return map

  const help = result?.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined
  const per = help?.per_variant
  if (!per) return map

  for (const [key, entry] of Object.entries(per)) {
    const ch = ocrChannelFromEngineKey(key)
    if (!ch) continue
    for (const row of entry.visual_support || []) {
      const fs = (row as { final_score?: number }).final_score
      const sc =
        fs != null && Number.isFinite(Number(fs))
          ? Number(fs)
          : Number(row.score) > 1.5
            ? Number(row.score) / 100
            : Number(row.score)
      bump(Number(row.id), ch, sc)
    }
  }
  return map
}

/** wine id → TextScore 0..1 per OCR channel. */
function collectOcrTextScoresByWine(
  result: FindWineResult | null,
): Map<number, Partial<Record<OcrScoreChannel, number>>> {
  const map = new Map<number, Partial<Record<OcrScoreChannel, number>>>()

  const bump = (wid: number, channel: OcrScoreChannel, sc: number) => {
    if (!wid || !Number.isFinite(sc)) return
    const cur = map.get(wid) || {}
    const prev = cur[channel]
    if (prev == null || sc > prev) cur[channel] = sc
    map.set(wid, cur)
  }

  const fbo = result?.status?.steps?.final_by_ocr as
    | {
        by_ocr?: Record<string, { text_by_id?: Record<string, number> }>
      }
    | undefined
  const byOcr = fbo?.by_ocr
  if (byOcr && typeof byOcr === 'object') {
    for (const [eng, pack] of Object.entries(byOcr)) {
      const ch = ocrChannelFromEngineKey(eng)
      if (!ch || !pack || typeof pack !== 'object') continue
      for (const [widS, raw] of Object.entries(pack.text_by_id || {})) {
        const sc = Number(raw)
        if (!Number.isFinite(sc)) continue
        bump(Number(widS), ch, sc)
      }
    }
  }
  if (map.size > 0) return map

  const help = result?.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined
  const per = help?.per_variant
  if (!per) return map

  for (const [key, entry] of Object.entries(per)) {
    const ch = ocrChannelFromEngineKey(key)
    if (!ch) continue
    for (const row of entry.visual_support || []) {
      const s01 = (row as { score_01?: number }).score_01
      const sc =
        s01 != null && Number.isFinite(Number(s01))
          ? Number(s01)
          : Number(row.score) > 1.5
            ? Number(row.score) / 100
            : Number(row.score)
      bump(Number(row.id), ch, sc)
    }
  }
  return map
}

function formatOcrScore(value: number | undefined): string {
  return formatScore00(value)
}

/** Все скоры единообразно 0.00 (0..1; значения >1.5 считаем шкалой ×100). */
function formatScore00(value: number | string | null | undefined): string {
  if (value == null || value === '') return '—'
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n)) return '—'
  const scaled = Math.abs(n) > 1.5 ? n / 100 : n
  return scaled.toFixed(2)
}

function ocrChannelLetter(ch: OcrScoreChannel): string {
  if (ch === 'gemini') return 'G'
  if (ch === 'openai') return 'O'
  if (ch === 'deepseek') return 'K'
  if (ch === 'qwen') return 'Q'
  if (ch === 'yandex') return 'Y'
  if (ch === 'google_vision') return 'V'
  return ch
}

function finalScoreMethodOf(result: FindWineResult | null): string {
  const fromStatus = result?.status?.final_score_method
  if (typeof fromStatus === 'string' && fromStatus) return fromStatus
  const fromSettings = (
    result?.status?.settings as { final_score_method?: string } | undefined
  )?.final_score_method
  if (typeof fromSettings === 'string' && fromSettings) return fromSettings
  const fromFbo = (
    result?.status?.steps?.final_by_ocr as { method?: string } | undefined
  )?.method
  if (typeof fromFbo === 'string' && fromFbo) return fromFbo
  return 'fin1'
}

function finalOcrPrimaryOf(result: FindWineResult | null): string | null {
  const resolved = result?.status?.final_ocr_resolved
  if (typeof resolved === 'string' && resolved) return resolved
  const fromFbo = (
    result?.status?.steps?.final_by_ocr as { primary?: string } | undefined
  )?.primary
  if (typeof fromFbo === 'string' && fromFbo) return fromFbo
  for (const w of [
    ...(result?.candidates_siglip2 || []),
    ...(result?.candidates_dinov3 || []),
  ]) {
    if (w.final_ocr_primary) return w.final_ocr_primary
  }
  return null
}

function finalOcrPrimaryChannel(
  primary: string | null,
): OcrScoreChannel | null {
  if (!primary) return null
  return ocrChannelFromEngineKey(primary)
}

/** Подпись ряда Fin на карточке / сортировке. */
function finalByOcrRowLabel(method: string): string {
  if (method === 'xgb') return 'XGB_fin'
  if (method === 'crenc' || method === 'crenc_srv') return 'CrEnc_fin'
  if (method === 'openai_txt') return 'LLM txt'
  if (method === 'fin2') return 'Fin2'
  return 'Fin'
}

function finalByOcrSortLabel(ch: OcrScoreChannel, method: string): string {
  return `${finalByOcrRowLabel(method)} ${ocrChannelLetter(ch)}`
}

/** Основной fin на карточке для выбранного text method (для сравнения с per-OCR). */
function mainFinScoreForMethod(
  wine: FindWineCandidate,
  method: string,
): number | null {
  const num = (v: unknown): number | null => {
    if (v == null) return null
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  if (method === 'xgb') return num(wine.xgb_fin)
  if (method === 'crenc' || method === 'crenc_srv') return num(wine.crenc_fin)
  if (method === 'openai_txt') return num(wine.llm_txt)
  return null
}

function scoresMatch00(a: number, b: number): boolean {
  const scale = (n: number) => (Math.abs(n) > 1.5 ? n / 100 : n)
  return Math.abs(scale(a) - scale(b)) < 0.005
}

/**
 * Каналы Final-по-OCR для отображения.
 * Если единственный канал и его score = основному XGB_fin/CrEnc_fin/… — скрыть
 * (иначе дубль «XGB_fin» и «XGB_fin V»).
 */
function visibleOcrFinChannels(
  wine: FindWineCandidate,
  method: string,
  channels: OcrScoreChannel[],
  scores?: Partial<Record<OcrScoreChannel, number>> | null,
): OcrScoreChannel[] {
  const scored = channels.filter((ch) => {
    const v = scores?.[ch]
    return v != null && Number.isFinite(Number(v))
  })
  if (scored.length !== 1) return scored.length > 0 ? scored : []
  const main = mainFinScoreForMethod(wine, method)
  if (main == null) return scored
  const only = Number(scores?.[scored[0]])
  if (Number.isFinite(only) && scoresMatch00(only, main)) return []
  return scored
}

/** Текст OCR запроса (искомое фото) — лучший канал по FinalScore, иначе любой доступный. */
function collectQueryOcrText(result: FindWineResult | null): string {
  if (!result) return ''
  const steps = (result.status?.steps || {}) as Record<string, any>
  const ocr = steps.ocr as
    | {
        text?: string
        text_aggregated?: string
        variants?: Record<string, { text?: string; alias_of?: string; ok?: boolean }>
      }
    | undefined
  const help = steps.ocr_wine_id as OcrWineIdHelp | undefined
  const per = help?.per_variant || {}

  const textFromKey = (key: string | null): string => {
    if (!key) return ''
    if (
      key === 'gemini' ||
      key === 'openai' ||
      key === 'deepseek' ||
      key === 'qwen' ||
      key === 'yandex' ||
      key === 'google_vision'
    ) {
      const ld = steps[`${key}_label_detect`]
      const ov = ld?.ocr_variant
      const fromLd = String(ov?.text || ld?.text || '').trim()
      if (fromLd) return fromLd
      const step = steps[`ocr_${key}`] || ocr?.variants?.[key]
      return String(step?.text || step?.full_text || '').trim()
    }
    const v = ocr?.variants?.[key]
    return String(v?.text || '').trim()
  }

  // 1) Selected Final OCR engine (auto → resolved)
  const resolved = String(
    (result.status as { final_ocr_resolved?: string | null } | undefined)
      ?.final_ocr_resolved || '',
  ).trim()
  const fromResolved = textFromKey(resolved)
  if (fromResolved) return fromResolved

  // 2) Best per-variant by final_score (when scored)
  let bestKey: string | null = null
  let bestFs = -1
  for (const [key, entry] of Object.entries(per)) {
    const raw = entry?.best_final?.final_score
    if (raw == null || raw === '') continue
    const fs = Number(raw)
    if (!Number.isFinite(fs)) continue
    if (fs > bestFs) {
      bestFs = fs
      bestKey = key
    }
  }
  const fromBest = textFromKey(bestKey)
  if (fromBest) return fromBest

  // 3) Primary single-engine text (never text_aggregated — it concatenates engines)
  const primary = String(ocr?.text || '').trim()
  if (primary) return primary

  for (const eng of [
    'google_vision',
    'gemini',
    'openai',
    'deepseek',
    'qwen',
    'yandex',
  ] as const) {
    const t = textFromKey(eng)
    if (t) return t
  }
  for (const [, v] of Object.entries(ocr?.variants || {})) {
    if (v?.alias_of) continue
    const t = String(v?.text || '').trim()
    if (t) return t
  }
  return ''
}

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

/** Общие значимые токены OCR ↔ каталог для подсветки. */
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

function highlightMatchedText(
  text: string,
  matched: Set<string>,
): ReactNode {
  if (!text) return '—'
  if (!matched.size) return text
  const parts = text.split(/([A-Za-zА-Яа-яЁё0-9]+)/)
  return parts.map((part, i) => {
    const n = normalizeMatchToken(part)
    if (n && matched.has(n)) {
      return (
        <mark key={i} className="cand-popup__match">
          {part}
        </mark>
      )
    }
    return <span key={i}>{part}</span>
  })
}

type ExclusiveKind = 'category' | 'type' | 'grape' | 'winery'

type ExclusiveLexiconStep = {
  query_hits?: {
    category?: Record<string, string[]>
    type?: Record<string, string[]>
    grape?: Record<string, string[]>
    winery?: Record<string, string[]>
  }
  rejected?: Array<{
    wine_id?: number
    name?: string | null
    conflicts?: Array<{
      lexicon?: string
      rule?: string
      missing_forms?: string[]
      query_forms?: Record<string, string[]>
      catalog_forms?: Record<string, string[]>
      query_canons?: string[]
      catalog_canons?: string[]
    }>
  }>
  lexicons?: {
    category?: { forms?: string[] }
    type?: { forms?: string[] }
    grape?: { forms?: string[] }
    winery?: { forms?: string[] }
  }
}

type HsvStepInfo = {
  hard_reject_enabled?: boolean
  hard_reject_max?: number
  hard_rejected_ids?: number[]
  rejected_ids?: number[]
}

type LabelTextHardRejectStep = {
  rejected_ids?: number[]
  reasons?: Record<string, string>
  rejected?: Array<{
    wine_id?: number
    reason?: string
    reason_title?: string
  }>
}

const LABEL_TEXT_REJECT_TITLES: Record<string, string> = {
  no_shared_words: 'нет общих слов OCR↔этикетка',
  query_text_catalog_empty: 'OCR есть, у кандидата нет текста этикетки',
  query_empty_catalog_text: 'OCR пустой, у кандидата есть текст этикетки',
}

const EXCLUSIVE_LEX_TITLES: Record<string, string> = {
  category: 'категория',
  type: 'тип',
  grape: 'сорт',
  winery: 'винодельня',
}

function flatFormMap(forms: Record<string, string[]> | undefined): string[] {
  if (!forms) return []
  const out: string[] = []
  for (const arr of Object.values(forms)) {
    for (const f of arr || []) {
      const t = String(f || '').trim()
      if (t && !out.includes(t)) out.push(t)
    }
  }
  return out
}

/** Human-readable hard-reject reasons for a wine (exclusive + HSV + label text). */
function hardRejectReasonsForWine(
  wineId: number,
  exclusive: ExclusiveLexiconStep | null | undefined,
  hsv: HsvStepInfo | null | undefined,
  wineHsv?: number | null,
  labelText?: LabelTextHardRejectStep | null,
): string[] {
  const lines: string[] = []
  const rejected = (exclusive?.rejected || []).find(
    (r) => Number(r.wine_id) === Number(wineId),
  )
  for (const c of rejected?.conflicts || []) {
    const lexKey = String(c.lexicon || '').toLowerCase()
    const lexTitle = EXCLUSIVE_LEX_TITLES[lexKey] || lexKey || 'lexicon'
    const missing = (c.missing_forms || []).map((x) => String(x).trim()).filter(Boolean)
    const qForms = flatFormMap(c.query_forms)
    const catForms = flatFormMap(c.catalog_forms)
    const ocrBit = (qForms.length ? qForms : missing).join(', ')
    if (ocrBit) {
      const catBit = catForms.length ? ` (на этикетке: «${catForms.join(', ')}»)` : ''
      lines.push(
        `exclusive / ${lexTitle}: в OCR «${ocrBit}», нет на этикетке кандидата${catBit}`,
      )
    } else {
      lines.push(`exclusive / ${lexTitle}`)
    }
  }
  const hardIds = (hsv?.hard_rejected_ids || []).map(Number)
  if (hardIds.includes(Number(wineId))) {
    const thr = hsv?.hard_reject_max
    if (wineHsv != null && Number.isFinite(wineHsv) && thr != null) {
      lines.push(
        `HSV hard reject: ${Number(wineHsv).toFixed(2)} > ${Number(thr).toFixed(2)}`,
      )
    } else {
      lines.push('HSV hard reject')
    }
  }
  const ltRow = (labelText?.rejected || []).find(
    (r) => Number(r.wine_id) === Number(wineId),
  )
  const ltReason =
    ltRow?.reason ||
    (labelText?.reasons ? labelText.reasons[String(wineId)] : undefined)
  if (ltReason) {
    const title =
      ltRow?.reason_title ||
      LABEL_TEXT_REJECT_TITLES[String(ltReason)] ||
      String(ltReason)
    lines.push(`label text: ${title}`)
  }
  return lines
}

function normalizeExclusiveForm(raw: string): string {
  let t = normalizeCompareText(raw).replace(/[^a-zа-я0-9]+/gi, ' ')
  while (t.includes('  ')) {
    t = t.replace(/  /g, ' ')
  }
  return t.trim()
}

function lineHasExclusiveForm(line: string, form: string): boolean {
  const nForm = normalizeExclusiveForm(form)
  if (nForm.length < 3) return false
  const nLine = normalizeExclusiveForm(line)
  if (!nLine) return false
  const parts = nForm.split(' ').filter(Boolean).map((p) =>
    p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
  )
  if (!parts.length) return false
  const re = new RegExp(`(?:^|[^a-zа-я0-9])${parts.join('[\\s\\-]*')}(?![a-zа-я0-9])`, 'i')
  return re.test(nLine)
}

const EXCLUSIVE_KIND_PRIORITY: ExclusiveKind[] = [
  'winery',
  'grape',
  'type',
  'category',
]

function addExclusiveForms(
  target: Map<string, ExclusiveKind>,
  forms: string[],
  kind: ExclusiveKind,
) {
  for (const f of forms) {
    if (!f) continue
    const prev = target.get(f)
    if (!prev) {
      target.set(f, kind)
      continue
    }
    if (
      EXCLUSIVE_KIND_PRIORITY.indexOf(kind) <
      EXCLUSIVE_KIND_PRIORITY.indexOf(prev)
    ) {
      target.set(f, kind)
    }
  }
}

/** Forms + lexicon kind for exclusive line highlight. */
function collectExclusiveLineForms(
  exclusive: ExclusiveLexiconStep | null | undefined,
  wineId: number,
): { query: Map<string, ExclusiveKind>; catalog: Map<string, ExclusiveKind> } {
  const queryForms = new Map<string, ExclusiveKind>()
  const catalogForms = new Map<string, ExclusiveKind>()
  if (!exclusive) return { query: queryForms, catalog: catalogForms }

  const qh = exclusive.query_hits || {}
  for (const lex of ['category', 'type', 'grape', 'winery'] as const) {
    const hits = qh[lex] || {}
    for (const forms of Object.values(hits)) {
      addExclusiveForms(queryForms, forms || [], lex)
    }
  }

  const rejected = (exclusive.rejected || []).find(
    (r) => Number(r.wine_id) === Number(wineId),
  )
  if (rejected) {
    for (const c of rejected.conflicts || []) {
      const lex = (c.lexicon || '') as ExclusiveKind
      const kind: ExclusiveKind =
        lex === 'type' || lex === 'grape' || lex === 'winery' || lex === 'category'
          ? lex
          : 'category'
      for (const forms of Object.values(c.query_forms || {})) {
        addExclusiveForms(queryForms, forms || [], kind)
      }
      for (const forms of Object.values(c.catalog_forms || {})) {
        addExclusiveForms(catalogForms, forms || [], kind)
      }
    }
  } else {
    // Не rejected: подсветить на каталоге те же формы, что найдены в OCR
    // (правило presence — форма запроса должна быть у вина).
    for (const lex of ['category', 'type', 'grape', 'winery'] as const) {
      const hits = qh[lex] || {}
      for (const forms of Object.values(hits)) {
        addExclusiveForms(catalogForms, forms || [], lex)
      }
    }
  }

  return { query: queryForms, catalog: catalogForms }
}

function lineExclusiveKind(
  line: string,
  forms: Map<string, ExclusiveKind>,
): ExclusiveKind | null {
  if (!forms.size) return null
  let best: { kind: ExclusiveKind; len: number } | null = null
  for (const [form, kind] of forms) {
    if (!lineHasExclusiveForm(line, form)) continue
    const len = normalizeExclusiveForm(form).length
    if (
      !best ||
      len > best.len ||
      (len === best.len &&
        EXCLUSIVE_KIND_PRIORITY.indexOf(kind) <
          EXCLUSIVE_KIND_PRIORITY.indexOf(best.kind))
    ) {
      best = { kind, len }
    }
  }
  return best?.kind ?? null
}

function highlightExclusiveLines(
  text: string,
  exclusiveForms: Map<string, ExclusiveKind>,
  matchedTokens: Set<string>,
): ReactNode {
  if (!text) return '—'
  const lines = text.split('\n')
  return lines.map((line, i) => {
    const kind = lineExclusiveKind(line, exclusiveForms)
    return (
      <span
        key={i}
        className={
          kind
            ? `cand-popup__line cand-popup__line--exclusive cand-popup__line--exclusive-${kind}`
            : 'cand-popup__line'
        }
      >
        {highlightMatchedText(line || '\u00a0', matchedTokens)}
      </span>
    )
  })
}

type CandSortKey =
  | 'cos'
  | 'geo'
  | 'xgb'
  | 'xgb_fin'
  | 'crenc'
  | 'crenc_fin'
  | 'llm_txt'
  | 'ocr_A'
  | 'ocr_B'
  | 'ocr_C'
  | 'ocr_D'
  | 'ocr_gemini'
  | 'ocr_openai'
  | 'ocr_deepseek'
  | 'ocr_qwen'
  | 'ocr_yandex'
  | 'ocr_google_vision'

type ScanSettingsSnapshot = {
  ocr_preprocess?: string[]
  use_gemini_ocr?: boolean
  use_openai_ocr?: boolean
  use_deepseek_ocr?: boolean
  use_qwen_ocr?: boolean
  use_yandex_ocr?: boolean
  use_google_vision_ocr?: boolean
  use_dinov3?: boolean
  geometry_siglip2?: boolean
  geometry_dinov3?: boolean
  label_detect_openai?: boolean
  label_detect_gemini?: boolean
}

/** OCR Gemini/OpenAI/DeepSeek/Qwen/Yandex уже есть. */
function llmOcrChannelAvailable(
  result: FindWineResult | null,
  engine: OcrLlmEngine,
): boolean {
  const steps = (result?.status?.steps || {}) as Record<string, any>
  const ld = steps[`${engine}_label_detect`]
  if (ld && typeof ld === 'object') {
    const ov = ld.ocr_variant
    if (ov && typeof ov === 'object' && (ov.ok || String(ov.text || '').trim())) {
      return true
    }
    if (ld.ok && String(ld.text || '').trim()) return true
  }
  const ocrStep =
    steps[`ocr_${engine}`] ||
    (steps.ocr as { variants?: Record<string, any> } | undefined)?.variants?.[
      engine
    ]
  if (ocrStep && typeof ocrStep === 'object') {
    if (ocrStep.skipped && !ocrStep.ok) {
      // skipped без текста — канала нет
    } else if (
      ocrStep.ok ||
      String(ocrStep.text || ocrStep.full_text || '').trim()
    ) {
      return true
    } else if (ocrStep.from_label_detect && !ocrStep.skipped) {
      return true
    }
  }
  const pv = (steps.ocr_wine_id as OcrWineIdHelp | undefined)?.per_variant?.[
    engine
  ] as
    | {
        best_visual?: { id?: number | null } | null
        top_wine_ids?: number[]
        ranked?: unknown[]
      }
    | undefined
  if (pv) {
    if (pv.best_visual?.id != null) return true
    if (Array.isArray(pv.top_wine_ids) && pv.top_wine_ids.length > 0) return true
    if (Array.isArray(pv.ranked) && pv.ranked.length > 0) return true
  }
  return false
}

/** OCR-каналы, включённые/посчитанные в этом скане. */
function activeOcrScoreChannels(
  result: FindWineResult | null,
): OcrScoreChannel[] {
  const settings = (result?.status?.settings || {}) as ScanSettingsSnapshot
  const channels: OcrScoreChannel[] = []

  const prep = settings.ocr_preprocess
  if (Array.isArray(prep) && prep.length > 0) {
    for (const l of OCR_LETTERS) {
      if (prep.map(String).includes(l)) channels.push(l)
    }
  } else {
    const per =
      (result?.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined)
        ?.per_variant || {}
    const variants =
      (
        result?.status?.steps?.ocr as
          | { variants?: Record<string, { skipped?: boolean }> }
          | undefined
      )?.variants || {}
    for (const l of OCR_LETTERS) {
      const keys = [...Object.keys(per), ...Object.keys(variants)]
      if (keys.some((k) => k === l || k.startsWith(`${l}_`))) {
        channels.push(l)
      }
    }
  }

  const geminiStep = (result?.status?.steps?.ocr_gemini ||
    (
      result?.status?.steps?.ocr as
        | { variants?: { gemini?: { skipped?: boolean; ok?: boolean } } }
        | undefined
    )?.variants?.gemini) as { skipped?: boolean; ok?: boolean } | undefined

  if (
    settings.use_gemini_ocr === true ||
    llmOcrChannelAvailable(result, 'gemini') ||
    (settings.use_gemini_ocr == null && geminiStep && !geminiStep.skipped)
  ) {
    channels.push('gemini')
  }

  const openaiStep = (result?.status?.steps?.ocr_openai ||
    (
      result?.status?.steps?.ocr as
        | { variants?: { openai?: { skipped?: boolean; ok?: boolean } } }
        | undefined
    )?.variants?.openai) as { skipped?: boolean; ok?: boolean } | undefined

  if (
    settings.use_openai_ocr === true ||
    llmOcrChannelAvailable(result, 'openai') ||
    (settings.use_openai_ocr == null && openaiStep && !openaiStep.skipped)
  ) {
    channels.push('openai')
  }

  const deepseekStep = (result?.status?.steps?.ocr_deepseek ||
    (
      result?.status?.steps?.ocr as
        | { variants?: { deepseek?: { skipped?: boolean; ok?: boolean } } }
        | undefined
    )?.variants?.deepseek) as { skipped?: boolean; ok?: boolean } | undefined

  if (
    settings.use_deepseek_ocr === true ||
    llmOcrChannelAvailable(result, 'deepseek') ||
    (settings.use_deepseek_ocr == null && deepseekStep && !deepseekStep.skipped)
  ) {
    channels.push('deepseek')
  }

  const qwenStep = (result?.status?.steps?.ocr_qwen ||
    (
      result?.status?.steps?.ocr as
        | { variants?: { qwen?: { skipped?: boolean; ok?: boolean } } }
        | undefined
    )?.variants?.qwen) as { skipped?: boolean; ok?: boolean } | undefined

  if (
    settings.use_qwen_ocr === true ||
    llmOcrChannelAvailable(result, 'qwen') ||
    (settings.use_qwen_ocr == null && qwenStep && !qwenStep.skipped)
  ) {
    channels.push('qwen')
  }

  const yandexStep = (result?.status?.steps?.ocr_yandex ||
    (
      result?.status?.steps?.ocr as
        | { variants?: { yandex?: { skipped?: boolean; ok?: boolean } } }
        | undefined
    )?.variants?.yandex) as { skipped?: boolean; ok?: boolean } | undefined

  if (
    settings.use_yandex_ocr === true ||
    llmOcrChannelAvailable(result, 'yandex') ||
    (settings.use_yandex_ocr == null && yandexStep && !yandexStep.skipped)
  ) {
    channels.push('yandex')
  }

  const gvisionStep = (result?.status?.steps?.ocr_google_vision ||
    (
      result?.status?.steps?.ocr as
        | { variants?: { google_vision?: { skipped?: boolean; ok?: boolean } } }
        | undefined
    )?.variants?.google_vision) as
    | { skipped?: boolean; ok?: boolean }
    | undefined

  if (
    settings.use_google_vision_ocr === true ||
    llmOcrChannelAvailable(result, 'google_vision') ||
    (settings.use_google_vision_ocr == null &&
      gvisionStep &&
      !gvisionStep.skipped)
  ) {
    channels.push('google_vision')
  }

  return channels
}

function ocrSortKeyForChannel(ch: OcrScoreChannel): CandSortKey {
  if (ch === 'gemini') return 'ocr_gemini'
  if (ch === 'openai') return 'ocr_openai'
  if (ch === 'deepseek') return 'ocr_deepseek'
  if (ch === 'qwen') return 'ocr_qwen'
  if (ch === 'yandex') return 'ocr_yandex'
  if (ch === 'google_vision') return 'ocr_google_vision'
  return `ocr_${ch}` as CandSortKey
}

/** Кнопки: Score OCR… → cos → geo → XGB → XGB_fin → CrEnc → CrEnc_fin → LLM txt. */
function activeCandSortKeys(
  result: FindWineResult | null,
  embedding: 'siglip2' | 'dinov3',
): CandSortKey[] {
  const keys: CandSortKey[] = []
  if (!result) return ['cos']

  const ocrCh = activeOcrScoreChannels(result)
  for (const ch of ocrCh) keys.push(ocrSortKeyForChannel(ch))
  keys.push('cos')

  const settings = (result.status?.settings || {}) as ScanSettingsSnapshot
  const geoFlag =
    embedding === 'siglip2'
      ? settings.geometry_siglip2
      : settings.geometry_dinov3
  const geoStep = (
    result.status?.steps?.geometry as
      | {
          channels?: Record<
            string,
            { skipped?: boolean; ok?: boolean; reason?: string }
          >
        }
      | undefined
  )?.channels?.[embedding]

  const geoComputed =
    geoFlag === true ||
    (geoFlag == null && geoStep != null && !geoStep.skipped) ||
    (geoFlag == null &&
      geoStep == null &&
      (result[
        embedding === 'siglip2' ? 'candidates_siglip2' : 'candidates_dinov3'
      ] ||
        []).some(
        (w) => w.geometry_score != null || w.geometry_inliers != null,
      ))

  if (geoComputed) keys.push('geo')

  const xgbStep = result.status?.steps?.xgb_match as
    | { ok?: boolean; by_id?: Record<string, unknown>; scores?: unknown[] }
    | undefined
  const cands =
    result[
      embedding === 'siglip2' ? 'candidates_siglip2' : 'candidates_dinov3'
    ] || []
  const hasXgb =
    (xgbStep?.ok &&
      ((xgbStep.by_id && Object.keys(xgbStep.by_id).length > 0) ||
        (Array.isArray(xgbStep.scores) && xgbStep.scores.length > 0))) ||
    cands.some((w) => w.xgb_score != null || w.xgb_fin != null)
  if (hasXgb) {
    keys.push('xgb')
    keys.push('xgb_fin')
  }
  const crencStep = result.status?.steps?.crenc_match as
    | { ok?: boolean; by_id?: Record<string, unknown>; scores?: unknown[] }
    | undefined
  const hasCrenc =
    (crencStep?.ok &&
      ((crencStep.by_id && Object.keys(crencStep.by_id).length > 0) ||
        (Array.isArray(crencStep.scores) && crencStep.scores.length > 0))) ||
    cands.some((w) => w.crenc_score != null || w.crenc_fin != null)
  if (hasCrenc) {
    keys.push('crenc')
    keys.push('crenc_fin')
  }
  const openaiTxtStep = result.status?.steps?.openai_txt_match as
    | { ok?: boolean; by_id?: Record<string, unknown>; scores?: unknown[] }
    | undefined
  const hasLlmTxt =
    (openaiTxtStep?.ok &&
      ((openaiTxtStep.by_id && Object.keys(openaiTxtStep.by_id).length > 0) ||
        (Array.isArray(openaiTxtStep.scores) &&
          openaiTxtStep.scores.length > 0))) ||
    cands.some((w) => w.llm_txt != null)
  if (hasLlmTxt) keys.push('llm_txt')
  return keys
}

function WineHoverPopup({
  wine,
  pos,
  embeddingLabel,
  cosSiglip2,
  cosDinov3,
  ocrScores,
  ocrChannels = [],
  ocrQueryText = '',
  ocrMarks = [],
  exclusiveLexicon = null,
  hsvStep = null,
  labelTextHr = null,
  finalMethod = 'fin1',
  onMouseEnter,
  onMouseLeave,
}: {
  wine: FindWineCandidate
  pos: { top: number; left: number }
  embeddingLabel?: string
  cosSiglip2?: number | null
  cosDinov3?: number | null
  ocrScores?: Partial<Record<OcrScoreChannel, number>>
  ocrChannels?: OcrScoreChannel[]
  ocrQueryText?: string
  ocrMarks?: OcrMarkId[]
  exclusiveLexicon?: ExclusiveLexiconStep | null
  hsvStep?: HsvStepInfo | null
  labelTextHr?: LabelTextHardRejectStep | null
  finalMethod?: string
  onMouseEnter?: () => void
  onMouseLeave?: () => void
}) {
  const [labelBroken, setLabelBroken] = useState(false)
  const [bottleBroken, setBottleBroken] = useState(false)

  const channelTitle = (ch: OcrScoreChannel): string => {
    if (ch === 'gemini') return 'Gemini'
    if (ch === 'openai') return 'OpenAI'
    if (ch === 'deepseek') return 'DeepSeek'
    if (ch === 'qwen') return 'Qwen'
    if (ch === 'yandex') return 'Yandex'
    if (ch === 'google_vision') return 'Google Vision'
    return OCR_PREPROCESS_TITLES[ch] || ch
  }

  const catalogLabel = String(wine.label || '').trim()
  const queryOcr = String(ocrQueryText || '').trim()
  const matchedTokens = useMemo(
    () => sharedMatchTokens(queryOcr, catalogLabel),
    [queryOcr, catalogLabel],
  )
  const exclusiveForms = useMemo(
    () => collectExclusiveLineForms(exclusiveLexicon, wine.id),
    [exclusiveLexicon, wine.id],
  )
  const hardRejectReasons = useMemo(
    () =>
      hardRejectReasonsForWine(
        wine.id,
        exclusiveLexicon,
        hsvStep,
        wine.hsv,
        labelTextHr,
      ),
    [wine.id, wine.hsv, exclusiveLexicon, hsvStep, labelTextHr],
  )
  const showGeometry =
    wine.geometry_score != null ||
    wine.geometry_inliers != null ||
    wine.geometry_matches != null ||
    wine.geometry_inlier_ratio != null ||
    wine.geometry_ok != null

  return createPortal(
    <div
      className="cand-popup"
      role="dialog"
      aria-label={`Вино ${wine.id}`}
      style={{ top: pos.top, left: pos.left }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="cand-popup__inner">
        <div className="cand-popup__left">
          <div className="cand-popup__photos">
            <div className="cand-popup__photo">
              <span>Бутылка</span>
              {wine.photo_url && !bottleBroken ? (
                <img
                  src={wine.photo_url}
                  alt="Бутылка"
                  onError={() => setBottleBroken(true)}
                />
              ) : (
                <div className="cand-card__placeholder" />
              )}
            </div>
            <div className="cand-popup__photo">
              <span>Этикетка</span>
              {wine.label_url && !labelBroken ? (
                <img
                  src={wine.label_url}
                  alt="Этикетка"
                  onError={() => setLabelBroken(true)}
                />
              ) : (
                <div className="cand-card__placeholder" />
              )}
            </div>
          </div>
          <div className="cand-popup__texts">
            <div className="cand-popup__label-cols">
              <div className="cand-popup__label-text">
                <strong>OCR искомого</strong>
                <pre>
                  {highlightExclusiveLines(
                    queryOcr,
                    exclusiveForms.query,
                    matchedTokens,
                  )}
                </pre>
              </div>
              <div className="cand-popup__label-text">
                <strong>Текст этикетки каталога</strong>
                <pre>
                  {highlightExclusiveLines(
                    catalogLabel,
                    exclusiveForms.catalog,
                    matchedTokens,
                  )}
                </pre>
              </div>
            </div>
            <div className="scan-winner__legend" aria-hidden>
              <span className="is-ex-category">категория</span>
              <span className="is-ex-type">тип</span>
              <span className="is-ex-grape">сорт</span>
              <span className="is-ex-winery">винодельня</span>
            </div>
          </div>
        </div>
        <div className="cand-popup__info">
          <p>
            <strong>ID:</strong> {wine.id}
          </p>
          <p>
            <strong>Название:</strong> {wine.name || '—'}
          </p>
          <p>
            <strong>Винодельня:</strong> {wine.winery || '—'}
          </p>

          {hardRejectReasons.length > 0 && (
            <div className="cand-popup__hard-reject" role="status">
              <strong>Hard reject</strong>
              <ul>
                {hardRejectReasons.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}

          {wine.hsv != null && (
          <div className="cand-popup__scores">
            <strong>HSV</strong>
            <ul>
              <li title="Расстояние Бхаттачарии между HSV-гистограммами. 0 — одна гамма, 1 — гаммы не пересекаются.">
                <span>HSV</span>
                <em>{formatScore00(wine.hsv)}</em>
              </li>
            </ul>
          </div>
          )}

          {wine.color_delta != null && (
          <div className="cand-popup__scores">
            <strong>ColorDelta</strong>
            <ul>
              <li title="CIEDE2000 доминантных цветов искомой этикетки и кандидата. Меньше — ближе по цвету.">
                <span>ColorDelta</span>
                <em>{Number(wine.color_delta).toFixed(2)}</em>
              </li>
            </ul>
          </div>
          )}

          <div className="cand-popup__scores">
            <strong>Косинусное сходство</strong>
            <ul>
              <li className={embeddingLabel === 'SigLIP2' ? 'is-current' : undefined}>
                <span>SigLIP2</span>
                <em>{formatScore00(cosSiglip2)}</em>
              </li>
              <li className={embeddingLabel === 'DINOv3' ? 'is-current' : undefined}>
                <span>DINOv3</span>
                <em>{formatScore00(cosDinov3)}</em>
              </li>
            </ul>
          </div>

          {showGeometry && (
          <div className="cand-popup__scores">
            <strong>Геометрия (keypoints + RANSAC)</strong>
            <ul>
              <li>
                <span>Инлайеры</span>
                <em>{wine.geometry_inliers ?? '—'}</em>
              </li>
              <li>
                <span>Совпадения</span>
                <em>{wine.geometry_matches ?? '—'}</em>
              </li>
              <li>
                <span>Доля инлайеров</span>
                <em>
                  {wine.geometry_inlier_ratio != null
                    ? formatScore00(wine.geometry_inlier_ratio)
                    : '—'}
                </em>
              </li>
              <li>
                <span>Score</span>
                <em>
                  {wine.geometry_score != null
                    ? formatScore00(wine.geometry_score)
                    : '—'}
                </em>
              </li>
            </ul>
          </div>
          )}

          {(wine.xgb_score != null || wine.xgb_fin != null) && (
          <div className="cand-popup__scores">
            <strong>XGBoost · OCR↔label</strong>
            <ul>
              <li>
                <span>XGB</span>
                <em>{formatScore00(wine.xgb_score)}</em>
              </li>
              <li>
                <span>XGB_fin</span>
                <em>{formatScore00(wine.xgb_fin)}</em>
                {(wine.exclusive_rejected || wine.label_text_hard_reject) &&
                  wine.xgb_fin_pre_reject != null && (
                    <span className="cand-popup__xgb-fin-ghost">
                      {' '}
                      ({formatScore00(wine.xgb_fin_pre_reject)})
                    </span>
                  )}
              </li>
            </ul>
          </div>
          )}

          {(wine.crenc_score != null || wine.crenc_fin != null) && (
          <div className="cand-popup__scores">
            <strong>Cross Encoder · OCR↔label</strong>
            <ul>
              <li>
                <span>CrEnc</span>
                <em>{formatScore00(wine.crenc_score)}</em>
              </li>
              <li>
                <span>CrEnc_fin</span>
                <em>{formatScore00(wine.crenc_fin)}</em>
              </li>
            </ul>
          </div>
          )}

          {wine.llm_txt != null && (
          <div className="cand-popup__scores">
            <strong>OpenAI сравнение текста</strong>
            <ul>
              <li>
                <span>LLM txt</span>
                <em>{formatScore00(wine.llm_txt)}</em>
              </li>
            </ul>
          </div>
          )}

          {(() => {
            const finCh = visibleOcrFinChannels(
              wine,
              finalMethod,
              ocrChannels,
              ocrScores,
            )
            if (!finCh.length) return null
            return (
              <div className="cand-popup__scores">
                <strong>OCR · Final (метод сравнения текстов)</strong>
                <ul>
                  {finCh.map((ch) => (
                    <li key={`s-${ch}`} className={`is-ocr-${ch}`}>
                      <span>
                        <i
                          className={`cand-popup__ocr-letter is-ocr-${ch}${
                            ocrMarks.includes(ch) ? ' is-best-channel' : ''
                          }`}
                        >
                          {ocrChannelLetter(ch)}
                        </i>
                        {channelTitle(ch)}
                      </span>
                      <em>{formatScore00(ocrScores?.[ch])}</em>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })()}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function placeNearAnchor(el: HTMLElement): { top: number; left: number } {
  const rect = el.getBoundingClientRect()
  const gap = 12
  const popupW = Math.min(window.innerWidth * 0.72, 920)
  const popupH = Math.min(window.innerHeight * 0.78, 640)

  let left = rect.right + gap
  if (left + popupW > window.innerWidth - 8) {
    left = rect.left - gap - popupW
  }
  if (left < 8) left = 8

  let top = rect.top
  if (top + popupH > window.innerHeight - 8) {
    top = window.innerHeight - popupH - 8
  }
  if (top < 8) top = 8
  return { top, left }
}

function WineIdHover({
  wineId,
  wineById,
  cosByWine,
  ocrScoresByWine,
  ocrQueryText = '',
  exclusiveLexicon = null,
  hsvStep = null,
  labelTextHr = null,
  label,
}: {
  wineId: number
  wineById: Map<number, FindWineCandidate>
  cosByWine: {
    siglip2: Map<number, number>
    dinov3: Map<number, number>
  }
  ocrScoresByWine: Map<number, Partial<Record<OcrScoreChannel, number>>>
  ocrQueryText?: string
  exclusiveLexicon?: ExclusiveLexiconStep | null
  hsvStep?: HsvStepInfo | null
  labelTextHr?: LabelTextHardRejectStep | null
  label?: string
}) {
  const anchorRef = useRef<HTMLSpanElement>(null)
  const [hovered, setHovered] = useState(false)
  const [popupPos, setPopupPos] = useState<{ top: number; left: number } | null>(
    null,
  )
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wine = wineById.get(wineId)
  const ocrScores = ocrScoresByWine.get(wineId)
  const ocrChannels = (ocrScores
    ? (Object.keys(ocrScores) as OcrScoreChannel[])
    : []) as OcrScoreChannel[]

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
    const el = anchorRef.current
    if (!el) return
    setPopupPos(placeNearAnchor(el))
  }, [])

  useLayoutEffect(() => {
    if (!hovered || !wine) {
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
  }, [hovered, wine, placePopup])

  if (!wine) {
    return <span className="ocr-wine-id is-missing">{label ?? String(wineId)}</span>
  }

  return (
    <>
      <span
        ref={anchorRef}
        className="ocr-wine-id"
        onMouseEnter={openHover}
        onMouseLeave={closeHoverSoon}
      >
        {label ?? String(wineId)}
      </span>
      {hovered && popupPos && (
        <WineHoverPopup
          wine={wine}
          pos={popupPos}
          cosSiglip2={cosByWine.siglip2.get(wineId) ?? null}
          cosDinov3={cosByWine.dinov3.get(wineId) ?? null}
          ocrScores={ocrScores}
          ocrChannels={ocrChannels}
          ocrQueryText={ocrQueryText}
          exclusiveLexicon={exclusiveLexicon}
          hsvStep={hsvStep}
          labelTextHr={labelTextHr}
          onMouseEnter={openHover}
          onMouseLeave={closeHoverSoon}
        />
      )}
    </>
  )
}

function timingLabel(
  key: string,
  embedDevices?: { siglip2?: string | null; dinov3?: string | null },
): string {
  if (key === 'embed_siglip2_gpu') return 'SigLIP2 GPU'
  if (key === 'embed_siglip2_cpu') return 'SigLIP2 CPU'
  if (key === 'embed_siglip2_local') return 'SigLIP2 local'
  if (key === 'embed_siglip2') {
    const d = String(embedDevices?.siglip2 || '').toUpperCase()
    if (d === 'LOCAL') return 'SigLIP2 local'
    return d === 'CPU' || d === 'GPU' ? `SigLIP2 ${d}` : 'SigLIP2'
  }
  if (key === 'embed_dinov3') {
    const d = String(embedDevices?.dinov3 || '').toUpperCase()
    return d === 'CPU' || d === 'GPU' ? `DINOv3 ${d}` : 'DINOv3'
  }
  if (TIMING_LABELS[key]) return TIMING_LABELS[key]
  const m = key.match(/^ocr_([A-D])_(rapid|easy|surya|tess)$/)
  if (m) return `OCR ${m[1]}/${OCR_ENGINE_TITLES[m[2]] || m[2]}`
  const prep = key.match(/^ocr_([A-D])_preprocess$/)
  if (prep) return `OCR ${prep[1]} prep`
  return key
}

function formatMs(ms: number): string {
  return `${Math.round(ms)} мс`
}

/** URL артефакта поиска + cache-bust (после reload бэкенда браузер мог закешировать обрыв). */
function searchPhotoSrc(
  url: string | null | undefined,
  bust?: string | number | null,
): string | undefined {
  if (!url) return undefined
  if (bust == null || bust === '') return url
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}v=${encodeURIComponent(String(bust))}`
}

function mimeLabel(mime: string, src?: string | null): string {
  const m = (mime || '').toLowerCase()
  if (m.includes('jpeg') || m.includes('jpg')) return 'JPEG'
  if (m.includes('png')) return 'PNG'
  if (m.includes('webp')) return 'WebP'
  if (m.includes('heic') || m.includes('heif')) return 'HEIC'
  if (m.includes('gif')) return 'GIF'
  const path = String(src || '').split('?')[0].toLowerCase()
  if (/\.jpe?g$/i.test(path)) return 'JPEG'
  if (/\.png$/i.test(path)) return 'PNG'
  if (/\.webp$/i.test(path)) return 'WebP'
  if (/\.heic$/i.test(path) || /\.heif$/i.test(path)) return 'HEIC'
  if (/\.gif$/i.test(path)) return 'GIF'
  return m.replace(/^image\//, '').toUpperCase() || '—'
}

function formatBytesKb(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  const kb = bytes / 1024
  if (kb < 10) return `${kb.toFixed(1)} KB`
  return `${Math.round(kb)} KB`
}

function formatImageFileMeta(
  width: number,
  height: number,
  bytes: number,
  mime: string,
  src?: string | null,
): string {
  const px =
    width > 0 && height > 0 ? `${width}×${height} px` : '— px'
  return `${px} · ${formatBytesKb(bytes)} · ${mimeLabel(mime, src)}`
}

/** Размер в px / KB / тип — без имени файла. */
function ImageFileMeta({
  src,
  file,
}: {
  src?: string | null
  file?: File | null
}) {
  const [text, setText] = useState('')
  useEffect(() => {
    let cancelled = false
    const run = async () => {
      try {
        let width = 0
        let height = 0
        let bytes = 0
        let mime = ''
        // File — только пока в дропзоне blob/data превью исходника; после поиска src = work URL
        const useFile =
          Boolean(file) &&
          (!src || src.startsWith('blob:') || src.startsWith('data:'))
        if (useFile && file) {
          bytes = file.size
          mime = file.type || ''
          const bmp = await createImageBitmap(file)
          width = bmp.width
          height = bmp.height
          bmp.close()
        } else if (src) {
          const res = await fetch(src)
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          const blob = await res.blob()
          bytes = blob.size
          mime = blob.type || ''
          const bmp = await createImageBitmap(blob)
          width = bmp.width
          height = bmp.height
          bmp.close()
        } else {
          if (!cancelled) setText('')
          return
        }
        if (!cancelled) {
          setText(formatImageFileMeta(width, height, bytes, mime, src))
        }
      } catch {
        if (!cancelled) setText('')
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [src, file])
  if (!text) return null
  return <div className="image-file-meta">{text}</div>
}

function SearchArtifactImg({
  src,
  alt,
  className,
}: {
  src: string
  alt: string
  className?: string
}) {
  const [tryN, setTryN] = useState(0)
  const displaySrc =
    tryN === 0 ? src : `${src}${src.includes('?') ? '&' : '?'}retry=${tryN}`
  return (
    <img
      src={displaySrc}
      alt={alt}
      className={className}
      loading="eager"
      decoding="async"
      onError={() => {
        if (tryN < 2) {
          window.setTimeout(() => setTryN((n) => n + 1), 400 * (tryN + 1))
        }
      }}
    />
  )
}

/** Миниатюра кропа/этикетки: при наведении — увеличенное фото (~2/3 высоты экрана). */
function DropzoneThumb({
  src,
  caption,
  alt,
}: {
  src: string
  caption: string
  alt: string
}) {
  const figRef = useRef<HTMLFigureElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearLeave = useCallback(() => {
    if (leaveTimer.current) {
      clearTimeout(leaveTimer.current)
      leaveTimer.current = null
    }
  }, [])

  const place = useCallback(() => {
    const el = figRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const popupH = window.innerHeight * (2 / 3)
    const popupW = Math.min(window.innerWidth * 0.72, popupH * 1.35)
    let left = rect.right + 12
    if (left + popupW > window.innerWidth - 12) {
      left = Math.max(12, rect.left - popupW - 12)
    }
    let top = rect.top + rect.height / 2 - popupH / 2
    if (top < 12) top = 12
    if (top + popupH > window.innerHeight - 12) {
      top = Math.max(12, window.innerHeight - popupH - 12)
    }
    setPos({ top, left })
  }, [])

  const show = useCallback(() => {
    clearLeave()
    place()
    setOpen(true)
  }, [clearLeave, place])

  const hideSoon = useCallback(() => {
    clearLeave()
    leaveTimer.current = setTimeout(() => setOpen(false), 180)
  }, [clearLeave])

  useEffect(() => () => clearLeave(), [clearLeave])

  useLayoutEffect(() => {
    if (!open) {
      setPos(null)
      return
    }
    place()
    const onMove = () => place()
    window.addEventListener('scroll', onMove, true)
    window.addEventListener('resize', onMove)
    return () => {
      window.removeEventListener('scroll', onMove, true)
      window.removeEventListener('resize', onMove)
    }
  }, [open, place])

  return (
    <>
      <figure
        ref={figRef}
        title={caption}
        onMouseEnter={show}
        onMouseLeave={hideSoon}
      >
        <figcaption>{caption}</figcaption>
        <SearchArtifactImg src={src} alt={alt} />
      </figure>
      {open &&
        pos &&
        createPortal(
          <div
            className="dropzone-thumb-popup"
            style={{ top: pos.top, left: pos.left }}
            onMouseEnter={show}
            onMouseLeave={hideSoon}
            role="dialog"
            aria-label={caption}
          >
            <p className="dropzone-thumb-popup__cap">{caption}</p>
            <img src={src} alt={alt} />
          </div>,
          document.body,
        )}
    </>
  )
}

function formatPct(ms: number, total: number): string {
  if (!total || total <= 0) return '—'
  return `${((ms / total) * 100).toFixed(1)}%`
}

/** Keys that are summary wrappers — not drawn as Gantt bars. */
const GANTT_HIDE = new Set(['total', 'parallel_branch'])

type GanttBar = {
  key: string
  label: string
  start: number
  duration: number
  color: string
}

function timingBarColor(key: string): string {
  if (key.startsWith('embed_')) return '#2f6fdb'
  if (key.startsWith('ocr_') && key.includes('gemini')) return '#c45c26'
  if (key.startsWith('ocr_') && key.includes('openai')) return '#0d9488'
  if (key.startsWith('ocr_') && key.includes('deepseek')) return '#0369a1'
  if (key.startsWith('ocr_') && key.includes('qwen')) return '#b45309'
  if (key.startsWith('ocr_') && key.includes('yandex')) return '#dc2626'
  if (key.startsWith('ocr_') && key.includes('google_vision')) return '#4285F4'
  if (key.startsWith('ocr_')) return '#7c3aed'
  if (key === 'geometry' || key === 'candidates') return '#1f9d6a'
  if (key === 'hsv') return '#be185d'
  if (key === 'openai_txt_match') return '#0d9488'
  if (
    key === 'yolo' ||
    key === 'orient' ||
    key === 'label_crop' ||
    key === 'hf_endpoint_start' ||
    key.startsWith('hf_endpoint_start_')
  )
    return '#d97706'
  return '#8b5e4b'
}

/** Build Gantt rows from timings_ms + timings_start_ms (with sequential fallback). */
function buildGanttBars(
  timings: Record<string, unknown> | null | undefined,
  starts: Record<string, unknown> | null | undefined,
  embedDevices?: { siglip2?: string | null; dinov3?: string | null },
): GanttBar[] {
  if (!timings) return []
  const durMap = new Map<string, number>()
  for (const [k, v] of Object.entries(timings)) {
    if (GANTT_HIDE.has(k)) continue
    const n = Number(v)
    if (!Number.isFinite(n) || n <= 0) continue
    durMap.set(k, n)
  }
  if (durMap.size === 0) return []

  const startMap = new Map<string, number>()
  let hasStarts = false
  if (starts && typeof starts === 'object') {
    for (const [k, v] of Object.entries(starts)) {
      if (!durMap.has(k)) continue
      const n = Number(v)
      if (!Number.isFinite(n) || n < 0) continue
      startMap.set(k, n)
      hasStarts = true
    }
  }

  // Fallback for old scans: approximate sequential then parallel cluster
  if (!hasStarts || startMap.size < Math.min(3, durMap.size)) {
    startMap.clear()
    // HF start идёт параллельно с embed/OCR — не ставить в sequential
    // (иначе Gantt ложно показывает CPU после окончания HF start).
    const sequential = [
      'save',
      'normalize',
      'yolo',
      'orient',
      'openai_label_detect',
      'gemini_label_detect',
      'label_detect',
      'label_crop',
    ]
    let cursor = 0
    for (const k of sequential) {
      if (!durMap.has(k)) continue
      startMap.set(k, cursor)
      cursor += durMap.get(k)!
    }
    const parallelStart = cursor
    for (const k of durMap.keys()) {
      if (startMap.has(k)) continue
      if (
        k.startsWith('embed_') ||
        k.startsWith('ocr_') ||
        k === 'ocr' ||
        k === 'ocr_deskew' ||
        k === 'ocr_gemini' ||
        k === 'ocr_openai' ||
        k === 'ocr_deepseek' ||
        k === 'ocr_qwen' ||
        k === 'hf_endpoint_start' ||
        k.startsWith('hf_endpoint_start_')
      ) {
        startMap.set(k, parallelStart)
      }
    }
    let afterParallel = parallelStart
    for (const k of durMap.keys()) {
      if (startMap.has(k)) {
        afterParallel = Math.max(
          afterParallel,
          startMap.get(k)! + durMap.get(k)!,
        )
      }
    }
    for (const k of ['candidates', 'geometry', 'ocr_wait_after_geometry', 'ocr_wine_id']) {
      if (!durMap.has(k) || startMap.has(k)) continue
      if (k === 'candidates') {
        // after embeds roughly
        const embEnd = Math.max(
          parallelStart,
          ...[
            'embed_siglip2',
            'embed_siglip2_gpu',
            'embed_siglip2_cpu',
            'embed_siglip2_local',
            'embed_dinov3',
          ]
            .filter((x) => startMap.has(x))
            .map((x) => startMap.get(x)! + durMap.get(x)!),
        )
        startMap.set(k, embEnd)
      } else if (k === 'geometry' || k === 'hsv') {
        const candStart = startMap.get('candidates') ?? afterParallel
        const candDur = durMap.get('candidates') ?? 0
        startMap.set(k, candStart + candDur)
      } else {
        startMap.set(k, afterParallel)
        afterParallel += durMap.get(k)!
      }
    }
    for (const k of durMap.keys()) {
      if (!startMap.has(k)) startMap.set(k, cursor)
    }
  }

  const bars: GanttBar[] = []
  for (const [key, duration] of durMap) {
    bars.push({
      key,
      label: timingLabel(key, embedDevices),
      start: startMap.get(key) ?? 0,
      duration,
      color: timingBarColor(key),
    })
  }
  bars.sort((a, b) => a.start - b.start || b.duration - a.duration)
  return bars
}

function TimingsGantt({
  timings,
  starts,
  totalMs,
  algoVersion,
  algoUpdated,
  embedDevices,
  extraRows = [],
}: {
  timings: Record<string, unknown>
  starts?: Record<string, unknown> | null
  totalMs: number
  algoVersion: string | null
  algoUpdated: string | null
  embedDevices?: { siglip2?: string | null; dinov3?: string | null }
  /** Шаги вне общего timeline (не входят в «Всего») */
  extraRows?: { key: string; label: string; ms: number }[]
}) {
  const bars = useMemo(
    () => buildGanttBars(timings, starts, embedDevices),
    [timings, starts, embedDevices],
  )
  const axisEnd = Math.max(
    totalMs || 0,
    ...bars.map((b) => b.start + b.duration),
    1,
  )
  const hasRealStarts = Boolean(
    starts && Object.keys(starts).length > 0,
  )

  return (
    <div className="scan-timings">
      <div className="scan-timings__head">
        <h2>Timeline</h2>
        {algoVersion && (
          <span className="scan-timings__algo" title={algoUpdated || undefined}>
            alg v{algoVersion}
            {algoUpdated ? ` · ${algoUpdated}` : ''}
          </span>
        )}
      </div>
      {!hasRealStarts && bars.length > 0 && (
        <p className="scan-timings__hint">
          Старый скан без offsets — шкала приблизительная. Новый поиск даст точный
          Gantt.
        </p>
      )}
      <div className="scan-gantt" role="img" aria-label="Диаграмма Ганта таймингов">
        <div className="scan-gantt__axis">
          <span>0</span>
          <span>{formatMs(axisEnd / 2)}</span>
          <span>{formatMs(axisEnd)}</span>
        </div>
        <ul className="scan-gantt__rows">
          {bars.map((b) => {
            const left = (b.start / axisEnd) * 100
            const width = Math.max(0.4, (b.duration / axisEnd) * 100)
            return (
              <li key={b.key} className="scan-gantt__row">
                <span className="scan-gantt__label" title={b.key}>
                  {b.label}
                </span>
                <div className="scan-gantt__track">
                  <div
                    className="scan-gantt__bar"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      background: b.color,
                    }}
                    title={`${b.label}: ${formatMs(b.start)} → +${formatMs(b.duration)}`}
                  />
                </div>
                <span className="scan-gantt__meta">
                  <span className="scan-gantt__ms">{formatMs(b.duration)}</span>
                  <span className="scan-gantt__pct">
                    {formatPct(b.duration, totalMs || axisEnd)}
                  </span>
                </span>
              </li>
            )
          })}
        </ul>
        {totalMs > 0 && (
          <div className="scan-gantt__total">
            <span>Всего</span>
            <strong>{formatMs(totalMs)}</strong>
          </div>
        )}
        {extraRows.length > 0 && (
          <div className="scan-gantt__extra">
            {extraRows.map((r) => (
              <div key={r.key} className="scan-gantt__total">
                <span>{r.label}</span>
                <strong>{formatMs(r.ms)}</strong>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/** Work JPEG / label for re-search when result opened via ?scanid= (no File in session). */
function workPhotoFilename(data: FindWineResult): string | null {
  const status = data.status || {}
  const steps = (status.steps || {}) as Record<string, any>
  const normalize = steps.normalize || {}
  const save = steps.save || {}
  const names = [
    status.work_filename,
    normalize.work_filename,
    save.filename,
  ].filter((n): n is string => typeof n === 'string' && n.length > 0)
  const browserExt = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']
  const preferred =
    names.find((n) => browserExt.some((ext) => n.toLowerCase().endsWith(ext))) ||
    names[0]
  if (!preferred) return null
  const safe = preferred.replace(/^.*[\\/]/, '')
  // Не брать _crops / _label / _orient в дропзону — там рамки или выровненный кадр
  if (/_(crops|label|orient)\./i.test(safe)) return null
  return safe
}

function workPhotoUrl(data: FindWineResult): string | null {
  const safe = workPhotoFilename(data)
  if (!safe) return null
  const bust = data.search_photos_id ?? data.status?.search_photos_id
  const base = `/api/search-photos/${encodeURIComponent(safe)}`
  return bust != null ? `${base}?v=${encodeURIComponent(String(bust))}` : base
}

async function fileFromFindWineResult(data: FindWineResult): Promise<File | null> {
  const status = data.status || {}
  const steps = (status.steps || {}) as Record<string, any>
  const normalize = steps.normalize || {}
  const save = steps.save || {}
  const names = [
    workPhotoFilename(data),
    status.work_filename,
    normalize.work_filename,
    save.filename,
    data.label_filename,
  ].filter((n): n is string => typeof n === 'string' && n.length > 0)

  const browserExt = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']
  const preferred =
    names.find((n) => {
      const base = n.replace(/^.*[\\/]/, '')
      return (
        browserExt.some((ext) => base.toLowerCase().endsWith(ext)) &&
        !/_(crops|label|orient)\./i.test(base)
      )
    }) || names[0]
  if (!preferred) return null

  const safe = preferred.replace(/^.*[\\/]/, '')
  const res = await fetch(`/api/search-photos/${encodeURIComponent(safe)}`)
  if (!res.ok) return null
  const blob = await res.blob()
  return new File([blob], safe, { type: blob.type || 'image/jpeg' })
}

const CAND_SORT_COOKIE = 'vino_cand_sort'

const CAND_SORT_OPTIONS: { key: CandSortKey; label: string }[] = [
  { key: 'ocr_A', label: 'Score A' },
  { key: 'ocr_B', label: 'Score B' },
  { key: 'ocr_C', label: 'Score C' },
  { key: 'ocr_D', label: 'Score D' },
  { key: 'ocr_gemini', label: 'Score G' },
  { key: 'ocr_openai', label: 'Score O' },
  { key: 'ocr_deepseek', label: 'Score K' },
  { key: 'ocr_qwen', label: 'Score Q' },
  { key: 'ocr_yandex', label: 'Score Y' },
  { key: 'ocr_google_vision', label: 'Score V' },
  { key: 'cos', label: 'cos' },
  { key: 'geo', label: 'geo' },
  { key: 'xgb', label: 'XGB' },
  { key: 'xgb_fin', label: 'XGB_fin' },
  { key: 'crenc', label: 'CrEnc' },
  { key: 'crenc_fin', label: 'CrEnc_fin' },
  { key: 'llm_txt', label: 'LLM txt' },
]

function isCandSortKey(value: string | null | undefined): value is CandSortKey {
  return CAND_SORT_OPTIONS.some((o) => o.key === value)
}

function readCandSortCookie(): CandSortKey {
  if (typeof document === 'undefined') return 'cos'
  const match = document.cookie.match(/(?:^|;\s*)vino_cand_sort=([^;]*)/)
  const raw = match ? decodeURIComponent(match[1]) : null
  return isCandSortKey(raw) ? raw : 'cos'
}

function writeCandSortCookie(key: CandSortKey) {
  const maxAge = 60 * 60 * 24 * 365
  document.cookie = `${CAND_SORT_COOKIE}=${encodeURIComponent(key)}; path=/; max-age=${maxAge}; SameSite=Lax`
}

function channelFromSortKey(sortBy: CandSortKey): OcrScoreChannel | null {
  if (!sortBy.startsWith('ocr_')) return null
  const rest = sortBy.slice('ocr_'.length)
  if (rest === 'A' || rest === 'B' || rest === 'C' || rest === 'D') return rest
  if (
    rest === 'gemini' ||
    rest === 'openai' ||
    rest === 'deepseek' ||
    rest === 'qwen' ||
    rest === 'yandex' ||
    rest === 'google_vision'
  ) {
    return rest
  }
  return null
}

function sortCandidates(
  items: FindWineCandidate[],
  sortBy: CandSortKey,
  ocrScoresByWine: Map<number, Partial<Record<OcrScoreChannel, number>>>,
  ocrTextScoresByWine: Map<
    number,
    Partial<Record<OcrScoreChannel, number>>
  > = new Map(),
): FindWineCandidate[] {
  const numOrNeg = (v: number | null | undefined): number =>
    v != null && Number.isFinite(Number(v)) ? Number(v) : -1

  const cosOf = (w: FindWineCandidate) => Number(w.cosine_similarity || 0)

  /** fin-метрики: Score OCR (fin1), XGB_fin, CrEnc_fin — каскад fin → txt → cos. */
  const isFinSort =
    sortBy.startsWith('ocr_') || sortBy === 'xgb_fin' || sortBy === 'crenc_fin'

  const finOf = (w: FindWineCandidate): number => {
    if (sortBy === 'xgb_fin') return numOrNeg(w.xgb_fin)
    if (sortBy === 'crenc_fin') return numOrNeg(w.crenc_fin)
    const ch = channelFromSortKey(sortBy)
    if (!ch) return -1
    return numOrNeg(ocrScoresByWine.get(w.id)?.[ch])
  }

  const txtOf = (w: FindWineCandidate): number => {
    if (sortBy === 'xgb_fin') return numOrNeg(w.xgb_score)
    if (sortBy === 'crenc_fin') return numOrNeg(w.crenc_score)
    const ch = channelFromSortKey(sortBy)
    if (!ch) return -1
    return numOrNeg(ocrTextScoresByWine.get(w.id)?.[ch])
  }

  const scoreOf = (w: FindWineCandidate): number => {
    if (sortBy === 'cos') return cosOf(w)
    if (sortBy === 'geo') return numOrNeg(w.geometry_score)
    if (sortBy === 'xgb') return numOrNeg(w.xgb_score)
    if (sortBy === 'xgb_fin') return numOrNeg(w.xgb_fin)
    if (sortBy === 'crenc') return numOrNeg(w.crenc_score)
    if (sortBy === 'crenc_fin') return numOrNeg(w.crenc_fin)
    if (sortBy === 'llm_txt') return numOrNeg(w.llm_txt)
    const ch = channelFromSortKey(sortBy)
    if (!ch) return -1
    return numOrNeg(ocrScoresByWine.get(w.id)?.[ch])
  }

  return [...items].sort((a, b) => {
    if (isFinSort) {
      const dFin = finOf(b) - finOf(a)
      if (dFin !== 0) return dFin
      const dTxt = txtOf(b) - txtOf(a)
      if (dTxt !== 0) return dTxt
      const dCos = cosOf(b) - cosOf(a)
      if (dCos !== 0) return dCos
      return a.id - b.id
    }
    const diff = scoreOf(b) - scoreOf(a)
    if (diff !== 0) return diff
    return a.id - b.id
  })
}

function sortMetricClass(
  sortBy: CandSortKey,
  metric: CandSortKey,
): string {
  if (sortBy !== metric) return ''
  if (metric === 'xgb' || metric === 'xgb_fin') {
    return ' is-sort-active is-sort-xgb'
  }
  if (metric === 'crenc' || metric === 'crenc_fin') {
    return ' is-sort-active is-sort-crenc'
  }
  if (metric === 'llm_txt') {
    return ' is-sort-active is-sort-llm-txt'
  }
  return ' is-sort-active'
}

function CandidateCard({
  wine,
  overlap,
  ocrMarks = [],
  ocrScores,
  ocrChannels = [],
  ocrQueryText = '',
  exclusiveLexicon = null,
  hsvStep = null,
  labelTextHr = null,
  embeddingLabel,
  cosSiglip2,
  cosDinov3,
  sortBy = 'cos',
  isFinalWinner = false,
  isXgbWinner = false,
  isSimilarBand = false,
  isManualMatch = false,
  manualMatchEnabled = false,
  onManualMatchChange,
  finalistRank = null,
  finalMethod = 'fin1',
  finalOcrPrimary = null,
}: {
  wine: FindWineCandidate
  overlap: boolean
  ocrMarks?: OcrMarkId[]
  ocrScores?: Partial<Record<OcrScoreChannel, number>>
  ocrChannels?: OcrScoreChannel[]
  ocrQueryText?: string
  exclusiveLexicon?: ExclusiveLexiconStep | null
  hsvStep?: HsvStepInfo | null
  labelTextHr?: LabelTextHardRejectStep | null
  embeddingLabel?: string
  cosSiglip2?: number | null
  cosDinov3?: number | null
  sortBy?: CandSortKey
  isFinalWinner?: boolean
  isXgbWinner?: boolean
  isSimilarBand?: boolean
  isManualMatch?: boolean
  manualMatchEnabled?: boolean
  onManualMatchChange?: (wineId: number, checked: boolean) => void
  finalistRank?: number | null
  finalMethod?: string
  finalOcrPrimary?: OcrScoreChannel | null
}) {
  const cardRef = useRef<HTMLAnchorElement>(null)
  const mediaRef = useRef<HTMLDivElement>(null)
  const [hovered, setHovered] = useState(false)
  const [popupPos, setPopupPos] = useState<{ top: number; left: number } | null>(
    null,
  )
  const [labelBroken, setLabelBroken] = useState(false)
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const href = `/wines/${encodeURIComponent(wine.slug || String(wine.id))}`

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
    const el = mediaRef.current || cardRef.current
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

  return (
    <>
    <a
      ref={cardRef}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`cand-card ${overlap ? 'is-overlap' : ''} ${hovered ? 'is-hovered' : ''} ${isFinalWinner ? 'is-final-winner' : ''} ${isXgbWinner ? 'is-xgb-winner' : ''} ${isSimilarBand && !isFinalWinner ? 'is-similar-band' : ''} ${isManualMatch ? 'is-manual-match' : ''}`.trim()}
      title={
        [
          isFinalWinner ? 'Победитель FinalScore' : '',
          isXgbWinner ? 'Победитель XGB_fin' : '',
          isSimilarBand && !isFinalWinner ? 'Similar (между similar и match)' : '',
          isManualMatch ? 'Это вино!' : '',
        ]
          .filter(Boolean)
          .join(' · ') || undefined
      }
    >
      <div
        ref={mediaRef}
        className="cand-card__media"
        onMouseEnter={openHover}
        onMouseLeave={closeHoverSoon}
      >
        {finalistRank != null && finalistRank >= 1 && finalistRank <= 5 && (
          <span
            className="cand-card__rank"
            title={`Место ${finalistRank} по score метода финалиста`}
          >
            {finalistRank}
          </span>
        )}
        {wine.label_url && !labelBroken ? (
          <img
            src={wine.label_url}
            alt={wine.name || `wine ${wine.id}`}
            loading="lazy"
            onError={() => setLabelBroken(true)}
          />
        ) : (
          <div className="cand-card__placeholder" />
        )}
      </div>
      <div className={`cand-card__meta ${overlap ? 'is-overlap-text' : ''}`}>
        <p className={`cand-card__sim${sortMetricClass(sortBy, 'cos')}`}>
          COS <em>{formatScore00(wine.cosine_similarity)}</em>
        </p>
        {wine.hsv != null && (
          <p
            className="cand-card__hsv"
            title="Расстояние Бхаттачарии между HSV-гистограммами. 0 — одна гамма, 1 — гаммы не пересекаются."
          >
            HSV <em>{formatScore00(wine.hsv)}</em>
          </p>
        )}
        {wine.color_delta != null && (
          <p
            className="cand-card__color-delta"
            title="CIEDE2000 доминантных цветов искомой этикетки и кандидата. Меньше — ближе по цвету."
          >
            ColorDelta <em>{Number(wine.color_delta).toFixed(2)}</em>
          </p>
        )}
        {(wine.xgb_score != null || wine.xgb_fin != null) && (
          <p
            className="cand-card__xgb"
            title="XGBoost OCR↔label"
          >
            <span className={sortMetricClass(sortBy, 'xgb').trim() || undefined}>
              XGB <em>{formatScore00(wine.xgb_score)}</em>
            </span>
            {wine.exclusive_rejected || wine.label_text_hard_reject ? (
              <span
                className={sortMetricClass(sortBy, 'xgb_fin').trim() || undefined}
                title={
                  wine.xgb_fin_reason
                    ? `hard reject → XGB_fin=0; без reject было бы ${formatScore00(wine.xgb_fin_pre_reject ?? wine.xgb_fin)} (${wine.xgb_fin_reason})`
                    : 'hard reject → XGB_fin=0; серым — значение без reject'
                }
              >
                XGB_fin <em>0</em>
                {wine.xgb_fin_pre_reject != null && (
                  <span className="cand-card__xgb-fin-ghost">
                    {' '}
                    ({formatScore00(wine.xgb_fin_pre_reject)})
                  </span>
                )}
              </span>
            ) : (
              <span
                className={sortMetricClass(sortBy, 'xgb_fin').trim() || undefined}
              >
                XGB_fin <em>{formatScore00(wine.xgb_fin)}</em>
              </span>
            )}
          </p>
        )}
        {(wine.crenc_score != null || wine.crenc_fin != null) && (
          <p
            className="cand-card__crenc"
            title="Cross Encoder OCR↔label"
          >
            <span className={sortMetricClass(sortBy, 'crenc').trim() || undefined}>
              CrEnc <em>{formatScore00(wine.crenc_score)}</em>
            </span>
            <span
              className={sortMetricClass(sortBy, 'crenc_fin').trim() || undefined}
            >
              CrEnc_fin <em>{formatScore00(wine.crenc_fin)}</em>
            </span>
          </p>
        )}
        {wine.llm_txt != null && (
          <p
            className={`cand-card__llm-txt${sortMetricClass(sortBy, 'llm_txt')}`}
            title="OpenAI сравнение текста — вероятность совпадения 0…1"
          >
            LLM txt <em>{formatScore00(wine.llm_txt)}</em>
          </p>
        )}
        {(wine.geometry_score != null || wine.geometry_inliers != null) && (
          <p
            className={`cand-card__geom${sortMetricClass(sortBy, 'geo')}`}
            title="Геометрия: инлайеры / score"
          >
            geo {wine.geometry_inliers ?? 0}
            {wine.geometry_score != null
              ? ` · ${formatScore00(wine.geometry_score)}`
              : ''}
          </p>
        )}
        {(() => {
          const finCh = visibleOcrFinChannels(
            wine,
            finalMethod,
            ocrChannels,
            ocrScores,
          )
          if (!finCh.length) return null
          return (
            <p
              className="cand-card__ocr-scores"
              title={`${finalByOcrRowLabel(finalMethod)} по каждому OCR (выбранный метод сравнения текстов)`}
            >
              <span className="cand-card__ocr-row-label">
                {finalByOcrRowLabel(finalMethod)}
              </span>
              {finCh.map((ch) => (
                <span
                  key={`s-${ch}`}
                  className={`cand-card__ocr-score is-ocr-${ch}${
                    finalOcrPrimary === ch ? ' is-final-ocr' : ''
                  }${sortMetricClass(
                    sortBy,
                    `ocr_${ch}` as CandSortKey,
                  )}`}
                  title={
                    finalOcrPrimary === ch
                      ? 'Final OCR (канал решения)'
                      : undefined
                  }
                >
                  <em
                    className={
                      ocrMarks.includes(ch) || finalOcrPrimary === ch
                        ? 'is-best-channel'
                        : undefined
                    }
                  >
                    {ocrChannelLetter(ch)}
                  </em>
                  {formatScore00(ocrScores?.[ch])}
                </span>
              ))}
            </p>
          )
        })()}
        <p className="cand-card__id">id {wine.id}</p>
        <p className="cand-card__name">{wine.name || '—'}</p>
        {manualMatchEnabled && (
          <label
            className="cand-card__manual"
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
            }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <input
              type="checkbox"
              checked={isManualMatch}
              onChange={(e) => {
                e.stopPropagation()
                onManualMatchChange?.(wine.id, e.target.checked)
              }}
            />
            Это вино!
          </label>
        )}
      </div>
    </a>
      {hovered && popupPos && (
        <WineHoverPopup
          wine={wine}
          pos={popupPos}
          embeddingLabel={embeddingLabel}
          cosSiglip2={cosSiglip2}
          cosDinov3={cosDinov3}
          ocrScores={ocrScores}
          ocrChannels={ocrChannels}
          ocrQueryText={ocrQueryText}
          ocrMarks={ocrMarks}
          exclusiveLexicon={exclusiveLexicon}
          hsvStep={hsvStep}
          labelTextHr={labelTextHr}
          finalMethod={finalMethod}
          onMouseEnter={openHover}
          onMouseLeave={closeHoverSoon}
        />
      )}
    </>
  )
}

function OcrVariantsBlock({
  result,
  wineById,
  cosByWine,
  ocrScoresByWine,
  ocrQueryText = '',
  exclusiveLexicon = null,
  hsvStep = null,
  labelTextHr = null,
}: {
  result: FindWineResult
  wineById: Map<number, FindWineCandidate>
  cosByWine: {
    siglip2: Map<number, number>
    dinov3: Map<number, number>
  }
  ocrScoresByWine: Map<number, Partial<Record<OcrScoreChannel, number>>>
  ocrQueryText?: string
  exclusiveLexicon?: ExclusiveLexiconStep | null
  hsvStep?: HsvStepInfo | null
  labelTextHr?: LabelTextHardRejectStep | null
}) {
  type LlmOcrStep = {
    ok?: boolean
    text?: string
    error?: string
    skipped?: boolean
    json?: Record<string, unknown>
    model?: string
    ms?: number
    ocr_ms?: number
    name?: string
    tokens_input?: number
    tokens_output?: number
    from_label_detect?: boolean
    reason?: string
  }

  const ocr = result.status?.steps?.ocr as
    | {
        ok?: boolean
        text_aggregated?: string
        gemini?: Record<string, unknown>
        openai?: Record<string, unknown>
        variants?: Record<
          string,
          {
            ok?: boolean
            name?: string
            text?: string
            preprocess_ms?: number
            ocr_ms?: number
            ms?: number
            filename?: string
            error?: string
            alias_of?: string
            engine?: string
            preprocess?: string
            json?: Record<string, unknown>
            skipped?: boolean
            model?: string
          }
        >
      }
    | undefined
  const help = result.status?.steps?.ocr_wine_id as OcrWineIdHelp | undefined
  const steps = (result.status?.steps || {}) as Record<string, any>
  const xgbThreshold =
    typeof steps.xgb_match?.threshold === 'number'
      ? steps.xgb_match.threshold
      : null
  const crencThreshold =
    typeof steps.crenc_match?.threshold === 'number'
      ? steps.crenc_match.threshold
      : null
  const softExplain = help?.explain?.soft_tfidf
  const finWeights = help?.explain?.final_weights
  const finWeightsLabel = finWeights
    ? Object.entries(finWeights)
        .map(([k, v]) => `${k}=${Number(v).toFixed(2)}`)
        .join(', ')
    : 'w_ocr / w_emb из Настроек'

  const resolveLlmOcrStep = (
    engine: 'gemini' | 'openai' | 'deepseek' | 'qwen',
  ): LlmOcrStep | null => {
    const raw = (steps[`ocr_${engine}`] ||
      ocr?.[engine] ||
      ocr?.variants?.[engine]) as LlmOcrStep | undefined
    const ld = steps[`${engine}_label_detect`] as
      | {
          ok?: boolean
          text?: string
          error?: string
          json?: Record<string, unknown>
          model?: string
          ms?: number
          api_ms?: number
          ocr_variant?: LlmOcrStep
        }
      | undefined

    const hasText = (s?: LlmOcrStep | null) =>
      Boolean(s && String(s.text || '').trim())

    // 1) Готовый OCR-only / reuse с текстом
    if (raw && !raw.skipped && (raw.ok || hasText(raw) || raw.error)) {
      return raw
    }
    if (raw && raw.from_label_detect && hasText(raw)) {
      return raw
    }

    // 2) OCR из label+OCR (рамка+текст одним запросом)
    if (ld) {
      const ov = ld.ocr_variant
      if (ov && (ov.ok || hasText(ov))) {
        return {
          ...ov,
          from_label_detect: true,
          name:
            ov.name ||
            (engine === 'gemini'
              ? 'Google Gemini · label+OCR'
              : 'OpenAI · label+OCR'),
          model: ov.model || ld.model,
          ms: ov.ms ?? ov.ocr_ms ?? ld.api_ms ?? ld.ms,
          ocr_ms: ov.ocr_ms ?? ov.ms ?? ld.api_ms ?? ld.ms,
          json: ov.json || ld.json,
        }
      }
      if (String(ld.text || '').trim()) {
        return {
          ok: true,
          text: ld.text,
          json: ld.json,
          model: ld.model,
          ms: ld.api_ms ?? ld.ms,
          ocr_ms: ld.api_ms ?? ld.ms,
          name:
            engine === 'gemini'
              ? 'Google Gemini · label+OCR'
              : 'OpenAI · label+OCR',
          from_label_detect: true,
        }
      }
      // label+OCR вызывался, но без текста — покажем ошибку/JSON ответа
      if (raw?.from_label_detect || ld.error || ld.json) {
        return {
          ok: false,
          text: undefined,
          error: ld.error || raw?.error || raw?.reason || 'нет текста OCR',
          json: ld.json,
          model: ld.model,
          ms: ld.api_ms ?? ld.ms,
          name:
            engine === 'gemini'
              ? 'Google Gemini · label+OCR'
              : 'OpenAI · label+OCR',
          from_label_detect: true,
        }
      }
    }

    // 3) OCR-only с ошибкой (не disabled)
    if (
      raw &&
      !raw.skipped &&
      raw.reason !== 'disabled_in_settings'
    ) {
      return raw
    }
    return null
  }

  const geminiStep = resolveLlmOcrStep('gemini')
  const openaiStep = resolveLlmOcrStep('openai')
  const deepseekStep = resolveLlmOcrStep('deepseek')
  const qwenStep = resolveLlmOcrStep('qwen')
  const yandexStep = resolveLlmOcrStep('yandex')
  const googleVisionStep = resolveLlmOcrStep('google_vision')

  if (
    !ocr?.variants &&
    !geminiStep &&
    !openaiStep &&
    !deepseekStep &&
    !qwenStep &&
    !yandexStep &&
    !googleVisionStep
  )
    return null
  const engines = (
    (result.status?.steps?.ocr as { engines?: string[] } | undefined)?.engines || []
  ).filter(Boolean)
  const engineLabel =
    engines.length > 0
      ? engines
          .map((e) => OCR_ENGINE_TITLES[e] || e)
          .join(' · ')
      : 'OCR'
  const keys = Object.keys(ocr?.variants || {})
    .filter((k) => {
      if (
        k === 'gemini' ||
        k === 'openai' ||
        k === 'deepseek' ||
        k === 'qwen' ||
        k === 'yandex' ||
        k === 'google_vision'
      )
        return false
      const v = ocr?.variants?.[k]
      return v && !(v as { alias_of?: string }).alias_of
    })
    .sort((a, b) => a.localeCompare(b))

  const groupBests = collectOcrGroupBests(result)
  const groups = groupBests.map((g) => ({
    ...g,
    keys: keys.filter((k) => k === g.letter || k.startsWith(`${g.letter}_`)),
  }))
  const showClassicGroups = groups.some((g) => g.keys.length > 0)

  const weights = help?.weights || help?.explain?.weights
  const geminiHelp = help?.per_variant?.gemini
  const geminiBest = geminiHelp?.best_final || geminiHelp?.best_visual
  const openaiHelp = help?.per_variant?.openai
  const openaiBest = openaiHelp?.best_final || openaiHelp?.best_visual
  const deepseekHelp = help?.per_variant?.deepseek
  const deepseekBest = deepseekHelp?.best_final || deepseekHelp?.best_visual
  const qwenHelp = help?.per_variant?.qwen
  const qwenBest = qwenHelp?.best_final || qwenHelp?.best_visual
  const yandexHelp = help?.per_variant?.yandex
  const yandexBest = yandexHelp?.best_final || yandexHelp?.best_visual
  const googleVisionHelp = help?.per_variant?.google_vision
  const googleVisionBest =
    googleVisionHelp?.best_final || googleVisionHelp?.best_visual

  return (
    <section className="ocr-block">
      <h2>
        OCR
        {showClassicGroups ? ` · preprocess A–D · ${engineLabel}` : ''}
        {geminiStep ? ' · Gemini' : ''}
        {openaiStep ? ' · OpenAI' : ''}
        {deepseekStep ? ' · DeepSeek' : ''}
        {qwenStep ? ' · Qwen' : ''}
        {yandexStep ? ' · Yandex' : ''}
        {googleVisionStep ? ' · Google Vision' : ''}
      </h2>
      <p className="ocr-legend">
        Цветная рамка = победитель FinalScore (embedding+OCR) по каналу:{' '}
        {showClassicGroups &&
          OCR_LETTERS.map((l) => (
            <span key={l} className={`ocr-legend__item is-ocr-${l}`}>
              {l}
            </span>
          ))}
        {geminiStep && (
          <span className="ocr-legend__item is-ocr-gemini">G Gemini</span>
        )}
        {openaiStep && (
          <span className="ocr-legend__item is-ocr-openai">O OpenAI</span>
        )}
        {deepseekStep && (
          <span className="ocr-legend__item is-ocr-deepseek">K DeepSeek</span>
        )}
        {qwenStep && (
          <span className="ocr-legend__item is-ocr-qwen">Q Qwen</span>
        )}
        {yandexStep && (
          <span className="ocr-legend__item is-ocr-yandex">Y Yandex</span>
        )}
        {googleVisionStep && (
          <span className="ocr-legend__item is-ocr-google_vision">
            V Google Vision
          </span>
        )}
      </p>
      <div className="ocr-explain">
        <p>
          <strong>Как считаются fin-метрики</strong>
        </p>
        <ul className="ocr-explain__fin">
          <li>
            <strong>fin1</strong> (= FinalScore, Score на карточках) — смесь
            TextScore (OCR↔каталог) и cosine:{' '}
            <code>
              fin1 = w_ocr×TextScore + w_emb×Cosine ({finWeightsLabel})
            </code>
            . Hard mismatch или exclusive → 0. Победитель канала = argmax fin1.
          </li>
          <li>
            <strong>fin2</strong> (= FinalScore2) — те же веса, но TextScore2 Soft
            TF-IDF (Google Vision / Yandex):{' '}
            <code>
              {softExplain?.final_formula ||
                'fin2 = w_ocr×TextScore2 + w_emb×Cosine'}
            </code>
            . Exclusive reject → 0.
          </li>
          <li>
            <strong>XGB_fin</strong> — XGB = TextScore (P(match) модели); затем как
            fin1:{' '}
            <code>
              XGB_fin = w_ocr×XGB + w_emb×Cosine
            </code>
            , если XGB в зоне match/similar
            {xgbThreshold != null ? ` (match>${xgbThreshold})` : ''}; ниже
            similar, exclusive или label-text → 0. Победитель — только из
            match-зоны по XGB_fin.
          </li>
          <li>
            <strong>CrEnc_fin</strong> — CrEnc = TextScore;{' '}
            <code>
              CrEnc_fin = w_ocr×CrEnc + w_emb×Cosine
            </code>
            , если CrEnc в зоне match/similar
            {crencThreshold != null ? ` (match>${crencThreshold})` : ''}; ниже
            similar или exclusive → 0. Вино в результате — argmax выбранного
            Final score (fin1, fin2, XGBoost, Cross Encoder или OpenAI txt).
          </li>
        </ul>
        {help?.explain?.formula && (
          <p>
            <strong>Как считается TextScore:</strong> {help.explain.formula}
          </p>
        )}
        {softExplain?.formula && (
          <p>
            <strong>TextScore2 (Soft):</strong> {softExplain.formula}
          </p>
        )}
        {weights && (
          <p className="ocr-explain__weights">
            Веса TextScore:{' '}
            {Object.entries(weights)
              .map(([k, v]) => `${k}=${Number(v).toFixed(2)}`)
              .join(' · ')}
          </p>
        )}
        {finWeights && (
          <p className="ocr-explain__weights">
            Веса fin1 / fin2:{' '}
            {Object.entries(finWeights)
              .map(([k, v]) => `${k}=${Number(v).toFixed(2)}`)
              .join(' · ')}
          </p>
        )}
        {(help?.explain?.notes?.length ||
          softExplain?.veto ||
          softExplain?.exclusive) && (
          <ul>
            {(help?.explain?.notes || []).map((n) => (
              <li key={n}>{n}</li>
            ))}
            {softExplain?.veto && <li key="soft-veto">{softExplain.veto}</li>}
            {softExplain?.exclusive && (
              <li key="soft-excl">{softExplain.exclusive}</li>
            )}
          </ul>
        )}
      </div>
      {help?.best_final && (
        <p className="ocr-block__best">
          Итог FinalScore:{' '}
          <strong>id {String((help.best_final as { id?: number }).id)}</strong>
          {(help.best_final as { final_score?: number }).final_score != null &&
            ` · ${(help.best_final as { final_score?: number }).final_score!.toFixed(4)}`}
          {(help.best_final as { score?: number }).score != null &&
            ` · Text ${(help.best_final as { score?: number }).score}`}
          {(help.best_final as { cosine?: number }).cosine != null &&
            ` · cos ${(help.best_final as { cosine?: number }).cosine!.toFixed(4)}`}
        </p>
      )}
      {help?.best_variant_for_visual_support && (
        <p className="ocr-block__best">
          Лучше всего поддерживает visual-кандидатов:{' '}
          <strong>{help.best_variant_for_visual_support}</strong>
          {help.best_visual_support_score != null &&
            ` (TextScore ${help.best_visual_support_score})`}
        </p>
      )}
      {showClassicGroups && (
      <div className="ocr-groups">
        {groups.map((group) => (
          <section
            key={group.letter}
            className={`ocr-variant-group is-ocr-${group.letter}`}
          >
            <h3>
              <span className={`ocr-variant-group__letter is-ocr-${group.letter}`}>
                {group.letter}
              </span>
              Вариант {group.letter} · {group.title}
              {group.bestKey && (
                <span className={`ocr-variant-group__best is-ocr-${group.letter}`}>
                  лучший: {ocrVariantTitle(group.bestKey)} · FinalScore{' '}
                  {group.bestScore != null ? group.bestScore.toFixed(4) : '—'}
                  {group.wineId != null && (
                    <>
                      {' · wine id '}
                      <WineIdHover
                        wineId={group.wineId}
                        wineById={wineById}
                        cosByWine={cosByWine}
                        ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                      />
                    </>
                  )}
                </span>
              )}
            </h3>
            <div className="ocr-grid">
              {group.keys.map((key) => {
                const v = ocr?.variants?.[key]
                const h = help?.per_variant?.[key]
                if (!v) return null
                const isBest = group.bestKey === key
                const ent = h?.entities as
                  | {
                      producer?: string
                      wine?: string
                      vintage?: string
                      region?: string
                    }
                  | undefined
                const parts = h?.best_visual?.parts
                return (
                  <article
                    key={key}
                    className={`ocr-card${isBest ? ` is-best is-ocr-${group.letter}` : ''}`}
                  >
                    <header>
                      <strong>
                        {ocrVariantTitle(key, v.name)}
                        {isBest ? ' · лучший' : ''}
                      </strong>
                      <span>
                        prep {formatMs(Number(v.preprocess_ms || 0))} · OCR{' '}
                        {formatMs(Number(v.ocr_ms || 0))}
                      </span>
                    </header>
                    <pre>{v.text || (v as { error?: string }).error || '—'}</pre>
                    {(v as { json?: unknown }).json != null && (
                      <pre className="ocr-card__json">
                        {JSON.stringify((v as { json?: unknown }).json, null, 2)}
                      </pre>
                    )}
                    {ent && (
                      <p className="ocr-card__entities">
                        entities: producer={ent.producer || '—'} · wine={ent.wine || '—'} ·
                        vintage={ent.vintage || '—'} · region={ent.region || '—'}
                      </p>
                    )}
                    {h?.best_visual && (
                      <p className="ocr-card__match">
                        visual support: id{' '}
                        <WineIdHover
                          wineId={h.best_visual.id}
                          wineById={wineById}
                          cosByWine={cosByWine}
                          ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                        />{' '}
                        · TextScore {h.best_visual.score}
                        {h.best_visual.name ? ` · ${h.best_visual.name}` : ''}
                        {parts
                          ? ` · parts: wine=${((parts.wine || 0) * 100).toFixed(0)}% producer=${((parts.producer || 0) * 100).toFixed(0)}% vintage=${((parts.vintage || 0) * 100).toFixed(0)}% region=${((parts.region || 0) * 100).toFixed(0)}% other=${((parts.other || 0) * 100).toFixed(0)}%`
                          : ''}
                        {h.top_wine_ids?.length ? (
                          <>
                            {' · top ids: '}
                            {h.top_wine_ids.slice(0, 5).map((id, i) => (
                              <span key={id}>
                                {i > 0 ? ', ' : ''}
                                <WineIdHover
                                  wineId={id}
                                  wineById={wineById}
                                  cosByWine={cosByWine}
                                  ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                                />
                              </span>
                            ))}
                          </>
                        ) : null}
                      </p>
                    )}
                  </article>
                )
              })}
            </div>
          </section>
        ))}
      </div>
      )}

      {geminiStep && (
        <section className="ocr-variant-group is-ocr-gemini">
          <h3>
            <span className="ocr-variant-group__letter is-ocr-gemini">G</span>
            Gemini ·{' '}
            {geminiStep.from_label_detect
              ? 'рамка+OCR → JSON'
              : 'original label → JSON'}
            {geminiBest && (
              <span className="ocr-variant-group__best is-ocr-gemini">
                лучший: FinalScore{' '}
                {Number(
                  (geminiBest as { final_score?: number }).final_score ??
                    geminiBest.score,
                ).toFixed(4)}
                {geminiBest.id != null && (
                  <>
                    {' · wine id '}
                    <WineIdHover
                      wineId={geminiBest.id}
                      wineById={wineById}
                      cosByWine={cosByWine}
                      ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                    />
                  </>
                )}
              </span>
            )}
          </h3>
          <div className="ocr-grid">
            <article
              className={`ocr-card${geminiBest ? ' is-best is-ocr-gemini' : ''}`}
            >
              <header>
                <strong>
                  {geminiStep.name || 'Google Gemini'}
                  {geminiStep.model ? ` · ${geminiStep.model}` : ''}
                  {geminiStep.from_label_detect ? ' · from label+OCR' : ''}
                </strong>
                <span>
                  OCR{' '}
                  {formatMs(
                    Number(geminiStep.ocr_ms || geminiStep.ms || 0),
                  )}
                </span>
              </header>
              <pre>
                {geminiStep.text ||
                  geminiStep.error ||
                  (geminiStep.skipped ? 'пропущено' : '—')}
              </pre>
              {geminiStep.json != null && (
                <pre className="ocr-card__json">
                  {JSON.stringify(geminiStep.json, null, 2)}
                </pre>
              )}
              {geminiHelp?.entities && (
                <p className="ocr-card__entities">
                  entities: producer=
                  {String(
                    (geminiHelp.entities as { producer?: string }).producer ||
                      '—',
                  )}{' '}
                  · wine=
                  {String(
                    (geminiHelp.entities as { wine?: string }).wine || '—',
                  ).slice(0, 80)}{' '}
                  · vintage=
                  {String(
                    (geminiHelp.entities as { vintage?: string }).vintage ||
                      '—',
                  )}{' '}
                  · region=
                  {String(
                    (geminiHelp.entities as { region?: string }).region || '—',
                  )}
                </p>
              )}
              {geminiBest && (
                <p className="ocr-card__match">
                  visual support: id{' '}
                  <WineIdHover
                    wineId={geminiBest.id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />{' '}
                  · TextScore {geminiBest.score}
                  {geminiBest.name ? ` · ${geminiBest.name}` : ''}
                  {geminiHelp?.top_wine_ids?.length ? (
                    <>
                      {' · top ids: '}
                      {geminiHelp.top_wine_ids.slice(0, 5).map((id, i) => (
                        <span key={id}>
                          {i > 0 ? ', ' : ''}
                          <WineIdHover
                            wineId={id}
                            wineById={wineById}
                            cosByWine={cosByWine}
                            ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                          />
                        </span>
                      ))}
                    </>
                  ) : null}
                </p>
              )}
            </article>
          </div>
        </section>
      )}

      {openaiStep && (
        <section className="ocr-variant-group is-ocr-openai">
          <h3>
            <span className="ocr-variant-group__letter is-ocr-openai">O</span>
            OpenAI ·{' '}
            {openaiStep.from_label_detect
              ? 'рамка+OCR → JSON'
              : 'original label → JSON'}
            {openaiBest && (
              <span className="ocr-variant-group__best is-ocr-openai">
                лучший: FinalScore{' '}
                {Number(
                  (openaiBest as { final_score?: number }).final_score ??
                    openaiBest.score,
                ).toFixed(4)}
                {openaiBest.id != null && (
                  <>
                    {' · wine id '}
                    <WineIdHover
                      wineId={openaiBest.id}
                      wineById={wineById}
                      cosByWine={cosByWine}
                      ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                    />
                  </>
                )}
              </span>
            )}
          </h3>
          <div className="ocr-grid">
            <article
              className={`ocr-card${openaiBest ? ' is-best is-ocr-openai' : ''}`}
            >
              <header>
                <strong>
                  {openaiStep.name || 'OpenAI'}
                  {openaiStep.model ? ` · ${openaiStep.model}` : ''}
                  {openaiStep.from_label_detect ? ' · from label+OCR' : ''}
                </strong>
                <span>
                  OCR{' '}
                  {formatMs(
                    Number(openaiStep.ocr_ms || openaiStep.ms || 0),
                  )}
                  {openaiStep.tokens_input != null &&
                    ` · in ${openaiStep.tokens_input}`}
                  {openaiStep.tokens_output != null &&
                    ` · out ${openaiStep.tokens_output}`}
                </span>
              </header>
              <pre>
                {openaiStep.text ||
                  openaiStep.error ||
                  (openaiStep.skipped ? 'пропущено' : '—')}
              </pre>
              {openaiStep.json != null && (
                <pre className="ocr-card__json">
                  {JSON.stringify(openaiStep.json, null, 2)}
                </pre>
              )}
              {openaiHelp?.entities && (
                <p className="ocr-card__entities">
                  entities: producer=
                  {String(
                    (openaiHelp.entities as { producer?: string }).producer ||
                      '—',
                  )}{' '}
                  · wine=
                  {String(
                    (openaiHelp.entities as { wine?: string }).wine || '—',
                  ).slice(0, 80)}{' '}
                  · vintage=
                  {String(
                    (openaiHelp.entities as { vintage?: string }).vintage ||
                      '—',
                  )}{' '}
                  · region=
                  {String(
                    (openaiHelp.entities as { region?: string }).region || '—',
                  )}
                </p>
              )}
              {openaiBest && (
                <p className="ocr-card__match">
                  visual support: id{' '}
                  <WineIdHover
                    wineId={openaiBest.id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />{' '}
                  · TextScore {openaiBest.score}
                  {openaiBest.name ? ` · ${openaiBest.name}` : ''}
                  {openaiHelp?.top_wine_ids?.length ? (
                    <>
                      {' · top ids: '}
                      {openaiHelp.top_wine_ids.slice(0, 5).map((id, i) => (
                        <span key={id}>
                          {i > 0 ? ', ' : ''}
                          <WineIdHover
                            wineId={id}
                            wineById={wineById}
                            cosByWine={cosByWine}
                            ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                          />
                        </span>
                      ))}
                    </>
                  ) : null}
                </p>
              )}
            </article>
          </div>
        </section>
      )}

      {deepseekStep && (
        <section className="ocr-variant-group is-ocr-deepseek">
          <h3>
            <span className="ocr-variant-group__letter is-ocr-deepseek">K</span>
            DeepSeek · original label → JSON
            {deepseekBest && (
              <span className="ocr-variant-group__best is-ocr-deepseek">
                лучший: FinalScore{' '}
                {Number(
                  (deepseekBest as { final_score?: number }).final_score ??
                    deepseekBest.score,
                ).toFixed(4)}
                {deepseekBest.id != null && (
                  <>
                    {' · wine id '}
                    <WineIdHover
                      wineId={deepseekBest.id}
                      wineById={wineById}
                      cosByWine={cosByWine}
                      ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                    />
                  </>
                )}
              </span>
            )}
          </h3>
          <div className="ocr-grid">
            <article
              className={`ocr-card${deepseekBest ? ' is-best is-ocr-deepseek' : ''}`}
            >
              <header>
                <strong>
                  {deepseekStep.name || 'DeepSeek'}
                  {deepseekStep.model ? ` · ${deepseekStep.model}` : ''}
                </strong>
                <span>
                  OCR{' '}
                  {formatMs(
                    Number(deepseekStep.ocr_ms || deepseekStep.ms || 0),
                  )}
                  {deepseekStep.tokens_input != null &&
                    ` · in ${deepseekStep.tokens_input}`}
                  {deepseekStep.tokens_output != null &&
                    ` · out ${deepseekStep.tokens_output}`}
                </span>
              </header>
              <pre>
                {deepseekStep.text ||
                  deepseekStep.error ||
                  (deepseekStep.skipped ? 'пропущено' : '—')}
              </pre>
              {deepseekStep.json != null && (
                <pre className="ocr-card__json">
                  {JSON.stringify(deepseekStep.json, null, 2)}
                </pre>
              )}
              {deepseekHelp?.entities && (
                <p className="ocr-card__entities">
                  entities: producer=
                  {String(
                    (deepseekHelp.entities as { producer?: string }).producer ||
                      '—',
                  )}{' '}
                  · wine=
                  {String(
                    (deepseekHelp.entities as { wine?: string }).wine || '—',
                  ).slice(0, 80)}{' '}
                  · vintage=
                  {String(
                    (deepseekHelp.entities as { vintage?: string }).vintage ||
                      '—',
                  )}{' '}
                  · region=
                  {String(
                    (deepseekHelp.entities as { region?: string }).region || '—',
                  )}
                </p>
              )}
              {deepseekBest && (
                <p className="ocr-card__match">
                  visual support: id{' '}
                  <WineIdHover
                    wineId={deepseekBest.id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />{' '}
                  · TextScore {deepseekBest.score}
                  {deepseekBest.name ? ` · ${deepseekBest.name}` : ''}
                  {deepseekHelp?.top_wine_ids?.length ? (
                    <>
                      {' · top ids: '}
                      {deepseekHelp.top_wine_ids.slice(0, 5).map((id, i) => (
                        <span key={id}>
                          {i > 0 ? ', ' : ''}
                          <WineIdHover
                            wineId={id}
                            wineById={wineById}
                            cosByWine={cosByWine}
                            ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                          />
                        </span>
                      ))}
                    </>
                  ) : null}
                </p>
              )}
            </article>
          </div>
        </section>
      )}

      {qwenStep && (
        <section className="ocr-variant-group is-ocr-qwen">
          <h3>
            <span className="ocr-variant-group__letter is-ocr-qwen">Q</span>
            Qwen2.5-VL · original label → JSON
            {qwenBest && (
              <span className="ocr-variant-group__best is-ocr-qwen">
                лучший: FinalScore{' '}
                {Number(
                  (qwenBest as { final_score?: number }).final_score ??
                    qwenBest.score,
                ).toFixed(4)}
                {qwenBest.id != null && (
                  <>
                    {' · wine id '}
                    <WineIdHover
                      wineId={qwenBest.id}
                      wineById={wineById}
                      cosByWine={cosByWine}
                      ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                    />
                  </>
                )}
              </span>
            )}
          </h3>
          <div className="ocr-grid">
            <article
              className={`ocr-card${qwenBest ? ' is-best is-ocr-qwen' : ''}`}
            >
              <header>
                <strong>
                  {qwenStep.name || 'Qwen2.5-VL (HF)'}
                  {qwenStep.model ? ` · ${qwenStep.model}` : ''}
                </strong>
                <span>
                  OCR{' '}
                  {formatMs(Number(qwenStep.ocr_ms || qwenStep.ms || 0))}
                  {qwenStep.tokens_input != null &&
                    ` · in ${qwenStep.tokens_input}`}
                  {qwenStep.tokens_output != null &&
                    ` · out ${qwenStep.tokens_output}`}
                </span>
              </header>
              <pre>
                {qwenStep.text ||
                  qwenStep.error ||
                  (qwenStep.skipped ? 'пропущено' : '—')}
              </pre>
              {qwenStep.json != null && (
                <pre className="ocr-card__json">
                  {JSON.stringify(qwenStep.json, null, 2)}
                </pre>
              )}
              {qwenHelp?.entities && (
                <p className="ocr-card__entities">
                  entities: producer=
                  {String(
                    (qwenHelp.entities as { producer?: string }).producer ||
                      '—',
                  )}{' '}
                  · wine=
                  {String(
                    (qwenHelp.entities as { wine?: string }).wine || '—',
                  ).slice(0, 80)}{' '}
                  · vintage=
                  {String(
                    (qwenHelp.entities as { vintage?: string }).vintage ||
                      '—',
                  )}{' '}
                  · region=
                  {String(
                    (qwenHelp.entities as { region?: string }).region || '—',
                  )}
                </p>
              )}
              {qwenBest && (
                <p className="ocr-card__match">
                  visual support: id{' '}
                  <WineIdHover
                    wineId={qwenBest.id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />{' '}
                  · TextScore {qwenBest.score}
                  {qwenBest.name ? ` · ${qwenBest.name}` : ''}
                  {qwenHelp?.top_wine_ids?.length ? (
                    <>
                      {' · top ids: '}
                      {qwenHelp.top_wine_ids.slice(0, 5).map((id, i) => (
                        <span key={id}>
                          {i > 0 ? ', ' : ''}
                          <WineIdHover
                            wineId={id}
                            wineById={wineById}
                            cosByWine={cosByWine}
                            ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                          />
                        </span>
                      ))}
                    </>
                  ) : null}
                </p>
              )}
            </article>
          </div>
        </section>
      )}

      {yandexStep && (
        <section className="ocr-variant-group is-ocr-yandex">
          <h3>
            <span className="ocr-variant-group__letter is-ocr-yandex">Y</span>
            Yandex Vision · original label → lines
            {yandexBest && (
              <span className="ocr-variant-group__best is-ocr-yandex">
                лучший: FinalScore{' '}
                {Number(
                  (yandexBest as { final_score?: number }).final_score ??
                    yandexBest.score,
                ).toFixed(4)}
                {yandexBest.id != null && (
                  <>
                    {' · id '}
                    <WineIdHover
                      wineId={yandexBest.id}
                      wineById={wineById}
                      cosByWine={cosByWine}
                      ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                    />
                  </>
                )}
              </span>
            )}
          </h3>
          <div className="ocr-cards">
            <article
              className={`ocr-card${yandexBest ? ' is-best is-ocr-yandex' : ''}`}
            >
              <header>
                <strong>{yandexStep.name || 'Yandex OCR'}</strong>
                <span>
                  {yandexStep.ok === false
                    ? 'ошибка'
                    : `${(yandexStep.text || '').length} симв.`}
                  {yandexStep.ms != null ? ` · ${Math.round(Number(yandexStep.ms))} мс` : ''}
                </span>
              </header>
              {yandexStep.error && (
                <p className="ocr-card__error">{String(yandexStep.error)}</p>
              )}
              <pre className="ocr-card__text">
                {(yandexStep.text || '').trim() || '—'}
              </pre>
              {yandexBest && (
                <p className="ocr-card__match">
                  FinalScore winner: id{' '}
                  <WineIdHover
                    wineId={yandexBest.id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />{' '}
                  ·{' '}
                  {Number(
                    (yandexBest as { final_score?: number }).final_score ??
                      yandexBest.score,
                  ).toFixed(4)}
                  {yandexBest.name ? ` · ${yandexBest.name}` : ''}
                </p>
              )}
            </article>
          </div>
        </section>
      )}

      {googleVisionStep && (
        <section className="ocr-variant-group is-ocr-google_vision">
          <h3>
            <span className="ocr-variant-group__letter is-ocr-google_vision">
              V
            </span>
            Google Cloud Vision · original label → lines
            {googleVisionBest && (
              <span className="ocr-variant-group__best is-ocr-google_vision">
                лучший: FinalScore{' '}
                {Number(
                  (googleVisionBest as { final_score?: number }).final_score ??
                    googleVisionBest.score,
                ).toFixed(4)}
                {googleVisionBest.id != null && (
                  <>
                    {' · id '}
                    <WineIdHover
                      wineId={googleVisionBest.id}
                      wineById={wineById}
                      cosByWine={cosByWine}
                      ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                    />
                  </>
                )}
              </span>
            )}
          </h3>
          <div className="ocr-cards">
            <article
              className={`ocr-card${googleVisionBest ? ' is-best is-ocr-google_vision' : ''}`}
            >
              <header>
                <strong>{googleVisionStep.name || 'Google Vision OCR'}</strong>
                <span>
                  {googleVisionStep.ok === false
                    ? 'ошибка'
                    : `${(googleVisionStep.text || '').length} симв.`}
                  {googleVisionStep.ms != null
                    ? ` · ${Math.round(Number(googleVisionStep.ms))} мс`
                    : ''}
                </span>
              </header>
              {googleVisionStep.error && (
                <p className="ocr-card__error">
                  {String(googleVisionStep.error)}
                </p>
              )}
              <pre className="ocr-card__text">
                {(googleVisionStep.text || '').trim() || '—'}
              </pre>
              {googleVisionBest && (
                <p className="ocr-card__match">
                  FinalScore winner: id{' '}
                  <WineIdHover
                    wineId={googleVisionBest.id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />{' '}
                  ·{' '}
                  {Number(
                    (googleVisionBest as { final_score?: number }).final_score ??
                      googleVisionBest.score,
                  ).toFixed(4)}
                  {googleVisionBest.name ? ` · ${googleVisionBest.name}` : ''}
                </p>
              )}
            </article>
          </div>
        </section>
      )}

      {help?.ensemble?.best_visual && (
        <p className="ocr-block__ensemble">
          Ensemble OCR: id{' '}
          <WineIdHover
            wineId={help.ensemble.best_visual.id}
            wineById={wineById}
            cosByWine={cosByWine}
            ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
          />{' '}
          · TextScore {help.ensemble.best_visual.score}
          {help.ensemble.best_visual.name
            ? ` · ${help.ensemble.best_visual.name}`
            : ''}
          {help.ensemble.top_wine_ids?.length ? (
            <>
              {' · top: '}
              {help.ensemble.top_wine_ids.slice(0, 8).map((id, i) => (
                <span key={id}>
                  {i > 0 ? ', ' : ''}
                  <WineIdHover
                    wineId={id}
                    wineById={wineById}
                    cosByWine={cosByWine}
                    ocrScoresByWine={ocrScoresByWine}
                        ocrQueryText={ocrQueryText}
                        exclusiveLexicon={exclusiveLexicon}
                        hsvStep={hsvStep}
                        labelTextHr={labelTextHr}
                  />
                </span>
              ))}
            </>
          ) : null}
        </p>
      )}
    </section>
  )
}

function CandidatesBlock({
  title,
  items,
  overlapIds,
  ocrMarksByWine,
  ocrScoresByWine,
  ocrTextScoresByWine,
  ocrChannels,
  ocrQueryText = '',
  exclusiveLexicon = null,
  hsvStep = null,
  labelTextHr = null,
  sortKeys,
  embeddingLabel,
  cosByWine,
  sortBy,
  onSortByChange,
  winnerId = null,
  xgbWinnerId = null,
  similarIds,
  manualWinesId = null,
  onManualMatchChange,
  finalistRanks,
  finalMethod = 'fin1',
  finalOcrPrimary = null,
}: {
  title: string
  items: FindWineCandidate[]
  overlapIds: Set<number>
  ocrMarksByWine: Map<number, OcrMarkId[]>
  ocrScoresByWine: Map<number, Partial<Record<OcrScoreChannel, number>>>
  ocrTextScoresByWine: Map<number, Partial<Record<OcrScoreChannel, number>>>
  ocrChannels: OcrScoreChannel[]
  ocrQueryText?: string
  exclusiveLexicon?: ExclusiveLexiconStep | null
  hsvStep?: HsvStepInfo | null
  labelTextHr?: LabelTextHardRejectStep | null
  sortKeys: CandSortKey[]
  embeddingLabel: 'SigLIP2' | 'DINOv3'
  cosByWine: {
    siglip2: Map<number, number>
    dinov3: Map<number, number>
  }
  sortBy: CandSortKey
  onSortByChange: (key: CandSortKey) => void
  winnerId?: number | null
  xgbWinnerId?: number | null
  similarIds?: Set<number>
  manualWinesId?: number | null
  onManualMatchChange?: (wineId: number, checked: boolean) => void
  finalistRanks?: Map<number, number>
  finalMethod?: string
  finalOcrPrimary?: OcrScoreChannel | null
}) {
  const sortOptions = useMemo(() => {
    const allowed = new Set(sortKeys)
    return CAND_SORT_OPTIONS.filter((o) => allowed.has(o.key)).map((o) => {
      const ch = channelFromSortKey(o.key)
      if (ch) {
        return { key: o.key, label: finalByOcrSortLabel(ch, finalMethod) }
      }
      return o
    })
  }, [sortKeys, finalMethod])
  const effectiveSort = sortOptions.some((o) => o.key === sortBy)
    ? sortBy
    : 'cos'
  const sorted = useMemo(
    () =>
      sortCandidates(
        items,
        effectiveSort,
        ocrScoresByWine,
        ocrTextScoresByWine,
      ),
    [items, effectiveSort, ocrScoresByWine, ocrTextScoresByWine],
  )
  if (!items.length) return null
  const row1 = sorted.slice(0, 10)
  const row2 = sorted.slice(10, 20)
  const renderCard = (w: FindWineCandidate, key: string) => (
    <CandidateCard
      key={key}
      wine={w}
      overlap={overlapIds.has(w.id)}
      ocrMarks={ocrMarksByWine.get(w.id) || []}
      ocrScores={ocrScoresByWine.get(w.id)}
      ocrChannels={ocrChannels}
      ocrQueryText={ocrQueryText}
      exclusiveLexicon={exclusiveLexicon}
      hsvStep={hsvStep}
      labelTextHr={labelTextHr}
      embeddingLabel={embeddingLabel}
      cosSiglip2={cosByWine.siglip2.get(w.id) ?? null}
      cosDinov3={cosByWine.dinov3.get(w.id) ?? null}
      sortBy={effectiveSort}
      isFinalWinner={winnerId != null && w.id === winnerId}
      isXgbWinner={xgbWinnerId != null && w.id === xgbWinnerId}
      isSimilarBand={Boolean(similarIds?.has(w.id))}
      isManualMatch={manualWinesId != null && w.id === manualWinesId}
      manualMatchEnabled={Boolean(onManualMatchChange)}
      onManualMatchChange={onManualMatchChange}
      finalistRank={finalistRanks?.get(w.id) ?? null}
      finalMethod={finalMethod}
      finalOcrPrimary={finalOcrPrimary}
    />
  )
  return (
    <section className="cand-block">
      <div className="cand-block__head">
        <h2>{title}</h2>
        <div className="cand-sort" role="group" aria-label="Сортировка">
          <span className="cand-sort__label">Сортировка:</span>
          {sortOptions.map((opt) => (
            <button
              key={opt.key}
              type="button"
              className={`cand-sort__btn${
                effectiveSort === opt.key ? ' is-active' : ''
              }${
                opt.key === 'xgb' || opt.key === 'xgb_fin'
                  ? ' is-xgb'
                  : opt.key === 'crenc' || opt.key === 'crenc_fin'
                    ? ' is-crenc'
                    : opt.key === 'llm_txt'
                      ? ' is-llm-txt'
                      : ''
              }`}
              onClick={() => onSortByChange(opt.key)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
      <div className="cand-row">
        {row1.map((w) => renderCard(w, `r1-${w.id}`))}
      </div>
      {row2.length > 0 && (
        <div className="cand-row">
          {row2.map((w) => renderCard(w, `r2-${w.id}`))}
        </div>
      )}
    </section>
  )
}

function analogFlagClass(flag: AnalogMatchFlag | undefined): string {
  if (flag === true) return ' is-hit'
  if (flag === 'partial') return ' is-partial'
  if (flag === false) return ' is-miss'
  return ''
}

function AnalogCard({ item }: { item: WineAnalogItem }) {
  const href = `/wines/${encodeURIComponent(item.slug || String(item.id))}`
  const m = item.matched
  const title =
    item.criteria_count > 0
      ? `score ${item.score.toFixed(2)} · совпало ${item.matched_count} из ${item.criteria_count}`
      : `cos ${item.cosine != null ? item.cosine.toFixed(2) : item.score.toFixed(2)} · по embedding`
  return (
    <article
      className="cand-card analog-card"
      title={title}
    >
      <a
        className="analog-card__photo-link"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        title="Открыть вино в каталоге"
      >
        <div className="cand-card__media">
          {item.label_url ? (
            <img
              src={item.label_url}
              alt={item.name || `wine ${item.id}`}
              loading="lazy"
            />
          ) : (
            <div className="cand-card__placeholder" />
          )}
          <span className="analog-card__score">{Math.round(item.score * 100)}%</span>
        </div>
      </a>
      <div className="cand-card__meta analog-card__meta">
        <p className="cand-card__name">{item.name || `id ${item.id}`}</p>
        <ul className="analog-card__crit">
          {item.winery ? (
            <li className={`is-winery${analogFlagClass(m.winery)}`}>{item.winery}</li>
          ) : null}
          {item.category ? (
            <li className={`is-category${analogFlagClass(m.category)}`}>{item.category}</li>
          ) : null}
          {item.type ? (
            <li className={`is-type${analogFlagClass(m.type)}`}>{item.type}</li>
          ) : null}
          {item.grapes.length > 0 ? (
            <li className="is-grape">
              {item.grapes.map((g, i) => (
                <span key={`${g.name}-${i}`}>
                  {i > 0 ? ', ' : ''}
                  <span className={g.matched ? 'is-hit' : undefined}>{g.name}</span>
                </span>
              ))}
            </li>
          ) : null}
        </ul>
      </div>
    </article>
  )
}

function AnalogsPanel({ analogs }: { analogs: WineAnalogs }) {
  const c = analogs.criteria
  const chips: { key: string; label: string; values: string[] }[] = [
    { key: 'winery', label: 'Винодельня', values: c?.winery || [] },
    { key: 'category', label: 'Категория', values: c?.category || [] },
    { key: 'type', label: 'Тип', values: c?.type ? [c.type] : [] },
    { key: 'grape', label: 'Купаж', values: c?.grape || [] },
  ]
  return (
    <section className="scan-analogs" aria-label="Похожие вина">
      <header className="scan-analogs__head">
        <h2>Похожие вина</h2>
        <ul className="scan-analogs__criteria">
          {chips.map((ch) => (
            <li
              key={ch.key}
              className={`is-${ch.key}${ch.values.length ? ' is-found' : ' is-missing'}`}
            >
              <span>{ch.label}:</span> {ch.values.length ? ch.values.join(', ') : 'не найдено'}
            </li>
          ))}
        </ul>
      </header>
      {analogs.items.length > 0 ? (
        <div className="cand-row">
          {analogs.items.map((it) => (
            <AnalogCard key={it.id} item={it} />
          ))}
        </div>
      ) : (
        <p className="scan-analogs__empty">
          {analogs.ok === false
            ? `Поиск аналогов не удался: ${analogs.error || 'ошибка'}`
            : analogs.reason === 'no_criteria' ||
                analogs.reason === 'no_criteria_cosine_fallback'
              ? 'На этикетке не удалось выделить винодельню, категорию, тип или сорт'
              : 'Подходящих вин в каталоге нет'}
        </p>
      )}
    </section>
  )
}

function WinnerMatchPanel({
  wine,
  ocrQueryText,
  ocrScores,
  ocrChannels,
  ocrMarks,
  exclusiveLexicon,
  hsvStep = null,
  labelTextHr = null,
  cosSiglip2,
  cosDinov3,
  confidence,
  falsePositive,
  falseNegative,
  onEvalChange,
  xgbScore,
  xgbFin,
  isXgbWinner = false,
  crencScore,
  crencFin,
  fin1Score,
  finalMethod = 'fin1',
  finalOcrPrimary = null,
}: {
  wine: FindWineCandidate | null
  ocrQueryText: string
  ocrScores?: Partial<Record<OcrScoreChannel, number>>
  ocrChannels: OcrScoreChannel[]
  ocrMarks: OcrMarkId[]
  exclusiveLexicon: ExclusiveLexiconStep | null
  hsvStep?: HsvStepInfo | null
  labelTextHr?: LabelTextHardRejectStep | null
  cosSiglip2?: number | null
  cosDinov3?: number | null
  confidence?: number | null
  falsePositive: number
  falseNegative: number
  onEvalChange?: (next: {
    false_positive: number
    false_negative: number
  }) => void
  xgbScore?: number | null
  xgbFin?: number | null
  isXgbWinner?: boolean
  crencScore?: number | null
  crencFin?: number | null
  fin1Score?: number | null
  finalMethod?: string
  finalOcrPrimary?: OcrScoreChannel | null
}) {
  const catalogLabel = String(wine?.label || '').trim()
  const queryOcr = String(ocrQueryText || '').trim()
  const matchedTokens = useMemo(
    () => sharedMatchTokens(queryOcr, catalogLabel),
    [queryOcr, catalogLabel],
  )
  const exclusiveForms = useMemo(
    () => collectExclusiveLineForms(exclusiveLexicon, wine?.id ?? -1),
    [exclusiveLexicon, wine?.id],
  )
  const hardRejectReasons = useMemo(
    () =>
      wine
        ? hardRejectReasonsForWine(
            wine.id,
            exclusiveLexicon,
            hsvStep,
            wine.hsv,
            labelTextHr,
          )
        : [],
    [wine, exclusiveLexicon, hsvStep, labelTextHr],
  )
  const href = wine
    ? `/wines/${encodeURIComponent(wine.slug || String(wine.id))}`
    : null
  const fp = falsePositive ? 1 : 0
  const fn = falseNegative ? 1 : 0

  const mediaRef = useRef<HTMLDivElement>(null)
  const [hovered, setHovered] = useState(false)
  const [popupPos, setPopupPos] = useState<{ top: number; left: number } | null>(
    null,
  )
  const leaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

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
    if (!hovered || !wine) {
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
  }, [hovered, wine, placePopup])

  return (
    <aside className="scan-winner" aria-label="Результат поиска">
      <div className="scan-winner__card-col">
        <div className="scan-winner__card-row">
          {wine ? (
            <div
              className={`scan-winner__card cand-card is-final-winner${
                isXgbWinner ? ' is-xgb-winner' : ''
              }${hovered ? ' is-hovered' : ''}`}
            >
              {href ? (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="scan-winner__photo-link"
                  title="Открыть вино в каталоге"
                >
                  <div
                    ref={mediaRef}
                    className="cand-card__media"
                    onMouseEnter={openHover}
                    onMouseLeave={closeHoverSoon}
                  >
                    {wine.label_url ? (
                      <img
                        src={wine.label_url}
                        alt={wine.name || `wine ${wine.id}`}
                        loading="lazy"
                      />
                    ) : (
                      <div className="cand-card__placeholder" />
                    )}
                  </div>
                </a>
              ) : (
                <div
                  ref={mediaRef}
                  className="cand-card__media"
                  onMouseEnter={openHover}
                  onMouseLeave={closeHoverSoon}
                >
                  {wine.label_url ? (
                    <img
                      src={wine.label_url}
                      alt={wine.name || `wine ${wine.id}`}
                      loading="lazy"
                    />
                  ) : (
                    <div className="cand-card__placeholder" />
                  )}
                </div>
              )}
              <div className="cand-card__meta">
                <p className="cand-card__id">id {wine.id}</p>
                <p className="cand-card__name">{wine.name || '—'}</p>
                <p className="scan-winner__winery">{wine.winery || '—'}</p>
              </div>
            </div>
          ) : (
            <div className="scan-winner__empty">
              <strong>Совпадение не найдено</strong>
              <span>FinalScore / выбранный метод не выбрали вино</span>
            </div>
          )}
          {onEvalChange && (
            <div
              className="scan-winner__eval"
              onClick={(e) => e.stopPropagation()}
            >
              <label className="scan-winner__check">
                <input
                  type="checkbox"
                  checked={fp === 1}
                  onChange={() => {
                    if (fp === 1)
                      onEvalChange({ false_positive: 0, false_negative: 0 })
                    else
                      onEvalChange({ false_positive: 1, false_negative: 0 })
                  }}
                />
                False positive
              </label>
              <label className="scan-winner__check">
                <input
                  type="checkbox"
                  checked={fn === 1}
                  onChange={() => {
                    if (fn === 1)
                      onEvalChange({ false_positive: 0, false_negative: 0 })
                    else
                      onEvalChange({ false_positive: 0, false_negative: 1 })
                  }}
                />
                False negative
              </label>
            </div>
          )}
        </div>
        {wine && hovered && popupPos && (
          <WineHoverPopup
            wine={wine}
            pos={popupPos}
            cosSiglip2={cosSiglip2}
            cosDinov3={cosDinov3}
            ocrScores={ocrScores}
            ocrChannels={ocrChannels}
            ocrQueryText={ocrQueryText}
            ocrMarks={ocrMarks}
            exclusiveLexicon={exclusiveLexicon}
            hsvStep={hsvStep}
            labelTextHr={labelTextHr}
            finalMethod={finalMethod}
            onMouseEnter={openHover}
            onMouseLeave={closeHoverSoon}
          />
        )}
        <div className="scan-winner__scores">
          <p>
            <span>cos SigLIP2</span>
            <em>{formatScore00(cosSiglip2)}</em>
          </p>
          <p>
            <span>cos DINOv3</span>
            <em>{formatScore00(cosDinov3)}</em>
          </p>
          {wine?.hsv != null && (
            <p
              className="scan-winner__hsv"
              title="Расстояние Бхаттачарии между HSV-гистограммами. 0 — одна гамма, 1 — гаммы не пересекаются."
            >
              <span>HSV</span>
              <em>{formatScore00(wine.hsv)}</em>
            </p>
          )}
          {wine?.color_delta != null && (
            <p
              className="scan-winner__color-delta"
              title="CIEDE2000 доминантных цветов искомой этикетки и кандидата. Меньше — ближе по цвету."
            >
              <span>ColorDelta</span>
              <em>{Number(wine.color_delta).toFixed(2)}</em>
            </p>
          )}
          {hardRejectReasons.length > 0 && (
            <div className="cand-popup__hard-reject" role="status">
              <strong>Hard reject</strong>
              <ul>
                {hardRejectReasons.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}
          <p
            title={
              'fin1 = w_ocr×TextScore + w_emb×Cosine; hard mismatch / exclusive → 0'
            }
          >
            <span>fin1</span>
            <em>{formatScore00(fin1Score ?? confidence)}</em>
          </p>
          {(xgbScore != null || xgbFin != null) && (
            <>
              <p className="scan-winner__xgb" title="XGBoost P(match) 0…1">
                <span>XGB</span>
                <em>{formatScore00(xgbScore)}</em>
              </p>
              <p
                className="scan-winner__xgb"
                title="XGB_fin = w_ocr×XGB + w_emb×Cosine (match/similar); иначе 0"
              >
                <span>XGB_fin</span>
                <em>{formatScore00(xgbFin)}</em>
              </p>
            </>
          )}
          {(crencScore != null || crencFin != null) && (
            <>
              <p className="scan-winner__crenc" title="Cross Encoder P(match) 0…1">
                <span>CrEnc</span>
                <em>{formatScore00(crencScore)}</em>
              </p>
              <p
                className="scan-winner__crenc"
                title="CrEnc_fin = w_ocr×CrEnc + w_emb×Cosine (match/similar); иначе 0"
              >
                <span>CrEnc_fin</span>
                <em>{formatScore00(crencFin)}</em>
              </p>
            </>
          )}
          {wine &&
            visibleOcrFinChannels(
              wine,
              finalMethod,
              ocrChannels,
              ocrScores,
            ).map((ch) => (
            <p
              key={`ws-${ch}`}
              className={finalOcrPrimary === ch ? 'is-final-ocr' : undefined}
              title={
                finalOcrPrimary === ch
                  ? 'Final OCR — канал решения matched_wine'
                  : `${finalByOcrRowLabel(finalMethod)} для OCR ${ocrChannelLetter(ch)}`
              }
            >
              <span>
                <em
                  className={
                    ocrMarks.includes(ch) || finalOcrPrimary === ch
                      ? `scan-winner__ch is-ocr-${ch} is-best-channel`
                      : `scan-winner__ch is-ocr-${ch}`
                  }
                >
                  {ocrChannelLetter(ch)}
                </em>{' '}
                {finalByOcrRowLabel(finalMethod)}
                {finalOcrPrimary === ch ? ' ★' : ''}
              </span>
              <em>{formatScore00(ocrScores?.[ch])}</em>
            </p>
          ))}
        </div>
      </div>
      <div className="scan-winner__texts">
        <div className="cand-popup__label-cols">
          <div className="cand-popup__label-text">
            <strong>OCR искомого</strong>
            <pre>
              {highlightExclusiveLines(
                queryOcr,
                exclusiveForms.query,
                matchedTokens,
              )}
            </pre>
          </div>
          <div className="cand-popup__label-text">
            <strong>Текст этикетки каталога</strong>
            <pre>
              {highlightExclusiveLines(
                catalogLabel,
                exclusiveForms.catalog,
                matchedTokens,
              )}
            </pre>
          </div>
        </div>
        <div className="scan-winner__legend" aria-hidden>
          <span className="is-ex-category">категория</span>
          <span className="is-ex-type">тип</span>
          <span className="is-ex-grape">сорт</span>
          <span className="is-ex-winery">винодельня</span>
        </div>
      </div>
    </aside>
  )
}

export function HomePage() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [searchParams, setSearchParams] = useSearchParams()
  const [dragging, setDragging] = useState(false)
  const [scanSourceOpen, setScanSourceOpen] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [result, setResult] = useState<FindWineResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [loadingScan, setLoadingScan] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [candSortBy, setCandSortBy] = useState<CandSortKey>('cos')
  const [isMobile, setIsMobile] = useState(false)
  const loadedIdRef = useRef<number | null>(null)
  const loadGenRef = useRef(0)
  const searchGenRef = useRef(0)
  const autoSearchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  /** true while viewing a saved scan from ?scanid= (no auto-search on hydrate). */
  const viewingScanRef = useRef(false)
  /** Blocks ?scanid= hydrate while POST /api/findwine is in flight. */
  const searchingRef = useRef(false)

  useEffect(() => {
    setCandSortBy(readCandSortCookie())
  }, [])

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 720px)')
    const apply = () => setIsMobile(mq.matches)
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [])

  const changeCandSort = useCallback((key: CandSortKey) => {
    setCandSortBy(key)
    writeCandSortCookie(key)
  }, [])

  const openPicker = useCallback((mode: 'camera' | 'file' = 'file') => {
    const input = inputRef.current
    if (!input) return
    setScanSourceOpen(false)
    if (mode === 'camera') input.setAttribute('capture', 'environment')
    else input.removeAttribute('capture')
    input.value = ''
    input.click()
  }, [])

  useEffect(() => {
    if (!scanSourceOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setScanSourceOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [scanSourceOpen])

  const syncPreviewFromResult = useCallback((data: FindWineResult) => {
    const workUrl = workPhotoUrl(data)
    const next =
      workUrl ||
      (data.label_url
        ? searchPhotoSrc(
            data.label_url,
            data.search_photos_id ?? data.status?.search_photos_id,
          ) || null
        : null)
    if (!next) return
    setPreview((prev) => {
      if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
      return next
    })
  }, [])

  const applyResult = useCallback((data: FindWineResult) => {
    searchingRef.current = false
    setResult(data)
    syncPreviewFromResult(data)
    const st = data.status
    if (st?.ok) {
      const ocrText = st.steps?.ocr?.text
      setStatus(
        ocrText
          ? `Готово. OCR: ${String(ocrText).slice(0, 120)}`
          : 'Готово. Обработка завершена.',
      )
    } else {
      setStatus(st?.error || 'Обработка завершилась с ошибкой')
    }
    if (data.search_photos_id != null) {
      viewingScanRef.current = true
      // Mark before URL push so ?scanid= effect does not refetch / cancel search
      loadedIdRef.current = data.search_photos_id
      // push — Back/Forward листает предыдущие сканы по ?scanid=
      setSearchParams({ scanid: String(data.search_photos_id) })
    }
  }, [setSearchParams, syncPreviewFromResult])

  useEffect(() => {
    const raw = searchParams.get('scanid')
    if (!raw) {
      viewingScanRef.current = false
      return
    }
    const id = Number(raw)
    if (!Number.isFinite(id) || id <= 0) return
    // Already showing this scan (e.g. just applied from findwine) — do not refetch
    // and do not bump searchGen (would drop an in-flight / just-finished search).
    if (loadedIdRef.current === id) return
    if (searchingRef.current) return

    // Открыли сохранённый скан — не гоняем findwine / auto-search
    viewingScanRef.current = true
    if (autoSearchTimerRef.current) {
      clearTimeout(autoSearchTimerRef.current)
      autoSearchTimerRef.current = null
    }
    // Сбросить in-flight POST /api/findwine, если был
    searchGenRef.current += 1

    let cancelled = false
    const gen = ++loadGenRef.current
    setLoadingScan(true)
    setBusy(false)
    setStatus('Загрузка результата…')
    fetchFindWineResult(id)
      .then((data) => {
        if (cancelled || loadGenRef.current !== gen) return
        if (searchingRef.current) return
        loadedIdRef.current = id
        viewingScanRef.current = true
        setResult(data)
        setFileName(`search #${id}`)
        // Показать UI сразу — File для «Поиск» догружаем в фоне
        setLoadingScan(false)
        // Дропзона: исходный work JPEG, не _crops с рамками
        const workUrl = workPhotoUrl(data)
        if (workUrl) {
          setPreview((prev) => {
            if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
            return workUrl
          })
        } else if (data.label_url) {
          const labelSrc =
            searchPhotoSrc(
              data.label_url,
              data.search_photos_id ?? data.status?.search_photos_id,
            ) || data.label_url
          setPreview((prev) => {
            if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
            return labelSrc
          })
        }
        const st = data.status
        setStatus(
          st?.ok
            ? 'Результат загружен по ссылке'
            : st?.error || 'Результат загружен (с ошибками)',
        )
        void fileFromFindWineResult(data)
          .then((hydrated) => {
            if (
              !cancelled &&
              loadGenRef.current === gen &&
              !searchingRef.current &&
              hydrated
            ) {
              setFile(hydrated)
            }
          })
          .catch(() => {
            /* кнопка останется disabled — нужен новый файл */
          })
      })
      .catch((err) => {
        if (cancelled || loadGenRef.current !== gen) return
        setStatus(err instanceof Error ? err.message : 'Не удалось загрузить результат')
        setLoadingScan(false)
      })
      .finally(() => {
        if (loadGenRef.current === gen) setLoadingScan(false)
      })
    return () => {
      cancelled = true
    }
  }, [searchParams])

  const pickFile = useCallback(async (next: File | null) => {
    if (!next) return
    if (autoSearchTimerRef.current) {
      clearTimeout(autoSearchTimerRef.current)
      autoSearchTimerRef.current = null
    }
    viewingScanRef.current = false
    searchingRef.current = false
    // Отменить in-flight findwine / загрузку ?scanid= — иначе старый ответ
    // перезапишет UI при уже новом превью.
    searchGenRef.current += 1
    loadGenRef.current += 1
    setLoadingScan(false)
    setBusy(false)
    setFile(next)
    setFileName(next.name)
    setStatus(null)
    setResult(null)
    loadedIdRef.current = null
    setSearchParams({}, { replace: true })
    setPreview((prev) => {
      if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev)
      return null
    })
    try {
      const url = await makePreviewUrl(next)
      setPreview(url)
    } catch (err) {
      setPreview(null)
      setStatus(
        err instanceof Error
          ? err.message
          : 'Превью недоступно — файл всё равно можно отправить в поиск',
      )
    }
    autoSearchTimerRef.current = setTimeout(() => {
      autoSearchTimerRef.current = null
      // Не стартовать, если за это время открыли ?scanid=
      if (viewingScanRef.current) return
      void runSearchRef.current?.(next)
    }, 200)
  }, [setSearchParams])

  const fileFromClipboard = useCallback((data: DataTransfer | null): File | null => {
    if (!data) return null
    const items = data.items
    if (items) {
      for (const item of items) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
          const f = item.getAsFile()
          if (f) {
            const ext = (f.type.split('/')[1] || 'png').replace('jpeg', 'jpg')
            return f.name && f.name !== 'image.png'
              ? f
              : new File([f], `paste-${Date.now()}.${ext}`, {
                  type: f.type || 'image/png',
                })
          }
        }
      }
    }
    const files = data.files
    if (files?.length) {
      const f = files[0]
      if (f && f.type.startsWith('image/')) return f
    }
    return null
  }, [])

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      if (busy || loadingScan) return
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable)
      ) {
        return
      }
      const next = fileFromClipboard(e.clipboardData)
      if (!next) return
      e.preventDefault()
      void pickFile(next)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [busy, loadingScan, fileFromClipboard, pickFile])

  const runSearch = useCallback(async (overrideFile?: File) => {
    const target = overrideFile ?? file
    if (!target) return
    if (autoSearchTimerRef.current) {
      clearTimeout(autoSearchTimerRef.current)
      autoSearchTimerRef.current = null
    }
    viewingScanRef.current = false
    searchingRef.current = true
    // Не подтягивать старый ?scanid= поверх нового поиска
    loadGenRef.current += 1
    setLoadingScan(false)
    const gen = ++searchGenRef.current
    setBusy(true)
    setStatus('Поиск…')
    setResult(null)
    loadedIdRef.current = null
    try {
      const data = await findWine(target)
      if (searchGenRef.current !== gen) return
      applyResult(data)
    } catch (err) {
      if (searchGenRef.current !== gen) return
      searchingRef.current = false
      setStatus(err instanceof Error ? err.message : 'Ошибка поиска')
    } finally {
      if (searchGenRef.current === gen) setBusy(false)
    }
  }, [file, applyResult])

  const runSearchRef = useRef(runSearch)
  runSearchRef.current = runSearch

  useEffect(() => {
    return () => {
      if (autoSearchTimerRef.current) clearTimeout(autoSearchTimerRef.current)
    }
  }, [])

  const uiBusy = busy || loadingScan
  const dropzoneBusyLabel = loadingScan
    ? {
        title: 'Загрузка результата…',
        hint: 'подтягиваем сохранённый скан из базы',
      }
    : {
        title: 'Идёт поиск…',
        hint: 'распознаём этикетку и сверяем с каталогом',
      }

  const timings = result?.status?.timings_ms
  const totalMs = Number(timings?.total || 0)
  const analogs = (result?.status?.analogs as WineAnalogs | undefined) ?? null
  const algoVersion =
    result?.algorithm_version ||
    result?.status?.algorithm_version ||
    null
  const algoUpdated =
    result?.algorithm_updated_at ||
    result?.status?.algorithm_updated_at ||
    null
  const siglip = result?.candidates_siglip2 ?? []
  const dinov3 = result?.candidates_dinov3 ?? []

  const overlapIds = useMemo(() => {
    const a = new Set(siglip.map((x) => x.id))
    const both = new Set<number>()
    for (const w of dinov3) {
      if (a.has(w.id)) both.add(w.id)
    }
    return both
  }, [siglip, dinov3])

  const ocrMarksByWine = useMemo(
    () => ocrBestWineMarks(collectOcrFinalBests(result)),
    [result],
  )
  const ocrScoresByWine = useMemo(
    () => collectOcrScoresByWine(result),
    [result],
  )
  const ocrTextScoresByWine = useMemo(
    () => collectOcrTextScoresByWine(result),
    [result],
  )
  const ocrQueryText = useMemo(() => collectQueryOcrText(result), [result])
  const exclusiveLexicon = useMemo((): ExclusiveLexiconStep | null => {
    const step = (result?.status?.steps as Record<string, unknown> | undefined)
      ?.exclusive_lexicon
    return step && typeof step === 'object'
      ? (step as ExclusiveLexiconStep)
      : null
  }, [result])
  const hsvStep = useMemo((): HsvStepInfo | null => {
    const step = (result?.status?.steps as Record<string, unknown> | undefined)
      ?.hsv
    return step && typeof step === 'object' ? (step as HsvStepInfo) : null
  }, [result])
  const labelTextHr = useMemo((): LabelTextHardRejectStep | null => {
    const step = (result?.status?.steps as Record<string, unknown> | undefined)
      ?.label_text_hard_reject
    return step && typeof step === 'object'
      ? (step as LabelTextHardRejectStep)
      : null
  }, [result])
  const ocrChannels = useMemo(() => activeOcrScoreChannels(result), [result])
  const finalMethod = useMemo(() => finalScoreMethodOf(result), [result])
  const finalOcrPrimary = useMemo(
    () => finalOcrPrimaryChannel(finalOcrPrimaryOf(result)),
    [result],
  )
  const finalWinnerId = useMemo(() => {
    const fromStatus = result?.status?.matched_wine_id
    if (typeof fromStatus === 'number') return fromStatus
    const fromRoot = result?.matched_wine_id
    if (typeof fromRoot === 'number') return fromRoot
    return null
  }, [result])
  const finalistRanks = useMemo(
    () => collectFinalistRanks(result),
    [result],
  )
  const similarWineIds = useMemo(() => {
    const raw = result?.status?.similar_wine_ids
    const set = new Set<number>()
    if (Array.isArray(raw)) {
      for (const id of raw) {
        const n = Number(id)
        if (Number.isFinite(n)) set.add(n)
      }
    }
    return set
  }, [result])
  const xgbWinnerId = useMemo(() => {
    const fromStatus = result?.status?.xgb_best_wine_id
    if (typeof fromStatus === 'number') return fromStatus
    const best = result?.status?.steps?.xgb_match?.best_xgb_fin
    if (best && typeof best === 'object' && typeof best.id === 'number') {
      return best.id as number
    }
    return null
  }, [result])
  const xgbScoresByWine = useMemo(() => {
    const map = new Map<
      number,
      { xgb: number | null; fin: number | null }
    >()
    const byId = result?.status?.steps?.xgb_match?.by_id
    if (byId && typeof byId === 'object') {
      for (const [k, v] of Object.entries(byId as Record<string, any>)) {
        const id = Number(k)
        if (!Number.isFinite(id) || !v || typeof v !== 'object') continue
        map.set(id, {
          xgb: typeof v.xgb_score === 'number' ? v.xgb_score : null,
          fin: typeof v.xgb_fin === 'number' ? v.xgb_fin : null,
        })
      }
    }
    return map
  }, [result])
  const crencScoresByWine = useMemo(() => {
    const map = new Map<
      number,
      { score: number | null; fin: number | null }
    >()
    const byId = result?.status?.steps?.crenc_match?.by_id
    if (byId && typeof byId === 'object') {
      for (const [k, v] of Object.entries(byId as Record<string, any>)) {
        const id = Number(k)
        if (!Number.isFinite(id) || !v || typeof v !== 'object') continue
        map.set(id, {
          score: typeof v.crenc_score === 'number' ? v.crenc_score : null,
          fin: typeof v.crenc_fin === 'number' ? v.crenc_fin : null,
        })
      }
    }
    return map
  }, [result])
  const winnerFin1 = useMemo(() => {
    if (finalWinnerId == null || !result) return null
    const matched = result.status?.matched_wine as
      | { source?: string; final_score?: number }
      | undefined
    if (
      matched?.source === 'final_score' &&
      typeof matched.final_score === 'number'
    ) {
      return matched.final_score
    }
    const help = result.status?.steps?.ocr_wine_id as
      | {
          final_ranked?: { id?: number; final_score?: number }[]
          best_final?: { id?: number; final_score?: number }
        }
      | undefined
    const ranked = help?.final_ranked
    if (Array.isArray(ranked)) {
      const row = ranked.find((r) => r?.id === finalWinnerId)
      if (row && typeof row.final_score === 'number') return row.final_score
    }
    const bf = help?.best_final
    if (bf?.id === finalWinnerId && typeof bf.final_score === 'number') {
      return bf.final_score
    }
    if (!matched?.source || matched.source === 'final_score') {
      const c =
        result.matched_wine_confidence ??
        (typeof result.status?.matched_wine_confidence === 'number'
          ? result.status.matched_wine_confidence
          : null)
      if (typeof c === 'number') return c
    }
    return null
  }, [result, finalWinnerId])
  const sortKeysSiglip = useMemo(
    () => activeCandSortKeys(result, 'siglip2'),
    [result],
  )
  const sortKeysDinov3 = useMemo(
    () => activeCandSortKeys(result, 'dinov3'),
    [result],
  )

  useEffect(() => {
    if (!result) return
    const allowed = new Set<CandSortKey>([
      ...sortKeysSiglip,
      ...sortKeysDinov3,
    ])
    if (!allowed.has(candSortBy)) {
      setCandSortBy('cos')
    }
  }, [result, sortKeysSiglip, sortKeysDinov3, candSortBy])

  const cosByWine = useMemo(
    () => ({
      siglip2: new Map(siglip.map((w) => [w.id, w.cosine_similarity])),
      dinov3: new Map(dinov3.map((w) => [w.id, w.cosine_similarity])),
    }),
    [siglip, dinov3],
  )
  const evalFlags = useMemo(() => {
    const raw = (result?.status as { eval?: unknown } | undefined)?.eval
    if (!raw || typeof raw !== 'object') return { fp: 0, fn: 0 }
    const ev = raw as { false_positive?: unknown; false_negative?: unknown }
    const fp = Number(ev.false_positive) ? 1 : 0
    const fn = Number(ev.false_negative) ? 1 : 0
    if (fp && fn) return { fp: 1, fn: 0 }
    return { fp, fn }
  }, [result])

  const setScanEval = useCallback(
    async (next: { false_positive: number; false_negative: number }) => {
      const scanId =
        result?.search_photos_id ??
        (typeof result?.status?.search_photos_id === 'number'
          ? result.status.search_photos_id
          : null)
      if (scanId == null) return
      const fp = next.false_positive ? 1 : 0
      const fn = next.false_negative ? 1 : 0
      setResult((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          status: {
            ...prev.status,
            eval: { false_positive: fp, false_negative: fn },
          },
        }
      })
      try {
        const saved = await updateScanHistoryEval(scanId, {
          false_positive: fp,
          false_negative: fn,
        })
        setResult((prev) => {
          if (!prev) return prev
          return {
            ...prev,
            status: {
              ...prev.status,
              eval: {
                false_positive: saved.false_positive ? 1 : 0,
                false_negative: saved.false_negative ? 1 : 0,
              },
            },
          }
        })
      } catch (e) {
        setStatus(e instanceof Error ? e.message : 'Ошибка сохранения Eval')
      }
    },
    [result?.search_photos_id, result?.status?.search_photos_id],
  )

  const manualWinesId = useMemo(() => {
    const v = result?.manual_wines_id
    return typeof v === 'number' && Number.isFinite(v) ? v : null
  }, [result?.manual_wines_id])

  const setManualMatch = useCallback(
    async (wineId: number, checked: boolean) => {
      const scanId =
        result?.search_photos_id ??
        (typeof result?.status?.search_photos_id === 'number'
          ? result.status.search_photos_id
          : null)
      if (scanId == null) return
      const nextId = checked ? wineId : null
      const prevId = result?.manual_wines_id ?? null
      setResult((prev) => (prev ? { ...prev, manual_wines_id: nextId } : prev))
      try {
        const saved = await updateFindwineManualWine(scanId, nextId)
        setResult((prev) =>
          prev
            ? { ...prev, manual_wines_id: saved.manual_wines_id ?? null }
            : prev,
        )
      } catch (e) {
        setResult((prev) =>
          prev ? { ...prev, manual_wines_id: prevId } : prev,
        )
        setStatus(
          e instanceof Error ? e.message : 'Ошибка сохранения соответствия',
        )
      }
    },
    [
      result?.search_photos_id,
      result?.status?.search_photos_id,
      result?.manual_wines_id,
    ],
  )

  const wineById = useMemo(() => {
    const map = new Map<number, FindWineCandidate>()
    for (const w of [...siglip, ...dinov3]) {
      const prev = map.get(w.id)
      if (!prev) {
        map.set(w.id, w)
        continue
      }
      // Prefer richer geometry / keep higher geometry_score
      const prevScore = Number(prev.geometry_score || 0)
      const nextScore = Number(w.geometry_score || 0)
      map.set(w.id, nextScore >= prevScore ? { ...prev, ...w } : { ...w, ...prev })
    }
    return map
  }, [siglip, dinov3])

  const showMobileLanding =
    isMobile && !preview && !result && !uiBusy && !fileName

  return (
    <main
      className={`page home-page${isMobile ? ' home-page--mobile' : ''}${
        showMobileLanding ? ' home-page--mobile-landing' : ''
      }`}
    >
      <ScanSettingsGear onClick={() => setSettingsOpen(true)} />
      <ScanSettingsPopup open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {!isMobile && (
        <div className="page__intro">
          <h1>Сканер вина</h1>
          <p className="page__lead">
            Загрузите фото этикетки — распознаем вино из каталога российских производителей
          </p>
        </div>
      )}

      {showMobileLanding && (
        <section className="mobile-scan-landing">
          <h1 className="mobile-scan-landing__title">Свои вина</h1>
          <p className="mobile-scan-landing__lead">
            Сфотографируйте этикетку Российского вина или загрузите фото, чтобы найти
            его
          </p>
          <div className="mobile-scan-card">
            <div className="mobile-scan-card__art" aria-hidden>
              <img
                className="mobile-scan-card__img"
                src="/scan-bottle.png?v=3"
                alt=""
                width={296}
                height={235}
              />
            </div>
            <button
              type="button"
              className="mobile-scan-card__scan"
              disabled={uiBusy}
              aria-label="Сфотографировать или загрузить этикетку"
              aria-haspopup="dialog"
              aria-expanded={scanSourceOpen}
              onClick={() => {
                if (uiBusy) return
                setScanSourceOpen(true)
              }}
            >
              <svg
                viewBox="0 0 24 24"
                width="28"
                height="28"
                aria-hidden
                focusable="false"
              >
                <path
                  fill="currentColor"
                  d="M9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"
                />
              </svg>
            </button>
          </div>
        </section>
      )}

      {scanSourceOpen &&
        createPortal(
          <div
            className="scan-source-overlay"
            role="presentation"
            onClick={() => setScanSourceOpen(false)}
          >
            <div
              className="scan-source-sheet"
              role="dialog"
              aria-modal="true"
              aria-label="Источник фото"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                type="button"
                className="scan-source-sheet__item"
                onClick={() => openPicker('camera')}
              >
                <span className="scan-source-sheet__icon" aria-hidden>
                  <svg viewBox="0 0 24 24" width="28" height="28">
                    <path
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                      d="M9 3.5 7.4 5.5H4.5A1.5 1.5 0 0 0 3 7v11.5A1.5 1.5 0 0 0 4.5 20h15a1.5 1.5 0 0 0 1.5-1.5V7a1.5 1.5 0 0 0-1.5-1.5H16.6L15 3.5H9Z"
                    />
                    <circle
                      cx="12"
                      cy="12.5"
                      r="3.6"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                    />
                  </svg>
                </span>
                <span className="scan-source-sheet__text">
                  <strong>Сделать фото</strong>
                </span>
              </button>
              <button
                type="button"
                className="scan-source-sheet__item"
                onClick={() => openPicker('file')}
              >
                <span className="scan-source-sheet__icon" aria-hidden>
                  <svg viewBox="0 0 24 24" width="28" height="28">
                    <rect
                      x="3.5"
                      y="5"
                      width="17"
                      height="14"
                      rx="2"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                    />
                    <path
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinejoin="round"
                      d="m5.5 16 4.2-4.2 2.8 2.8 3-3.5L18.5 16"
                    />
                    <circle cx="8.2" cy="9.2" r="1.2" fill="currentColor" />
                    <path
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.7"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M17.2 14.2v4.6M15 16.5l2.2-2.3 2.2 2.3"
                    />
                  </svg>
                </span>
                <span className="scan-source-sheet__text">
                  <strong>Загрузить изображение</strong>
                  <em>JPG, PNG, HEIC, WEBP не более 10 Мб</em>
                </span>
              </button>
            </div>
          </div>,
          document.body,
        )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*,.heic,.heif,.HEIC,.HEIF,.webp,.avif,.tif,.tiff,.bmp,.gif,.jfif,.jp2"
        hidden
        onChange={(e) => void pickFile(e.target.files?.[0] ?? null)}
      />

      <div
        className={`scan-hero${
          result && !uiBusy ? ' scan-hero--with-winner' : ''
        }${showMobileLanding ? ' is-hidden-mobile' : ''}`}
      >
      <div
        className={`dropzone ${dragging ? 'is-dragging' : ''} ${uiBusy ? 'is-busy' : ''} ${preview ? 'has-preview' : ''}${
          result && !uiBusy && (result.crops_url || result.label_url)
            ? ' has-thumbs'
            : ''
        }`}
        onDragEnter={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (uiBusy) return
          void pickFile(e.dataTransfer.files?.[0] ?? null)
        }}
        onClick={() => {
          if (uiBusy) return
          openPicker(isMobile ? 'camera' : 'file')
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (uiBusy) return
          if (e.key === 'Enter' || e.key === ' ') openPicker(isMobile ? 'camera' : 'file')
        }}
      >
        <div className="dropzone__main">
          {preview ? (
            <>
              <img src={preview} alt="Превью этикетки" className="dropzone__preview" />
              <ImageFileMeta src={preview} file={file} />
            </>
          ) : fileName ? (
            <div className="dropzone__hint">
              <div className="dropzone__icon" aria-hidden>
                ⌕
              </div>
              <strong className="dropzone__hint-text">{fileName}</strong>
              <span className="dropzone__hint-text">Готовим превью…</span>
            </div>
          ) : (
            <div className="dropzone__hint">
              <div className="dropzone__icon dropzone__icon--desktop" aria-hidden>
                ⌕
              </div>
              <strong className="dropzone__hint-text">
                Выберите файл или перетащите фото сюда или вставьте изображение из
                буфера (Ctrl+V / ⌘V)
              </strong>
              <span className="dropzone__hint-text">
                jpg, png, webp, heic…
              </span>
            </div>
          )}
        </div>

        {result && !uiBusy && (result.crops_url || result.label_url) && (
          <aside
            className="dropzone__thumbs"
            aria-label="Кроп и этикетка"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.stopPropagation()}
          >
            {result.crops_url && (
              <DropzoneThumb
                caption={cropsFigcaption(result)}
                alt={cropsFigcaption(result)}
                src={
                  searchPhotoSrc(
                    result.crops_url,
                    result.search_photos_id ?? result.status?.search_photos_id,
                  )!
                }
              />
            )}
            {result.label_url && (
              <DropzoneThumb
                caption="Этикетка"
                alt="Кроп этикетки"
                src={
                  searchPhotoSrc(
                    result.label_url,
                    result.search_photos_id ?? result.status?.search_photos_id,
                  )!
                }
              />
            )}
          </aside>
        )}

        {uiBusy && (
          <div className="dropzone__searching" aria-live="polite" aria-busy="true">
            <span className="search-spinner" aria-hidden />
            <strong>{dropzoneBusyLabel.title}</strong>
            <span>{dropzoneBusyLabel.hint}</span>
          </div>
        )}
      </div>

      {result && !uiBusy && (
        <WinnerMatchPanel
          wine={finalWinnerId != null ? wineById.get(finalWinnerId) ?? null : null}
          ocrQueryText={ocrQueryText}
          ocrScores={
            finalWinnerId != null
              ? ocrScoresByWine.get(finalWinnerId)
              : undefined
          }
          ocrChannels={ocrChannels}
          ocrMarks={
            finalWinnerId != null
              ? ocrMarksByWine.get(finalWinnerId) || []
              : []
          }
          exclusiveLexicon={exclusiveLexicon}
          hsvStep={hsvStep}
          labelTextHr={labelTextHr}
          cosSiglip2={
            finalWinnerId != null
              ? cosByWine.siglip2.get(finalWinnerId) ?? null
              : null
          }
          cosDinov3={
            finalWinnerId != null
              ? cosByWine.dinov3.get(finalWinnerId) ?? null
              : null
          }
          confidence={
            result.matched_wine_confidence ??
            (typeof result.status?.matched_wine_confidence === 'number'
              ? result.status.matched_wine_confidence
              : null)
          }
          falsePositive={evalFlags.fp}
          falseNegative={evalFlags.fn}
          onEvalChange={
            (result.search_photos_id ??
              (typeof result.status?.search_photos_id === 'number'
                ? result.status.search_photos_id
                : null)) != null
              ? setScanEval
              : undefined
          }
          xgbScore={
            finalWinnerId != null
              ? xgbScoresByWine.get(finalWinnerId)?.xgb ??
                wineById.get(finalWinnerId)?.xgb_score ??
                null
              : null
          }
          xgbFin={
            finalWinnerId != null
              ? xgbScoresByWine.get(finalWinnerId)?.fin ??
                wineById.get(finalWinnerId)?.xgb_fin ??
                null
              : null
          }
          isXgbWinner={
            finalWinnerId != null &&
            xgbWinnerId != null &&
            finalWinnerId === xgbWinnerId
          }
          crencScore={
            finalWinnerId != null
              ? crencScoresByWine.get(finalWinnerId)?.score ??
                wineById.get(finalWinnerId)?.crenc_score ??
                null
              : null
          }
          crencFin={
            finalWinnerId != null
              ? crencScoresByWine.get(finalWinnerId)?.fin ??
                wineById.get(finalWinnerId)?.crenc_fin ??
                null
              : null
          }
          fin1Score={winnerFin1}
          finalMethod={finalMethod}
          finalOcrPrimary={finalOcrPrimary}
        />
      )}
      </div>

      {result && !uiBusy && finalWinnerId == null && analogs && (
        <AnalogsPanel analogs={analogs} />
      )}

      {preview && (
        <p className="dropzone__change-hint">
          Нажмите на фото, чтобы выбрать другое · вставьте (Ctrl+V) ·{' '}
          <button type="button" className="linkish" onClick={() => openPicker('file')}>
            выбрать файл
          </button>
        </p>
      )}

      <div className="scan-actions">
        <button
          type="button"
          className="scan-search-btn"
          disabled={uiBusy || (!isMobile && !file)}
          aria-label={isMobile ? 'Сфотографировать этикетку' : 'Поиск'}
          onClick={(e) => {
            e.stopPropagation()
            if (uiBusy) return
            if (isMobile) {
              openPicker('camera')
              return
            }
            void runSearch()
          }}
        >
          <span className="scan-search-btn__desktop">
            {busy ? (
              <>
                <span className="search-spinner search-spinner--btn" aria-hidden />
                Поиск…
              </>
            ) : loadingScan ? (
              <>
                <span className="search-spinner search-spinner--btn" aria-hidden />
                Загрузка…
              </>
            ) : (
              'Поиск'
            )}
          </span>
          <span className="scan-search-btn__camera" aria-hidden>
            {uiBusy ? (
              <span className="search-spinner search-spinner--btn" />
            ) : (
              <svg viewBox="0 0 24 24" width="36" height="36" focusable="false">
                <path
                  fill="currentColor"
                  d="M9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"
                />
              </svg>
            )}
          </span>
        </button>
      </div>

      {(timings) && (
        <div className="scan-results scan-results--triple">
          <TimingsGantt
              timings={timings as Record<string, unknown>}
              starts={
                (result?.status?.timings_start_ms as
                  | Record<string, unknown>
                  | undefined) || null
              }
              totalMs={totalMs}
              extraRows={
                typeof analogs?.ms === 'number'
                  ? [{ key: 'analogs', label: 'Поиск аналогов', ms: analogs.ms }]
                  : []
              }
              algoVersion={algoVersion}
              algoUpdated={algoUpdated}
              embedDevices={{
                siglip2:
                  (result?.status?.embed_siglip2_device as string | undefined) ||
                  (
                    (result?.status?.steps as Record<string, any> | undefined)
                      ?.embeddings?.siglip2?.device as string | undefined
                  ) ||
                  (
                    (result?.status?.steps as Record<string, any> | undefined)
                      ?.embeddings?.siglip2?.provider as string | undefined
                  ) ||
                  null,
                dinov3:
                  (result?.status?.embed_dinov3_device as string | undefined) ||
                  (
                    (result?.status?.steps as Record<string, any> | undefined)
                      ?.embeddings?.dinov3?.device as string | undefined
                  ) ||
                  (
                    (result?.status?.steps as Record<string, any> | undefined)
                      ?.embeddings?.dinov3?.provider as string | undefined
                  ) ||
                  null,
              }}
            />
        </div>
      )}

      <CandidatesBlock
        title="Top-20 · SigLIP2"
        items={siglip}
        overlapIds={overlapIds}
        ocrMarksByWine={ocrMarksByWine}
        ocrScoresByWine={ocrScoresByWine}
        ocrTextScoresByWine={ocrTextScoresByWine}
        ocrChannels={ocrChannels}
        ocrQueryText={ocrQueryText}
        exclusiveLexicon={exclusiveLexicon}
        hsvStep={hsvStep}
        labelTextHr={labelTextHr}
        sortKeys={sortKeysSiglip}
        embeddingLabel="SigLIP2"
        cosByWine={cosByWine}
        sortBy={candSortBy}
        onSortByChange={changeCandSort}
        winnerId={finalWinnerId}
        xgbWinnerId={xgbWinnerId}
        similarIds={similarWineIds}
        manualWinesId={manualWinesId}
        finalistRanks={finalistRanks}
        finalMethod={finalMethod}
        finalOcrPrimary={finalOcrPrimary}
        onManualMatchChange={
          (result?.search_photos_id ?? result?.status?.search_photos_id) != null
            ? setManualMatch
            : undefined
        }
      />
      <CandidatesBlock
        title="Top-20 · DINOv3"
        items={dinov3}
        overlapIds={overlapIds}
        ocrMarksByWine={ocrMarksByWine}
        ocrScoresByWine={ocrScoresByWine}
        ocrTextScoresByWine={ocrTextScoresByWine}
        ocrChannels={ocrChannels}
        ocrQueryText={ocrQueryText}
        exclusiveLexicon={exclusiveLexicon}
        hsvStep={hsvStep}
        labelTextHr={labelTextHr}
        sortKeys={sortKeysDinov3}
        embeddingLabel="DINOv3"
        cosByWine={cosByWine}
        sortBy={candSortBy}
        onSortByChange={changeCandSort}
        winnerId={finalWinnerId}
        xgbWinnerId={xgbWinnerId}
        similarIds={similarWineIds}
        manualWinesId={manualWinesId}
        finalistRanks={finalistRanks}
        finalMethod={finalMethod}
        finalOcrPrimary={finalOcrPrimary}
        onManualMatchChange={
          (result?.search_photos_id ?? result?.status?.search_photos_id) != null
            ? setManualMatch
            : undefined
        }
      />

      {result && (
        <OcrVariantsBlock
          result={result}
          wineById={wineById}
          cosByWine={cosByWine}
          ocrScoresByWine={ocrScoresByWine}
          ocrQueryText={ocrQueryText}
          exclusiveLexicon={exclusiveLexicon}
          hsvStep={hsvStep}
          labelTextHr={labelTextHr}
        />
      )}

      {(fileName || status) && (
        <div className="scan-status">
          {fileName && <p>Файл: {fileName}</p>}
          {status && <p>{status}</p>}
          {result?.search_photos_id != null && (
            <p>ID: {result.search_photos_id}</p>
          )}
        </div>
      )}
    </main>
  )
}
