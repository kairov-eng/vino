import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchPipelineSettings,
  updatePipelineSettings,
  type PipelineSettings,
} from '../api/client'
import './ScanSettings.css'

type Props = {
  open: boolean
  onClose: () => void
}

type TabId = 'pipeline' | 'weights' | 'gates'

function _clamp01(v: number): number {
  return Math.max(0, Math.min(1, Math.round(v * 100) / 100))
}

function _clamp099(v: number): number {
  return Math.max(0, Math.min(0.99, Math.round(v * 100) / 100))
}

const FALLBACK_TEXT_WEIGHT_OPTIONS = [
  { key: 'wine', label: 'Название вина', env: 'TEXT_WEIGHT_WINE', value: 0.35 },
  { key: 'producer', label: 'Винодельня', env: 'TEXT_WEIGHT_PRODUCER', value: 0.28 },
  { key: 'color', label: 'Цвет', env: 'TEXT_WEIGHT_COLOR', value: 0.14 },
  { key: 'type', label: 'Тип (сухое…)', env: 'TEXT_WEIGHT_TYPE', value: 0.08 },
  { key: 'grape', label: 'Купаж / сорта', env: 'TEXT_WEIGHT_GRAPE', value: 0.12 },
  { key: 'vintage', label: 'Год', env: 'TEXT_WEIGHT_VINTAGE', value: 0.12 },
  { key: 'region', label: 'Регион', env: 'TEXT_WEIGHT_REGION', value: 0.08 },
  { key: 'other', label: 'Прочее', env: 'TEXT_WEIGHT_OTHER', value: 0.05 },
  { key: 'brand', label: 'Brand overlap', env: 'TEXT_WEIGHT_BRAND', value: 0.12 },
] as const

const FALLBACK_FINAL_WEIGHT_OPTIONS = [
  { key: 'text', label: 'OCR (TextScore)', env: 'FINAL_SCORE_W_TEXT', value: 0.45 },
  { key: 'cosine', label: 'Embedding (cosine)', env: 'FINAL_SCORE_W_COS', value: 0.55 },
] as const

function mergeWeightOptions(
  fromApi: PipelineSettings['text_weight_options'] | undefined,
  fallback: readonly { key: string; label: string; env: string; value: number }[],
  values: Record<string, number>,
) {
  const byKey = new Map((fromApi || []).map((o) => [o.key, o]))
  return fallback.map((fb) => {
    const api = byKey.get(fb.key)
    return {
      key: fb.key,
      label: api?.label || fb.label,
      env: api?.env || fb.env,
      value:
        values[fb.key] ??
        api?.value ??
        fb.value,
    }
  })
}

export function ScanSettingsPopup({ open, onClose }: Props) {
  const [tab, setTab] = useState<TabId>('pipeline')
  const [settings, setSettings] = useState<PipelineSettings | null>(null)
  const [weights, setWeights] = useState<Record<string, number>>(() =>
    Object.fromEntries(FALLBACK_TEXT_WEIGHT_OPTIONS.map((o) => [o.key, o.value])),
  )
  const [finalWeights, setFinalWeights] = useState<Record<string, number>>(() =>
    Object.fromEntries(FALLBACK_FINAL_WEIGHT_OPTIONS.map((o) => [o.key, o.value])),
  )
  const [error, setError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const weightTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const finalWeightTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!open) return
    setError(null)
    setTab('pipeline')
    fetchPipelineSettings()
      .then((data) => {
        setSettings(data)
        setWeights({
          ...Object.fromEntries(
            FALLBACK_TEXT_WEIGHT_OPTIONS.map((o) => [o.key, o.value]),
          ),
          ...data.text_weights,
        })
        setFinalWeights({
          ...Object.fromEntries(
            FALLBACK_FINAL_WEIGHT_OPTIONS.map((o) => [o.key, o.value]),
          ),
          ...(data.final_weights || {}),
        })
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : 'Ошибка загрузки'),
      )
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  useEffect(() => {
    return () => {
      if (weightTimer.current) clearTimeout(weightTimer.current)
      if (finalWeightTimer.current) clearTimeout(finalWeightTimer.current)
    }
  }, [])

  const persist = useCallback(async (patch: {
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
  }) => {
    // Mutual exclusions before save (mirrors backend _normalize).
    const gated: typeof patch = { ...patch }
    const cur = settings
    if (cur) {
      const soft = _clamp01(
        gated.hsv_bhattacharyya_max ?? cur.hsv_bhattacharyya_max ?? 0.5,
      )
      const hard = _clamp01(
        gated.hsv_hard_reject_max ?? cur.hsv_hard_reject_max ?? 0.9,
      )
      const useSoft =
        gated.use_hsv_filter !== undefined
          ? gated.use_hsv_filter
          : cur.use_hsv_filter
      const useHard =
        gated.use_hsv_hard_reject !== undefined
          ? gated.use_hsv_hard_reject
          : cur.use_hsv_hard_reject
      if (useSoft && useHard) {
        if (
          gated.hsv_bhattacharyya_max !== undefined &&
          soft > hard
        ) {
          gated.hsv_hard_reject_max = soft
        } else if (
          gated.hsv_hard_reject_max !== undefined &&
          hard < soft
        ) {
          gated.hsv_bhattacharyya_max = hard
        }
      }
      const xgbCur = cur.text_match_thresholds?.xgb || {
        match: 0.55,
        similar: 0.25,
      }
      const xgbPatch = gated.text_match_thresholds?.xgb
      const xgbMatch = _clamp01(
        xgbPatch?.match ?? Number(xgbCur.match ?? 0.55),
      )
      let dead = _clamp099(gated.xgb_dead_max ?? cur.xgb_dead_max ?? 0.15)
      if (xgbPatch?.match !== undefined && dead > xgbMatch) {
        gated.xgb_dead_max = xgbMatch
        dead = xgbMatch
      }
      if (gated.xgb_dead_max !== undefined && dead > xgbMatch) {
        gated.xgb_dead_max = xgbMatch
      }
      if (xgbPatch && typeof xgbPatch.similar === 'number' && typeof xgbPatch.match === 'number') {
        if (xgbPatch.similar > xgbPatch.match) {
          gated.text_match_thresholds = {
            ...gated.text_match_thresholds,
            xgb: { ...xgbPatch, similar: xgbPatch.match },
          }
        }
      }
    }
    setSaving(true)
    setError(null)
    try {
      const next = await updatePipelineSettings(gated)
      setSettings(next)
      setWeights({
        ...Object.fromEntries(
          FALLBACK_TEXT_WEIGHT_OPTIONS.map((o) => [o.key, o.value]),
        ),
        ...next.text_weights,
      })
      setFinalWeights({
        ...Object.fromEntries(
          FALLBACK_FINAL_WEIGHT_OPTIONS.map((o) => [o.key, o.value]),
        ),
        ...(next.final_weights || {}),
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ошибка сохранения')
    } finally {
      setSaving(false)
    }
  }, [settings])

  const togglePreprocess = (id: string, checked: boolean) => {
    if (!settings) return
    const set = new Set(settings.ocr_preprocess)
    if (checked) set.add(id)
    else set.delete(id)
    const next = ['A', 'B', 'C', 'D'].filter((k) => set.has(k))
    void persist({ ocr_preprocess: next })
  }

  const toggleEngine = (id: string, checked: boolean) => {
    if (!settings) return
    const order = ['rapid', 'easy', 'surya', 'tess']
    const set = new Set(settings.ocr_engines || [])
    if (checked) set.add(id)
    else set.delete(id)
    const next = order.filter((k) => set.has(k))
    void persist({ ocr_engines: next })
  }

  const onWeightInput = (key: string, value: number) => {
    setWeights((prev) => {
      const next = { ...prev, [key]: value }
      if (weightTimer.current) clearTimeout(weightTimer.current)
      weightTimer.current = setTimeout(() => {
        void persist({ text_weights: next })
      }, 250)
      return next
    })
  }

  const onFinalWeightInput = (key: string, value: number) => {
    setFinalWeights((prev) => {
      const next = { ...prev, [key]: value }
      if (finalWeightTimer.current) clearTimeout(finalWeightTimer.current)
      finalWeightTimer.current = setTimeout(() => {
        void persist({ final_weights: next })
      }, 250)
      return next
    })
  }

  if (!open) return null

  const weightSum = Object.values(weights).reduce((a, b) => a + Number(b || 0), 0)
  const finalSum = Object.values(finalWeights).reduce(
    (a, b) => a + Number(b || 0),
    0,
  )
  const textWeightOptions = mergeWeightOptions(
    settings?.text_weight_options,
    FALLBACK_TEXT_WEIGHT_OPTIONS,
    weights,
  )
  const finalWeightOptions = mergeWeightOptions(
    settings?.final_weight_options,
    FALLBACK_FINAL_WEIGHT_OPTIONS,
    finalWeights,
  )


  return (
    <div className="scan-settings-overlay" onClick={onClose} role="presentation">
      <div
        className="scan-settings"
        role="dialog"
        aria-modal="true"
        aria-label="Настройки сканера"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="scan-settings__head">
          <h2>Настройки поиска</h2>
          <button type="button" className="scan-settings__close" onClick={onClose}>
            ×
          </button>
        </header>

        <div className="scan-settings__tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'pipeline'}
            className={`scan-settings__tab${tab === 'pipeline' ? ' is-active' : ''}`}
            onClick={() => setTab('pipeline')}
          >
            1. Пайплайн
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'weights'}
            className={`scan-settings__tab${tab === 'weights' ? ' is-active' : ''}`}
            onClick={() => setTab('weights')}
          >
            2. Score
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === 'gates'}
            className={`scan-settings__tab${tab === 'gates' ? ' is-active' : ''}`}
            onClick={() => setTab('gates')}
          >
            3. Гейты
          </button>
        </div>

        {error && <p className="scan-settings__error">{error}</p>}
        {!settings && !error && <p className="scan-settings__loading">Загрузка…</p>}

        {settings && tab === 'pipeline' && (
          <>
            <section className="scan-settings__section">
              <h3>Поиск по embeddings</h3>
              <ul className="scan-settings__checks">
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.use_dinov3}
                      onChange={(e) =>
                        void persist({ use_dinov3: e.target.checked })
                      }
                    />
                    <span>
                      Искать по DINOv3
                      <small>SigLIP2 всегда включён</small>
                    </span>
                  </label>
                </li>
              </ul>
              <p className="scan-settings__hint" style={{ marginTop: '0.75rem' }}>
                Embedding with:
              </p>
              <ul className="scan-settings__checks">
                <li>
                  <label>
                    <input
                      type="radio"
                      name="embedding_device"
                      checked={(settings.embedding_device || 'cpu') === 'cpu'}
                      onChange={() => void persist({ embedding_device: 'cpu' })}
                    />
                    <span>
                      CPU
                      <small>endpoint на vino-svoe.online (по умолчанию)</small>
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="radio"
                      name="embedding_device"
                      checked={settings.embedding_device === 'gpu'}
                      onChange={() => void persist({ embedding_device: 'gpu' })}
                    />
                    <span>
                      GPU
                      <small>Hugging Face; при таймауте/ошибке start → свой CPU endpoint</small>
                    </span>
                  </label>
                </li>
              </ul>
              {(settings.embedding_device || 'cpu') === 'gpu' && (
                <label
                  className="scan-settings__hint"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginTop: '0.75rem',
                  }}
                >
                  <span>HF start timeout (сек)</span>
                  <input
                    type="number"
                    min={1}
                    max={300}
                    step={1}
                    value={settings.hf_start_timeout_sec ?? 15}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      if (!Number.isFinite(v)) return
                      void persist({ hf_start_timeout_sec: v })
                    }}
                    style={{ width: '4.5rem' }}
                  />
                  <small>по умолчанию 15; дальше — CPU endpoint</small>
                </label>
              )}
              <ul className="scan-settings__checks" style={{ marginTop: '0.75rem' }}>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={Boolean(settings.embed_cpu_use_cache)}
                      onChange={(e) =>
                        void persist({ embed_cpu_use_cache: e.target.checked })
                      }
                    />
                    <span>
                      Использовать кеш embedding на CPU
                      <small>
                        Redis на SigLIP2 / DINOv3 (use_cache=1); при GPU fallback тоже
                      </small>
                    </span>
                  </label>
                </li>
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>Нормализация фото</h3>
              <p className="scan-settings__hint">
                Макс. длинная сторона work JPEG перед поиском (без апскейла).
              </p>
              <ul className="scan-settings__checks">
                {(settings.normalize_max_side_options || [1280, 1024, 800]).map(
                  (side) => (
                    <li key={side}>
                      <label>
                        <input
                          type="radio"
                          name="normalize_max_side"
                          checked={
                            (settings.normalize_max_side || 1024) === side
                          }
                          onChange={() =>
                            void persist({ normalize_max_side: side })
                          }
                        />
                        <span>
                          {side}px
                          <small>
                            {side === 1024
                              ? 'по умолчанию'
                              : side === 1280
                                ? 'больше деталей'
                                : 'быстрее / меньше трафик'}
                          </small>
                        </span>
                      </label>
                    </li>
                  ),
                )}
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>Геометрия ключевых точек</h3>
              <p className="scan-settings__hint">
                ORB/SIFT + RANSAC: сверка кропа запроса с этикеткой каждого
                кандидата из top embedding.
              </p>
              <ul className="scan-settings__checks">
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.geometry_siglip2}
                      onChange={(e) =>
                        void persist({ geometry_siglip2: e.target.checked })
                      }
                    />
                    <span>
                      Геометрия для SigLIP2
                      <small>после top-20 SigLIP2</small>
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.geometry_dinov3}
                      disabled={!settings.use_dinov3}
                      onChange={(e) =>
                        void persist({ geometry_dinov3: e.target.checked })
                      }
                    />
                    <span>
                      Геометрия для DINOv3
                      <small>
                        {settings.use_dinov3
                          ? 'после top-20 DINOv3'
                          : 'нужен поиск DINOv3'}
                      </small>
                    </span>
                  </label>
                </li>
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>Движки OCR</h3>
              <p className="scan-settings__hint">
                Классические движки (preprocess × engine) и LLM OCR. Final score —
                какой OCR даёт текст для exclusive / XGB / CrEnc / OpenAI txt и канал финального
                score (fin1/fin2/xgb/crenc/openai_txt).
              </p>
              <ul className="scan-settings__checks">
                {(settings.ocr_engine_options || []).map((opt) => (
                  <li key={opt.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={opt.disabled ? false : opt.enabled}
                        disabled={Boolean(opt.disabled)}
                        onChange={(e) => {
                          if (opt.disabled) return
                          toggleEngine(opt.id, e.target.checked)
                        }}
                      />
                      <span>
                        {opt.title}
                        <small>
                          {opt.disabled ? 'не используется' : opt.name}
                        </small>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
              <ul className="scan-settings__method-rows">
                {(
                  [
                    {
                      id: 'gemini',
                      flag: 'use_gemini_ocr' as const,
                      title: 'Google Gemini OCR',
                      hint: 'OCR-only по кропу',
                      checked: settings.use_gemini_ocr,
                    },
                    {
                      id: 'openai',
                      flag: 'use_openai_ocr' as const,
                      title: 'OpenAI OCR',
                      hint: 'OCR-only по кропу этикетки',
                      checked: settings.use_openai_ocr !== false,
                    },
                    {
                      id: 'deepseek',
                      flag: 'use_deepseek_ocr' as const,
                      title: 'DeepSeek OCR',
                      hint: 'deepseek-flash',
                      checked: settings.use_deepseek_ocr !== false,
                    },
                    {
                      id: 'qwen',
                      flag: 'use_qwen_ocr' as const,
                      title: 'Qwen2.5-VL OCR (HF)',
                      hint: 'выкл. по умолчанию',
                      checked: settings.use_qwen_ocr === true,
                    },
                    {
                      id: 'yandex',
                      flag: 'use_yandex_ocr' as const,
                      title: 'Yandex OCR',
                      hint: 'только строки (lines)',
                      checked: settings.use_yandex_ocr !== false,
                    },
                    {
                      id: 'google_vision',
                      flag: 'use_google_vision_ocr' as const,
                      title: 'Google Vision OCR',
                      hint: 'DOCUMENT_TEXT · SA credentials',
                      checked: settings.use_google_vision_ocr !== false,
                    },
                  ] as const
                ).map((row) => {
                  const enabled = Boolean(row.checked)
                  const finalOcr = settings.final_ocr || 'auto'
                  return (
                    <li key={row.id} className="scan-settings__method-row">
                      <label className="scan-settings__method-enable">
                        <input
                          type="checkbox"
                          checked={enabled}
                          onChange={(e) => {
                            const on = e.target.checked
                            const patch: Record<string, unknown> = {
                              [row.flag]: on,
                            }
                            if (!on && finalOcr === row.id) {
                              patch.final_ocr = 'auto'
                            }
                            void persist(patch as Parameters<typeof persist>[0])
                          }}
                        />
                        <span>
                          {row.title}
                          <small>{row.hint}</small>
                        </span>
                      </label>
                      <label className="scan-settings__final-radio">
                        <input
                          type="radio"
                          name="final_ocr"
                          checked={finalOcr === row.id}
                          disabled={!enabled}
                          onChange={() => {
                            void persist({
                              final_ocr: row.id,
                              [row.flag]: true,
                            } as Parameters<typeof persist>[0])
                          }}
                        />
                        <span>Final score</span>
                      </label>
                    </li>
                  )
                })}
                <li className="scan-settings__method-row">
                  <label className="scan-settings__method-enable">
                    <span>
                      Auto
                      <small>первый доступный: Vision→…→Qwen</small>
                    </span>
                  </label>
                  <label className="scan-settings__final-radio">
                    <input
                      type="radio"
                      name="final_ocr"
                      checked={(settings.final_ocr || 'auto') === 'auto'}
                      onChange={() => void persist({ final_ocr: 'auto' })}
                    />
                    <span>Final score</span>
                  </label>
                </li>
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>YOLO / этикетка</h3>
              <ul className="scan-settings__checks">
                <li>
                  <label>
                    <input
                      type="radio"
                      name="label_detect_mode"
                      checked={
                        settings.use_yolo !== false &&
                        (settings.yolo_variant || 'label') === 'label'
                      }
                      onChange={() =>
                        void persist({
                          use_yolo: true,
                          yolo_variant: 'label',
                        })
                      }
                    />
                    <span>
                      YOLO
                      <small>только этикетка</small>
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="radio"
                      name="label_detect_mode"
                      checked={
                        settings.use_yolo !== false &&
                        settings.yolo_variant === 'bottle_label'
                      }
                      onChange={() =>
                        void persist({
                          use_yolo: true,
                          yolo_variant: 'bottle_label',
                        })
                      }
                    />
                    <span>
                      YOLO bottle+label
                      <small>бутылка + все этикетки ≥85%</small>
                    </span>
                  </label>
                  <label className="scan-settings__bottle-conf">
                    <span>
                      conf бутылки &gt;
                      <small>если бутылок несколько</small>
                    </span>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={(settings.bottle_min_conf ?? 0.6).toFixed(2)}
                      disabled={settings.yolo_variant !== 'bottle_label'}
                      onChange={(e) => {
                        const v = Number(e.target.value)
                        if (!Number.isFinite(v)) return
                        void persist({
                          bottle_min_conf: Math.max(0, Math.min(1, v)),
                        })
                      }}
                    />
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="radio"
                      name="label_detect_mode"
                      checked={
                        settings.use_yolo === false &&
                        settings.label_detect_openai !== false
                      }
                      onChange={() =>
                        void persist({
                          use_yolo: false,
                          label_detect_openai: true,
                          label_detect_gemini: false,
                        })
                      }
                    />
                    <span>
                      OpenAI: рамка + OCR
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="radio"
                      name="label_detect_mode"
                      checked={
                        settings.use_yolo === false &&
                        settings.label_detect_openai === false &&
                        settings.label_detect_gemini !== false
                      }
                      onChange={() =>
                        void persist({
                          use_yolo: false,
                          label_detect_openai: false,
                          label_detect_gemini: true,
                        })
                      }
                    />
                    <span>
                      Gemini: рамка + OCR
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.yolo_fallback_full_image !== false}
                      onChange={(e) =>
                        void persist({
                          yolo_fallback_full_image: e.target.checked,
                        })
                      }
                    />
                    <span>
                      Fallback: исходное фото как этикетка
                      <small>выкл. — стоп, если рамка не найдена</small>
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.use_bottle_orient === true}
                      onChange={(e) =>
                        void persist({
                          use_bottle_orient: e.target.checked,
                        })
                      }
                    />
                    <span>
                      Исправлять поворот этикетки
                      <small>
                        выровнять кадр по оси бутылки перед кропом OCR
                      </small>
                    </span>
                  </label>
                </li>
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>Преобразование этикетки (OCR)</h3>
              <ul className="scan-settings__checks">
                {settings.ocr_preprocess_options.map((opt) => (
                  <li key={opt.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={opt.enabled}
                        onChange={(e) => togglePreprocess(opt.id, e.target.checked)}
                      />
                      <span className={`scan-settings__letter is-ocr-${opt.id}`}>
                        {opt.id}
                      </span>
                      <span>
                        {opt.title}
                        <small>{opt.name}</small>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            </section>

            {saving && <p className="scan-settings__saving">Сохранение…</p>}
          </>
        )}

        {settings && tab === 'weights' && (
          <>
            <section className="scan-settings__section">
              <h3>Exclusive lexicon (сорта и винодельни)</h3>
              <p className="scan-settings__hint">
                Как искать формы сорта и винодельни в OCR и на этикетке
                кандидата (category/type не затрагиваются).
              </p>
              <ul className="scan-settings__checks">
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.exclusive_use_translit !== false}
                      onChange={(e) =>
                        void persist({
                          exclusive_use_translit: e.target.checked,
                        })
                      }
                    />
                    <span>
                      Использовать транслитерацию
                      <small>денисов ↔ denisov в обе стороны</small>
                    </span>
                  </label>
                </li>
                <li>
                  <label>
                    <input
                      type="checkbox"
                      checked={settings.exclusive_match_spaced !== false}
                      onChange={(e) =>
                        void persist({
                          exclusive_match_spaced: e.target.checked,
                        })
                      }
                    />
                    <span>
                      Искать с пробелами
                      <small>
                        0–2 пробела/дефиса между буквами («ка б е р н е»)
                      </small>
                    </span>
                  </label>
                </li>
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>Веса полей TextScore (OCR)</h3>
              <p className="scan-settings__hint">
                Сравнение OCR с карточкой вина: название, винодельня, цвет, тип,
                купаж и др. Сумма {weightSum.toFixed(2)} (нормализуется).
              </p>
              <ul className="scan-settings__sliders">
                {textWeightOptions.map((opt) => (
                  <li key={opt.key}>
                    <div className="scan-settings__slider-head">
                      <span>{opt.label}</span>
                      <strong>
                        {Number(weights[opt.key] ?? opt.value).toFixed(2)}
                      </strong>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={weights[opt.key] ?? opt.value}
                      onChange={(e) =>
                        onWeightInput(opt.key, Number(e.target.value))
                      }
                    />
                  </li>
                ))}
              </ul>
            </section>

            {saving && <p className="scan-settings__saving">Сохранение…</p>}
          </>
        )}

        {settings && tab === 'gates' && (
          <>
            <section className="scan-settings__section">
              <h3>Финальное решение — порядок и исключения</h3>
              <ol className="scan-settings__rules">
                <li>
                  TextScore (XGB / fin / CrEnc) → band:{' '}
                  <strong>match</strong> / similar / none (пороги ниже).
                </li>
                <li>
                  Если final = XGB и max xgb_score &lt;{' '}
                  <strong>мёртвый XGB (τ)</strong> → не брать XGB_fin; fallback
                  Soft TF‑IDF (fin2) или cos ≥ порога пустого OCR.
                </li>
                <li>
                  Иначе победитель = max{' '}
                  <strong>w_ocr×Text + w_emb×Cosine</strong> среди match-band.
                </li>
                <li>
                  HSV soft: d &gt; порога → отсев (кроме cos ≥ ignore). HSV hard:
                  d &gt; hard → отсев всегда (байпас cos не действует).
                </li>
              </ol>
              <ul className="scan-settings__rules scan-settings__rules--mutex">
                <li>
                  <strong>similar ≤ match</strong> (для каждого метода)
                </li>
                <li>
                  <strong>HSV soft ≤ HSV hard</strong> (если оба включены)
                </li>
                <li>
                  <strong>мёртвый XGB τ ≤ XGB match</strong>
                </li>
                <li>
                  Порог cos пустого OCR = тот же для dead-XGB fallback
                </li>
              </ul>
              {Number(settings.text_match_thresholds?.xgb?.match ?? 0) < 0.1 && (
                <p className="scan-settings__warn">
                  XGB match &lt; 0.10 — почти любой слабый XGB станет «совпадением»
                  (частая причина FP). Рекомендуется ≥ 0.25.
                </p>
              )}
              <p className="scan-settings__hint scan-settings__hint--live">
                Сейчас: XGB match{' '}
                {Number(settings.text_match_thresholds?.xgb?.match ?? 0).toFixed(2)}{' '}
                / similar{' '}
                {Number(
                  settings.text_match_thresholds?.xgb?.similar ?? 0,
                ).toFixed(2)}
                ; τ={Number(settings.xgb_dead_max ?? 0).toFixed(2)}; cos≥
                {Number(settings.empty_ocr_cosine_threshold ?? 0).toFixed(2)};
                fin w_ocr=
                {Number(finalWeights.text ?? 0).toFixed(2)} / w_cos=
                {Number(finalWeights.cosine ?? 0).toFixed(2)}; HSV{' '}
                {settings.compute_hsv ? 'вкл' : 'выкл'} soft=
                {Number(settings.hsv_bhattacharyya_max ?? 0).toFixed(2)}
                {settings.use_hsv_filter ? '' : ' (soft выкл)'} / hard=
                {Number(settings.hsv_hard_reject_max ?? 0).toFixed(2)}
                {settings.use_hsv_hard_reject ? '' : ' (hard выкл)'}; ColorDelta{' '}
                {settings.compute_color_delta === false ? 'выкл' : 'вкл'}.
              </p>
            </section>

            <section className="scan-settings__section">
              <h3>Способ сравнения текстов</h3>
              <p className="scan-settings__hint">
                Снятый вариант не считается. Final score — один метод для
                результата. Score &gt; match → совпадение; &lt; similar → нет;
                между ними — similar (пунктир), final пустой.
              </p>
              <ul className="scan-settings__method-rows">
                {(settings.text_match_options || []).map((opt) => {
                  const thr =
                    settings.text_match_thresholds?.[opt.id] || {
                      match: opt.match ?? 0.55,
                      similar: opt.similar ?? 0.25,
                    }
                  const matchV = Number(thr.match ?? 0.55)
                  const similarV = Number(thr.similar ?? 0.25)
                  const stepThr = (
                    field: 'match' | 'similar',
                    delta: number,
                  ) => {
                    let next =
                      Math.round(
                        (field === 'match' ? matchV : similarV) * 100 +
                          delta * 100,
                      ) / 100
                    next = Math.max(0, Math.min(1, next))
                    const patch: {
                      match: number
                      similar: number
                    } = {
                      match: matchV,
                      similar: similarV,
                      [field]: next,
                    }
                    if (patch.similar > patch.match) {
                      if (field === 'match') patch.similar = patch.match
                      else patch.match = patch.similar
                    }
                    void persist({
                      text_match_thresholds: { [opt.id]: patch },
                    })
                  }
                  const isFinal =
                    (settings.final_score_method || 'fin1') === opt.id
                  return (
                    <li
                      key={opt.id}
                      className={`scan-settings__method-row${isFinal ? ' is-final' : ''}`}
                    >
                      <label className="scan-settings__method-enable">
                        <input
                          type="checkbox"
                          checked={opt.enabled}
                          onChange={(e) => {
                            const order = [
                              'fin1',
                              'fin2',
                              'xgb',
                              'crenc',
                              'crenc_srv',
                              'openai_txt',
                            ]
                            const set = new Set(
                              settings.text_match_methods || [],
                            )
                            if (e.target.checked) {
                              set.add(opt.id)
                              if (opt.id === 'crenc') set.delete('crenc_srv')
                              if (opt.id === 'crenc_srv') set.delete('crenc')
                            } else {
                              set.delete(opt.id)
                            }
                            const next = order.filter((k) => set.has(k))
                            let method = settings.final_score_method
                            if (!next.includes(method)) {
                              method = next[0] || method
                            }
                            void persist({
                              text_match_methods: next,
                              final_score_method: method,
                            })
                          }}
                        />
                        <span>
                          {opt.label}
                          {opt.id === 'xgb' ? (
                            <small>match / similar + τ dead-XGB</small>
                          ) : null}
                        </span>
                      </label>
                      <div className="scan-settings__thr">
                        <span className="scan-settings__thr-lab">match &gt;</span>
                        <div className="scan-settings__stepper">
                          <input
                            type="number"
                            min={0}
                            max={1}
                            step={0.01}
                            value={matchV.toFixed(2)}
                            disabled={!opt.enabled}
                            onChange={(e) => {
                              const v = Number(e.target.value)
                              if (!Number.isFinite(v)) return
                              let match = Math.max(0, Math.min(1, v))
                              let similar = similarV
                              if (similar > match) similar = match
                              void persist({
                                text_match_thresholds: {
                                  [opt.id]: { match, similar },
                                },
                              })
                            }}
                          />
                          <div className="scan-settings__stepper-btns">
                            <button
                              type="button"
                              disabled={!opt.enabled}
                              aria-label="match +0.01"
                              onClick={() => stepThr('match', 0.01)}
                            >
                              ▲
                            </button>
                            <button
                              type="button"
                              disabled={!opt.enabled}
                              aria-label="match -0.01"
                              onClick={() => stepThr('match', -0.01)}
                            >
                              ▼
                            </button>
                          </div>
                        </div>
                        <span className="scan-settings__thr-lab">
                          similar &gt;
                        </span>
                        <div className="scan-settings__stepper">
                          <input
                            type="number"
                            min={0}
                            max={1}
                            step={0.01}
                            value={similarV.toFixed(2)}
                            disabled={!opt.enabled}
                            onChange={(e) => {
                              const v = Number(e.target.value)
                              if (!Number.isFinite(v)) return
                              let similar = Math.max(0, Math.min(1, v))
                              let match = matchV
                              if (similar > match) match = similar
                              void persist({
                                text_match_thresholds: {
                                  [opt.id]: { match, similar },
                                },
                              })
                            }}
                          />
                          <div className="scan-settings__stepper-btns">
                            <button
                              type="button"
                              disabled={!opt.enabled}
                              aria-label="similar +0.01"
                              onClick={() => stepThr('similar', 0.01)}
                            >
                              ▲
                            </button>
                            <button
                              type="button"
                              disabled={!opt.enabled}
                              aria-label="similar -0.01"
                              onClick={() => stepThr('similar', -0.01)}
                            >
                              ▼
                            </button>
                          </div>
                        </div>
                      </div>
                      <label className="scan-settings__final-radio">
                        <input
                          type="radio"
                          name="final_score_method"
                          checked={
                            (settings.final_score_method || 'fin1') === opt.id
                          }
                          disabled={!opt.enabled}
                          onChange={() => {
                            const order = [
                              'fin1',
                              'fin2',
                              'xgb',
                              'crenc',
                              'crenc_srv',
                              'openai_txt',
                            ]
                            const set = new Set(
                              settings.text_match_methods || [],
                            )
                            set.add(opt.id)
                            if (opt.id === 'crenc') set.delete('crenc_srv')
                            if (opt.id === 'crenc_srv') set.delete('crenc')
                            void persist({
                              final_score_method: opt.id,
                              text_match_methods: order.filter((k) =>
                                set.has(k),
                              ),
                            })
                          }}
                        />
                        <span>Final score</span>
                      </label>
                    </li>
                  )
                })}
              </ul>
            </section>

            <section className="scan-settings__section">
              <h3>Цветовая гамма</h3>
              <p className="scan-settings__hint">
                HSV — расстояние Бхаттачарии (0 — одна гамма, 1 — не
                пересекаются). ColorDelta — CIEDE2000 доминантных цветов
                (меньше = ближе). Пороги HSV работают только если включено
                «Считать HSV».
              </p>
              <div className="scan-settings__hsv-row">
                <label className="scan-settings__hsv-check">
                  <input
                    type="checkbox"
                    checked={settings.compute_hsv === true}
                    onChange={(e) => {
                      const on = e.target.checked
                      void persist({
                        compute_hsv: on,
                        ...(!on
                          ? {
                              use_hsv_filter: false,
                              hsv_ignore_high_cosine: false,
                              use_hsv_hard_reject: false,
                            }
                          : {}),
                      })
                    }}
                  />
                  <span>Считать HSV</span>
                </label>
              </div>
              <div className="scan-settings__hsv-row">
                <label className="scan-settings__hsv-check">
                  <input
                    type="checkbox"
                    checked={settings.compute_color_delta !== false}
                    onChange={(e) =>
                      void persist({ compute_color_delta: e.target.checked })
                    }
                  />
                  <span>Считать Доминантный цвет CIEDE2000 (ColorDelta)</span>
                </label>
              </div>
              <div className="scan-settings__hsv-row">
                <label className="scan-settings__hsv-check">
                  <input
                    type="checkbox"
                    checked={settings.use_hsv_filter === true}
                    disabled={settings.compute_hsv !== true}
                    onChange={(e) =>
                      void persist({
                        use_hsv_filter: e.target.checked,
                        ...(e.target.checked ? { compute_hsv: true } : {}),
                      })
                    }
                  />
                  <span>
                    Soft filter HSV (расстояние Бхаттачарии &gt; порога → отсев)
                  </span>
                </label>
                <div className="scan-settings__thr scan-settings__thr--inline">
                  <span className="scan-settings__thr-lab">&gt;</span>
                  <div className="scan-settings__stepper">
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      disabled={
                        settings.compute_hsv !== true ||
                        settings.use_hsv_filter !== true
                      }
                      value={Number(
                        settings.hsv_bhattacharyya_max ?? 0.5,
                      ).toFixed(2)}
                      onChange={(e) => {
                        const v = Number(e.target.value)
                        if (!Number.isFinite(v)) return
                        void persist({
                          hsv_bhattacharyya_max: Math.max(
                            0,
                            Math.min(1, Math.round(v * 100) / 100),
                          ),
                        })
                      }}
                    />
                    <div className="scan-settings__stepper-btns">
                      <button
                        type="button"
                        aria-label="HSV порог +0.01"
                        disabled={
                        settings.compute_hsv !== true ||
                        settings.use_hsv_filter !== true
                      }
                        onClick={() => {
                          const cur = Number(
                            settings.hsv_bhattacharyya_max ?? 0.5,
                          )
                          void persist({
                            hsv_bhattacharyya_max: Math.max(
                              0,
                              Math.min(
                                1,
                                Math.round((cur + 0.01) * 100) / 100,
                              ),
                            ),
                          })
                        }}
                      >
                        ▲
                      </button>
                      <button
                        type="button"
                        aria-label="HSV порог -0.01"
                        disabled={
                        settings.compute_hsv !== true ||
                        settings.use_hsv_filter !== true
                      }
                        onClick={() => {
                          const cur = Number(
                            settings.hsv_bhattacharyya_max ?? 0.5,
                          )
                          void persist({
                            hsv_bhattacharyya_max: Math.max(
                              0,
                              Math.min(
                                1,
                                Math.round((cur - 0.01) * 100) / 100,
                              ),
                            ),
                          })
                        }}
                      >
                        ▼
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <div className="scan-settings__hsv-row">
                <label className="scan-settings__hsv-check">
                  <input
                    type="checkbox"
                    checked={Boolean(settings.hsv_ignore_high_cosine)}
                    disabled={settings.compute_hsv !== true}
                    onChange={(e) => {
                      const on = e.target.checked
                      void persist({
                        hsv_ignore_high_cosine: on,
                        ...(on
                          ? { use_hsv_filter: true, compute_hsv: true }
                          : {}),
                      })
                    }}
                  />
                  <span>
                    Игнорировать HSV при высоком косинусном сходстве
                  </span>
                </label>
                <div className="scan-settings__thr scan-settings__thr--inline">
                  <span className="scan-settings__thr-lab">cos ≥</span>
                  <div className="scan-settings__stepper">
                    <input
                      type="number"
                      min={0}
                      max={0.99}
                      step={0.01}
                      disabled={
                        settings.compute_hsv !== true ||
                        !settings.hsv_ignore_high_cosine
                      }
                      value={Number(
                        settings.hsv_ignore_cosine_min ?? 0.85,
                      ).toFixed(2)}
                      onChange={(e) => {
                        const v = Number(e.target.value)
                        if (!Number.isFinite(v)) return
                        void persist({
                          hsv_ignore_cosine_min: Math.max(
                            0,
                            Math.min(0.99, Math.round(v * 100) / 100),
                          ),
                        })
                      }}
                    />
                    <div className="scan-settings__stepper-btns">
                      <button
                        type="button"
                        aria-label="HSV ignore cos +0.01"
                        disabled={
                        settings.compute_hsv !== true ||
                        !settings.hsv_ignore_high_cosine
                      }
                        onClick={() => {
                          const cur = Number(
                            settings.hsv_ignore_cosine_min ?? 0.85,
                          )
                          void persist({
                            hsv_ignore_cosine_min: Math.max(
                              0,
                              Math.min(
                                0.99,
                                Math.round((cur + 0.01) * 100) / 100,
                              ),
                            ),
                          })
                        }}
                      >
                        ▲
                      </button>
                      <button
                        type="button"
                        aria-label="HSV ignore cos -0.01"
                        disabled={
                        settings.compute_hsv !== true ||
                        !settings.hsv_ignore_high_cosine
                      }
                        onClick={() => {
                          const cur = Number(
                            settings.hsv_ignore_cosine_min ?? 0.85,
                          )
                          void persist({
                            hsv_ignore_cosine_min: Math.max(
                              0,
                              Math.min(
                                0.99,
                                Math.round((cur - 0.01) * 100) / 100,
                              ),
                            ),
                          })
                        }}
                      >
                        ▼
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <p className="scan-settings__hint">
                Если HSV-фильтр отсекает кандидата, но max cosine ≥ порога —
                кандидат остаётся (не unsuitable). Включение этой опции также
                включает сравнение HSV выше.
              </p>
              <div className="scan-settings__hsv-row">
                <label className="scan-settings__hsv-check">
                  <input
                    type="checkbox"
                    checked={settings.use_hsv_hard_reject === true}
                    disabled={settings.compute_hsv !== true}
                    onChange={(e) =>
                      void persist({
                        use_hsv_hard_reject: e.target.checked,
                        ...(e.target.checked ? { compute_hsv: true } : {}),
                      })
                    }
                  />
                  <span>
                    Hard reject при HSV &gt; (даже при высоком cosine)
                  </span>
                </label>
                <div className="scan-settings__thr scan-settings__thr--inline">
                  <span className="scan-settings__thr-lab">&gt;</span>
                  <div className="scan-settings__stepper">
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      disabled={
                        settings.compute_hsv !== true ||
                        settings.use_hsv_hard_reject !== true
                      }
                      value={Number(
                        settings.hsv_hard_reject_max ?? 0.9,
                      ).toFixed(2)}
                      onChange={(e) => {
                        const v = Number(e.target.value)
                        if (!Number.isFinite(v)) return
                        void persist({
                          hsv_hard_reject_max: Math.max(
                            0,
                            Math.min(1, Math.round(v * 100) / 100),
                          ),
                        })
                      }}
                    />
                    <div className="scan-settings__stepper-btns">
                      <button
                        type="button"
                        aria-label="HSV hard reject +0.01"
                        disabled={
                        settings.compute_hsv !== true ||
                        settings.use_hsv_hard_reject !== true
                      }
                        onClick={() => {
                          const cur = Number(
                            settings.hsv_hard_reject_max ?? 0.9,
                          )
                          void persist({
                            hsv_hard_reject_max: Math.max(
                              0,
                              Math.min(
                                1,
                                Math.round((cur + 0.01) * 100) / 100,
                              ),
                            ),
                          })
                        }}
                      >
                        ▲
                      </button>
                      <button
                        type="button"
                        aria-label="HSV hard reject -0.01"
                        disabled={
                        settings.compute_hsv !== true ||
                        settings.use_hsv_hard_reject !== true
                      }
                        onClick={() => {
                          const cur = Number(
                            settings.hsv_hard_reject_max ?? 0.9,
                          )
                          void persist({
                            hsv_hard_reject_max: Math.max(
                              0,
                              Math.min(
                                1,
                                Math.round((cur - 0.01) * 100) / 100,
                              ),
                            ),
                          })
                        }}
                      >
                        ▼
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <p className="scan-settings__hint">
                Абсолютный порог: HSV &gt; значения → unsuitable всегда, байпас
                по cosine не действует. При обоих фильтрах soft ≤ hard
                (иначе hard поднимается / soft опускается автоматически).
              </p>
            </section>

            <section className="scan-settings__section">
              <h3>Порог cosine (пустой OCR и dead-XGB)</h3>
              <p className="scan-settings__hint">
                Если OCR пустой — text match не запускается; final = кандидат с
                max cos ≥ порога. Тот же порог для fallback при «мёртвом XGB»
                (когда τ срабатывает и Soft TF‑IDF не дал match). Диапазон
                0…0.99.
              </p>
              <div className="scan-settings__thr scan-settings__thr--solo">
                <span className="scan-settings__thr-lab">cos ≥</span>
                <div className="scan-settings__stepper">
                  <input
                    type="number"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={Number(
                      settings.empty_ocr_cosine_threshold ?? 0.75,
                    ).toFixed(2)}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      if (!Number.isFinite(v)) return
                      void persist({
                        empty_ocr_cosine_threshold: Math.max(
                          0,
                          Math.min(0.99, v),
                        ),
                      })
                    }}
                  />
                  <div className="scan-settings__stepper-btns">
                    <button
                      type="button"
                      aria-label="cos +0.01"
                      onClick={() => {
                        const cur = Number(
                          settings.empty_ocr_cosine_threshold ?? 0.75,
                        )
                        void persist({
                          empty_ocr_cosine_threshold: Math.max(
                            0,
                            Math.min(0.99, Math.round((cur + 0.01) * 100) / 100),
                          ),
                        })
                      }}
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      aria-label="cos -0.01"
                      onClick={() => {
                        const cur = Number(
                          settings.empty_ocr_cosine_threshold ?? 0.75,
                        )
                        void persist({
                          empty_ocr_cosine_threshold: Math.max(
                            0,
                            Math.min(0.99, Math.round((cur - 0.01) * 100) / 100),
                          ),
                        })
                      }}
                    >
                      ▼
                    </button>
                  </div>
                </div>
              </div>
            </section>

            <section className="scan-settings__section">
              <h3>Мёртвый XGB (τ)</h3>
              <p className="scan-settings__hint">
                Если final = XGB и max xgb_score &lt; τ — не брать XGB_fin;
                fallback Soft TF‑IDF (fin2 всегда, даже если выключен в методах),
                иначе max cos ≥ порога выше. τ автоматически ≤ XGB match.
              </p>
              <div className="scan-settings__thr scan-settings__thr--solo">
                <span className="scan-settings__thr-lab">τ max xgb</span>
                <div className="scan-settings__stepper">
                  <input
                    type="number"
                    min={0}
                    max={0.99}
                    step={0.01}
                    value={Number(settings.xgb_dead_max ?? 0.15).toFixed(2)}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      if (!Number.isFinite(v)) return
                      void persist({
                        xgb_dead_max: Math.max(0, Math.min(0.99, v)),
                      })
                    }}
                  />
                  <div className="scan-settings__stepper-btns">
                    <button
                      type="button"
                      aria-label="τ +0.01"
                      onClick={() => {
                        const cur = Number(settings.xgb_dead_max ?? 0.15)
                        void persist({
                          xgb_dead_max: Math.max(
                            0,
                            Math.min(0.99, Math.round((cur + 0.01) * 100) / 100),
                          ),
                        })
                      }}
                    >
                      ▲
                    </button>
                    <button
                      type="button"
                      aria-label="τ -0.01"
                      onClick={() => {
                        const cur = Number(settings.xgb_dead_max ?? 0.15)
                        void persist({
                          xgb_dead_max: Math.max(
                            0,
                            Math.min(0.99, Math.round((cur - 0.01) * 100) / 100),
                          ),
                        })
                      }}
                    >
                      ▼
                    </button>
                  </div>
                </div>
              </div>
            </section>

            <section className="scan-settings__section">
              <h3>Вклад в FinalScore (cos / XGB)</h3>
              <p className="scan-settings__hint">
                FinalScore = w_ocr×TextScore + w_emb×Cosine. Сумма сейчас{' '}
                {finalSum.toFixed(2)} (при расчёте нормализуется). HSV в смесь не
                входит — только soft/hard reject выше.
              </p>
              <ul className="scan-settings__sliders">
                {finalWeightOptions.map((opt) => (
                  <li key={opt.key}>
                    <div className="scan-settings__slider-head">
                      <span>{opt.label}</span>
                      <strong>
                        {Number(finalWeights[opt.key] ?? opt.value).toFixed(2)}
                      </strong>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={finalWeights[opt.key] ?? opt.value}
                      onChange={(e) =>
                        onFinalWeightInput(opt.key, Number(e.target.value))
                      }
                    />
                  </li>
                ))}
              </ul>
            </section>

            {saving && <p className="scan-settings__saving">Сохранение…</p>}
          </>
        )}
      </div>
    </div>
  )
}

export function ScanSettingsGear({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      className="scan-settings-gear"
      onClick={onClick}
      title="Настройки поиска"
      aria-label="Настройки поиска"
    >
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
        <path
          fill="currentColor"
          d="M19.14 12.94c.04-.31.06-.63.06-.94s-.02-.63-.06-.94l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.6-.22l-2.39.96a7.03 7.03 0 0 0-1.63-.94l-.36-2.54A.5.5 0 0 0 13.9 2h-3.8a.5.5 0 0 0-.49.42l-.36 2.54c-.59.24-1.14.55-1.63.94l-2.39-.96a.5.5 0 0 0-.6.22L2.71 8.48a.5.5 0 0 0 .12.64l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94L2.83 14.58a.5.5 0 0 0-.12.64l1.92 3.32c.14.24.43.34.68.22l2.39-.96c.49.39 1.04.7 1.63.94l.36 2.54c.05.24.25.42.49.42h3.8c.24 0 .44-.18.49-.42l.36-2.54c.59-.24 1.14-.55 1.63-.94l2.39.96c.25.12.54.02.68-.22l1.92-3.32a.5.5 0 0 0-.12-.64l-2.03-1.58ZM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7Z"
        />
      </svg>
    </button>
  )
}
