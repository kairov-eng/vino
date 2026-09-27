from typing import Any

from pydantic import BaseModel, ConfigDict


class WineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    wineries_id: int | None = None
    name: str
    category: str | None = None
    color: str | None = None
    region: str | None = None
    grape_variety: str | None = None
    description: str | None = None
    winery: str | None = None
    slug: str | None = None
    photo_name: str | None = None
    photo_url: str | None = None
    crop: int | None = None
    label: str | None = None
    label_url: str | None = None
    # Structured OCR JSON (detail page); omitted/null in list responses
    label_ocr: dict[str, Any] | None = None


class WineListResponse(BaseModel):
    items: list[WineOut]
    total: int
    offset: int
    limit: int
    has_more: bool


class FilterFacet(BaseModel):
    value: str
    count: int


class FiltersResponse(BaseModel):
    category: list[FilterFacet]
    color: list[FilterFacet]
    region: list[FilterFacet]
    grape_variety: list[FilterFacet]
    winery: list[FilterFacet]
    name: list[FilterFacet]


class ScanResponse(BaseModel):
    filename: str
    message: str
    size: int


class FindWineCandidate(BaseModel):
    id: int
    cosine_similarity: float
    name: str | None = None
    winery: str | None = None
    slug: str | None = None
    label: str | None = None
    color: str | None = None
    category: str | None = None
    grape_variety: str | None = None
    vintage: str | None = None
    region: str | None = None
    photo_url: str | None = None
    label_url: str | None = None
    geometry_ok: bool | None = None
    geometry_inliers: int | None = None
    geometry_matches: int | None = None
    geometry_inlier_ratio: float | None = None
    geometry_score: float | None = None
    xgb_score: float | None = None
    xgb_fin: float | None = None
    xgb_fin_pre_reject: float | None = None
    xgb_suitable: bool | None = None
    xgb_fin_reason: str | None = None
    exclusive_rejected: bool | None = None
    label_text_hard_reject: str | None = None
    crenc_score: float | None = None
    crenc_fin: float | None = None
    crenc_suitable: bool | None = None
    crenc_fin_reason: str | None = None
    # OpenAI сравнение текста (вероятность 0…1)
    llm_txt: float | None = None
    # Bhattacharyya distance of HSV histograms (0 same gamut, 1 no overlap)
    hsv: float | None = None
    # CIEDE2000 of dominant Lab colors (lower = closer gamut)
    color_delta: float | None = None
    # Final score per OCR engine for selected text method (fin1/fin2/xgb/crenc)
    final_by_ocr: dict[str, float] | None = None
    final_ocr_primary: str | None = None


class EvalPredictResponse(BaseModel):
    """Ответ для participant_test.sh: только slug победителя."""

    slug: str | None = None


class FindWineResponse(BaseModel):
    search_photos_id: int | None = None
    crops_filename: str | None = None
    label_filename: str | None = None
    crops_url: str | None = None
    label_url: str | None = None
    algorithm_version: str | None = None
    algorithm_updated_at: str | None = None
    candidates_siglip2: list[FindWineCandidate] = []
    candidates_dinov3: list[FindWineCandidate] = []
    status: dict
    matched_wine_id: int | None = None
    matched_wine_confidence: float | None = None
    matched_wine_name: str | None = None
    matched_wine_slug: str | None = None
    matched_wine_photo_url: str | None = None
    # Ручная отметка «Соответствие» (search_photos.manual_wines_id)
    manual_wines_id: int | None = None


class ManualWineUpdate(BaseModel):
    """Установить / снять manual_wines_id для search_photos (взаимоисключающе в UI)."""

    wine_id: int | None = None


class ManualWineOut(BaseModel):
    search_photos_id: int
    manual_wines_id: int | None = None


class ScanHistoryXgbTopItem(BaseModel):
    """Top XGB_fin candidate when search found no final match."""

    id: int
    name: str | None = None
    slug: str | None = None
    winery: str | None = None
    label: str | None = None
    label_url: str | None = None
    photo_url: str | None = None
    xgb_score: float | None = None
    xgb_fin: float | None = None
    cosine: float | None = None


class ScanHistoryItem(BaseModel):
    id: int
    created_at: str | None = None
    query_photo_url: str | None = None
    query_filename: str | None = None
    total_ms: float | None = None
    ocr_texts: dict[str, str | None] = {}
    # Лучший OCR-текст искомой этикетки (для колонки OCR)
    query_ocr_text: str | None = None
    matched_wine_id: int | None = None
    matched_wine_confidence: float | None = None
    matched_wine_name: str | None = None
    matched_wine_slug: str | None = None
    matched_wine_photo_url: str | None = None
    matched_wine_label: str | None = None
    # Если финального вина нет — top-3 по XGB_fin (для колонки «Найденная этикетка»)
    xgb_top: list[ScanHistoryXgbTopItem] | None = None
    # Скоры как на карточках результатов (0..1 где применимо)
    scores: dict[str, float | str | None] | None = None
    algorithm_version: str | None = None
    # Ручная разметка: 0/1, взаимоисключающие
    false_positive: int = 0
    false_negative: int = 0


class ScanHistoryEvalUpdate(BaseModel):
    """Exactly one of FP/FN may be 1; both may be 0."""

    false_positive: int = 0
    false_negative: int = 0


class SiteAuthRequest(BaseModel):
    password: str = ""


class SiteAuthResponse(BaseModel):
    ok: bool
    role: str | None = None
    enabled: bool = True


class ScanHistoryResponse(BaseModel):
    items: list[ScanHistoryItem]
    total: int
    offset: int
    limit: int
    has_more: bool


class ScanHistoryReportErrorItem(BaseModel):
    id: int
    matched_wine_id: int | None = None
    manual_wines_id: int | None = None
    hist_fp: int = 0
    hist_fn: int = 0
    reason: str
    cls: str


class ScanHistoryReportCounts(BaseModel):
    TP: int
    FP: int
    TN: int
    FN: int
    n: int
    min_id: int | None = None
    max_id: int | None = None


class ScanHistoryReportMetrics(BaseModel):
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    specificity: float | None = None
    npv: float | None = None
    accuracy: float | None = None
    fpr: float | None = None
    fnr: float | None = None
    match_rate: float | None = None


class ScanHistoryReportBin(BaseModel):
    range: str
    n: int
    pct: float


class ScanHistoryReportScoreDist(BaseModel):
    """Гистограмма 0.1 по cosine или xgb_score."""

    metric: str
    polarity: str  # plus | minus
    n: int
    bins: list[ScanHistoryReportBin]
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None


class ScanHistoryReportResponse(BaseModel):
    id_from: int
    id_to: int
    counts: ScanHistoryReportCounts
    metrics: ScanHistoryReportMetrics
    false_positives: list[ScanHistoryReportErrorItem]
    false_negatives: list[ScanHistoryReportErrorItem]
    rules: list[str]
    score_dists: list[ScanHistoryReportScoreDist] = []


class PipelineSettingsUpdate(BaseModel):
    ocr_preprocess: list[str] | None = None
    ocr_engines: list[str] | None = None
    text_weights: dict[str, float] | None = None
    final_weights: dict[str, float] | None = None
    use_dinov3: bool | None = None
    geometry_siglip2: bool | None = None
    geometry_dinov3: bool | None = None
    use_gemini_ocr: bool | None = None
    use_openai_ocr: bool | None = None
    use_deepseek_ocr: bool | None = None
    use_qwen_ocr: bool | None = None
    use_yandex_ocr: bool | None = None
    use_google_vision_ocr: bool | None = None
    yolo_fallback_full_image: bool | None = None
    use_yolo: bool | None = None
    yolo_variant: str | None = None
    bottle_min_conf: float | None = None
    use_bottle_orient: bool | None = None
    bottle_orient_min_deg: float | None = None
    label_detect_openai: bool | None = None
    label_detect_gemini: bool | None = None
    embedding_device: str | None = None
    embed_cpu_use_cache: bool | None = None
    normalize_max_side: int | None = None
    hf_start_timeout_sec: float | None = None
    text_match_methods: list[str] | None = None
    final_score_method: str | None = None
    text_match_thresholds: dict[str, dict[str, float]] | None = None
    empty_ocr_cosine_threshold: float | None = None
    xgb_dead_max: float | None = None
    compute_hsv: bool | None = None
    use_hsv_filter: bool | None = None
    hsv_bhattacharyya_max: float | None = None
    hsv_ignore_high_cosine: bool | None = None
    hsv_ignore_cosine_min: float | None = None
    use_hsv_hard_reject: bool | None = None
    hsv_hard_reject_max: float | None = None
    compute_color_delta: bool | None = None
    final_ocr: str | None = None
    exclusive_use_translit: bool | None = None
    exclusive_match_spaced: bool | None = None


class PipelineSettingsOut(BaseModel):
    ocr_preprocess: list[str]
    ocr_preprocess_options: list[dict]
    ocr_engines: list[str] = []
    ocr_engine_options: list[dict] = []
    text_weights: dict[str, float]
    text_weight_options: list[dict]
    final_weights: dict[str, float] = {"text": 0.45, "cosine": 0.55}
    final_weight_options: list[dict] = []
    use_dinov3: bool = True
    geometry_siglip2: bool = True
    geometry_dinov3: bool = True
    use_gemini_ocr: bool = True
    use_openai_ocr: bool = True
    use_deepseek_ocr: bool = True
    use_qwen_ocr: bool = False
    use_yandex_ocr: bool = True
    use_google_vision_ocr: bool = True
    yolo_fallback_full_image: bool = True
    use_yolo: bool = True
    yolo_variant: str = "label"
    bottle_min_conf: float = 0.60
    use_bottle_orient: bool = True
    bottle_orient_min_deg: float = 8.0
    label_detect_openai: bool = True
    label_detect_gemini: bool = True
    embedding_device: str = "cpu"
    embed_cpu_use_cache: bool = False
    normalize_max_side: int = 1024
    normalize_max_side_options: list[int] = [1280, 1024, 800]
    hf_start_timeout_sec: float = 15.0
    text_match_methods: list[str] = ["fin1", "fin2", "xgb", "crenc"]
    text_match_options: list[dict] = []
    text_match_thresholds: dict[str, dict[str, float]] = {}
    final_score_method: str = "fin1"
    final_score_method_options: list[dict] = []
    empty_ocr_cosine_threshold: float = 0.75
    xgb_dead_max: float = 0.15
    compute_hsv: bool = False
    use_hsv_filter: bool = False
    hsv_bhattacharyya_max: float = 0.50
    hsv_ignore_high_cosine: bool = False
    hsv_ignore_cosine_min: float = 0.85
    use_hsv_hard_reject: bool = False
    hsv_hard_reject_max: float = 0.90
    compute_color_delta: bool = True
    final_ocr: str = "auto"
    final_ocr_options: list[dict] = []
    exclusive_use_translit: bool = True
    exclusive_match_spaced: bool = True
