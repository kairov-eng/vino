"""Multi-preprocess OCR (A–D) × engines (RapidOCR, EasyOCR, Surya, Tesseract).

Branches (after optional light deskew):
  A: Original
  B: Upscale ×3 → CLAHE
  C: Grayscale → Sharpen
  D: Grayscale → Adaptive Threshold

Each preprocessed image is OCR'd by every available engine.
Languages: only when the engine requires them (EasyOCR → ru+en;
Tesseract → rus+eng). Surya / RapidOCR run without language args.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from app.db.config import (
    OCR_BUDGET_SEC,
    OCR_DET_MODEL_PATH,
    OCR_ENGINE_TIMEOUT_SEC,
    OCR_HEAVY_ONLY_PREPROCESS_A,
    OCR_MAX_SIDE_EASY,
    OCR_MAX_SIDE_SURYA,
    OCR_REC_MODEL_PATH,
    OCR_TIMEOUT_MS,
)
from app.pipeline.ocr_match import normalize_ocr_text
from app.pipeline.runtime_settings import get_ocr_engines, get_ocr_preprocess

PREPROCESS_KEYS = ("A", "B", "C", "D")
# Prefer fast engines first; Surya last (CPU-heavy).
ENGINE_KEYS = ("rapid", "tess", "easy", "surya")
_HEAVY_ENGINES = frozenset({"easy", "surya"})

_rapid_engine = None
_easy_reader = None
_easy_failed: str | None = None
_surya_pair = None  # (recognition, detection) | False if unavailable
_tess_ready: bool | None = None
# One worker per engine (+ spare): timeout must not include queue wait behind
# a stuck Easy/Surya load. Timed-out jobs may keep running; extra slots avoid deadlock.
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
_warmup_started = False
# Easy + Surya оба на Torch/CPU — параллельно сильно замедляют друг друга
_torch_ocr_lock = threading.Lock()


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _downscale_bgr(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink long side for heavy OCR; returns (image, scale)."""
    max_side = max(64, int(max_side))
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return bgr, 1.0
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), scale


def _get_rapid():
    global _rapid_engine
    if _rapid_engine is not None:
        return _rapid_engine
    from rapidocr_onnxruntime import RapidOCR

    kwargs: dict[str, Any] = {}
    if OCR_DET_MODEL_PATH:
        kwargs["det_model_path"] = OCR_DET_MODEL_PATH
    if OCR_REC_MODEL_PATH:
        kwargs["rec_model_path"] = OCR_REC_MODEL_PATH
    _rapid_engine = RapidOCR(**kwargs) if kwargs else RapidOCR()
    return _rapid_engine


def _get_easy():
    """EasyOCR: ru+en, GPU if available. Quantize off — breaks on Torch 2.x (meta device)."""
    global _easy_reader, _easy_failed
    if _easy_failed:
        raise RuntimeError(_easy_failed)
    if _easy_reader is not None:
        return _easy_reader
    try:
        import easyocr

        use_gpu = _torch_device() == "cuda"
        kwargs: dict[str, Any] = {
            "gpu": use_gpu,
            "verbose": False,
            # quantize=True → "Tensor on device meta is not on the expected device cpu"
            # на актуальных torch/easyocr; выигрыш на CPU невелик vs стабильность
            "quantize": False,
        }
        _easy_reader = easyocr.Reader(["ru", "en"], **kwargs)
        return _easy_reader
    except Exception as exc:  # noqa: BLE001
        _easy_failed = f"EasyOCR недоступен: {exc}"
        raise RuntimeError(_easy_failed) from exc


def _get_surya():
    """Surya multilingual OCR — keep predictors warm in-process."""
    global _surya_pair
    if _surya_pair is False:
        raise RuntimeError("Surya OCR недоступен")
    if _surya_pair is not None:
        return _surya_pair
    # Disable progress bars (huge slowdown in logs / TTY)
    import os

    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        try:
            from surya.foundation import FoundationPredictor

            foundation = FoundationPredictor()
            recognition = RecognitionPredictor(foundation)
        except Exception:  # noqa: BLE001
            recognition = RecognitionPredictor()
        detection = DetectionPredictor()
        _surya_pair = (recognition, detection)
        return _surya_pair
    except Exception as exc:  # noqa: BLE001
        _surya_pair = False
        raise RuntimeError(f"Surya OCR недоступен: {exc}") from exc


def _ensure_tesseract() -> None:
    """Locate tesseract binary; prefer no-lang OCR, fallback rus+eng."""
    global _tess_ready
    if _tess_ready is True:
        return
    if _tess_ready is False:
        raise RuntimeError("Tesseract OCR недоступен")
    try:
        import pytesseract

        # Common Windows install path if not on PATH
        candidates = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
        for cand in candidates:
            if cand.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(cand)
                break
        pytesseract.get_tesseract_version()
        _tess_ready = True
    except Exception as exc:  # noqa: BLE001
        _tess_ready = False
        raise RuntimeError(f"Tesseract OCR недоступен: {exc}") from exc


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не удалось открыть label: {path}")
    return img


def _deskew(bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Light auto-deskew (±15°). Returns image, angle_deg, ms."""
    t0 = time.perf_counter()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = 0.0
    if len(coords) >= 50:
        rect = cv2.minAreaRect(coords)
        angle = float(rect[-1])
        if angle < -45:
            angle = 90.0 + angle
        if abs(angle) > 15:
            angle = 0.0
        elif abs(angle) < 0.3:
            angle = 0.0
    out = bgr
    if abs(angle) >= 0.3:
        h, w = bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(
            bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    return out, angle, (time.perf_counter() - t0) * 1000


def _variant_a(bgr: np.ndarray) -> np.ndarray:
    return bgr


def _variant_b(bgr: np.ndarray) -> np.ndarray:
    """Upscale ×3 → CLAHE on L channel (color preserved for OCR)."""
    up = cv2.resize(bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _variant_c(bgr: np.ndarray) -> np.ndarray:
    """Grayscale → unsharp mask."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _variant_d(bgr: np.ndarray) -> np.ndarray:
    """Grayscale → adaptive threshold."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


_PREPROCESS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "A": ("original", _variant_a),
    "B": ("upscale_clahe", _variant_b),
    "C": ("gray_sharpen", _variant_c),
    "D": ("gray_adaptive_threshold", _variant_d),
}

_ENGINE_TITLES = {
    "rapid": "RapidOCR",
    "easy": "EasyOCR",
    "surya": "Surya",
    "tess": "Tesseract",
}


def _bgr_to_rgb_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _parse_rapid_result(result) -> tuple[str, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    if result:
        for item in result:
            if not item or len(item) < 2:
                continue
            text = str(item[1]).strip()
            score = float(item[2]) if len(item) > 2 else None
            box = item[0]
            if text:
                texts.append(text)
                lines.append({"text": text, "score": score, "box": box})
    return "\n".join(texts), lines


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _run_rapid(bgr: np.ndarray) -> tuple[str, list[dict[str, Any]], Any]:
    engine = _get_rapid()
    result, elapse = engine(bgr)
    text, lines = _parse_rapid_result(result)
    return text, lines, elapse


def _run_easy(bgr: np.ndarray) -> tuple[str, list[dict[str, Any]], Any]:
    with _torch_ocr_lock:
        reader = _get_easy()
        small, scale = _downscale_bgr(bgr, OCR_MAX_SIDE_EASY)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # greedy + workers=0 — быстрее на CPU, чем beamsearch
        result = reader.readtext(
            rgb,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=4,
            workers=0,
        )
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    for item in result or []:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        score = float(item[2]) if len(item) > 2 else None
        if text:
            texts.append(text)
            box = item[0]
            if scale < 1.0 and box is not None:
                try:
                    box = [[float(x) / scale, float(y) / scale] for x, y in box]
                except Exception:  # noqa: BLE001
                    pass
            lines.append({"text": text, "score": score, "box": box})
    return "\n".join(texts), lines, {"max_side": OCR_MAX_SIDE_EASY, "scale": round(scale, 4)}


def _run_surya_inprocess(bgr: np.ndarray) -> tuple[str, list[dict[str, Any]], Any]:
    """Surya OCR in current process (used by warmup / subprocess worker)."""
    recognition, detection = _get_surya()
    small, scale = _downscale_bgr(bgr, OCR_MAX_SIDE_SURYA)
    pil = _bgr_to_rgb_pil(small)
    os.environ.setdefault("TQDM_DISABLE", "1")
    predictions = recognition([pil], det_predictor=detection)
    pred = predictions[0] if predictions else None
    lines: list[dict[str, Any]] = []
    texts: list[str] = []
    text_lines = getattr(pred, "text_lines", None) if pred is not None else None
    if text_lines is None and pred is not None:
        text_lines = getattr(pred, "blocks", None) or []
    for line in text_lines or []:
        raw = getattr(line, "text", None)
        if raw is None:
            raw = _strip_html(getattr(line, "html", "") or "")
        text = str(raw or "").strip()
        if not text:
            continue
        score = getattr(line, "confidence", None)
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None
        box = getattr(line, "bbox", None) or getattr(line, "polygon", None)
        if box is not None:
            try:
                box = [[float(x) for x in pt] for pt in box]  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                try:
                    box = [float(x) for x in box]  # type: ignore[arg-type]
                except Exception:  # noqa: BLE001
                    box = None
        texts.append(text)
        lines.append({"text": text, "score": score, "box": box})
    return (
        "\n".join(texts),
        lines,
        {"max_side": OCR_MAX_SIDE_SURYA, "scale": round(scale, 4)},
    )


def _surya_subprocess_entry(img_path: str, result_path: str) -> None:
    """Child process entry: native crash here must not kill the API worker."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    try:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"не удалось открыть {img_path}")
        text, lines, meta = _run_surya_inprocess(bgr)
        payload = {"ok": True, "text": text, "lines": lines, "meta": meta}
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "error": str(exc)}
    Path(result_path).write_text(
        __import__("json").dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_surya(bgr: np.ndarray) -> tuple[str, list[dict[str, Any]], Any]:
    """Run Surya in a spawn subprocess so segfaults don't take down uvicorn."""
    import json

    with tempfile.TemporaryDirectory(prefix="vino_surya_") as tmp:
        img_path = str(Path(tmp) / "in.jpg")
        result_path = str(Path(tmp) / "out.json")
        if not cv2.imwrite(img_path, bgr):
            raise RuntimeError("surya: не удалось записать временный JPEG")
        timeout_sec = float(OCR_ENGINE_TIMEOUT_SEC.get("surya", 120))
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_surya_subprocess_entry,
            args=(img_path, result_path),
            name="ocr-surya",
            daemon=True,
        )
        proc.start()
        proc.join(timeout=timeout_sec + 5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            if proc.is_alive():
                proc.kill()
                proc.join(2.0)
            raise TimeoutError(f"surya: превышен лимит {timeout_sec:.0f}s")
        if proc.exitcode not in (0, None) and not Path(result_path).is_file():
            raise RuntimeError(
                f"surya: процесс завершился с кодом {proc.exitcode} "
                "(возможен native crash — API продолжит работу)"
            )
        if not Path(result_path).is_file():
            raise RuntimeError(
                f"surya: нет результата (exit={proc.exitcode})"
            )
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "surya failed")
        return payload["text"], payload.get("lines") or [], payload.get("meta")


def _run_tess(bgr: np.ndarray) -> tuple[str, list[dict[str, Any]], Any]:
    """Tesseract needs language packs for Cyrillic → rus+eng."""
    import pytesseract

    _ensure_tesseract()
    pil = _bgr_to_rgb_pil(bgr)
    meta: dict[str, Any] = {"lang": "rus+eng"}
    try:
        text = (pytesseract.image_to_string(pil, lang="rus+eng") or "").strip()
    except Exception:
        # Fallback if rus pack missing — no lang (default eng)
        text = (pytesseract.image_to_string(pil) or "").strip()
        meta["lang"] = None
    lines = [{"text": ln, "score": None, "box": None} for ln in text.splitlines() if ln.strip()]
    return text, lines, meta


_ENGINE_RUNNERS: dict[str, Callable[[np.ndarray], tuple[str, list[dict[str, Any]], Any]]] = {
    "rapid": _run_rapid,
    "easy": _run_easy,
    "surya": _run_surya,
    "tess": _run_tess,
}


def _active_preprocess() -> list[str]:
    """Active OCR preprocess keys; empty list = classical OCR engines skipped."""
    return [p for p in PREPROCESS_KEYS if p in set(get_ocr_preprocess())]


def _active_engines() -> list[str]:
    """May be empty — only LLM OCR (Gemini/OpenAI) may still run."""
    wanted = set(get_ocr_engines())
    return [e for e in ENGINE_KEYS if e in wanted]


def _run_in_thread(
    eng: str, bgr: np.ndarray, timeout_sec: float
) -> tuple[str, list[dict[str, Any]], Any]:
    """Run engine with wall-clock timeout counted from job start (not queue wait)."""
    started = threading.Event()
    holder: dict[str, Any] = {}

    def _job() -> None:
        started.set()
        try:
            holder["result"] = _ENGINE_RUNNERS[eng](bgr)
        except Exception as exc:  # noqa: BLE001
            holder["error"] = exc

    fut = _thread_pool.submit(_job)
    # Wait until a pool worker actually picks the job up.
    queue_wait = max(30.0, float(timeout_sec))
    if not started.wait(timeout=queue_wait):
        raise TimeoutError(f"{eng}: не дождался слота OCR ({queue_wait:.0f}s)")
    try:
        fut.result(timeout=timeout_sec)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(
            f"{eng}: превышен лимит {timeout_sec:.0f}s"
        ) from exc
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


def warmup_ocr_engines(engines: list[str] | None = None) -> dict[str, Any]:
    """Load heavy models once (background). Call at API startup."""
    wanted = engines or [e for e in _active_engines() if e in _HEAVY_ENGINES]
    out: dict[str, Any] = {}
    # tiny blank image — just to init graphs
    blank = np.full((128, 96, 3), 240, dtype=np.uint8)
    for eng in wanted:
        t0 = time.perf_counter()
        try:
            if eng == "easy":
                _get_easy()
                _run_easy(blank)
            elif eng == "surya":
                # Warm via isolated subprocess (same path as production)
                _run_surya(blank)
            out[eng] = {"ok": True, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            out[eng] = {"ok": False, "error": str(exc)}
    return out


def start_ocr_warmup_background() -> None:
    """Non-blocking warm of Easy/Surya so first scan is not 20–90s cold-start."""
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True
    wanted = [e for e in _active_engines() if e in _HEAVY_ENGINES]
    if not wanted:
        return

    def _job() -> None:
        try:
            warmup_ocr_engines(wanted)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_job, name="ocr-warmup", daemon=True).start()


def variant_keys(
    engines: list[str] | None = None,
    preprocess: list[str] | None = None,
) -> list[str]:
    engs = _active_engines() if engines is None else engines
    preps = _active_preprocess() if preprocess is None else preprocess
    return [f"{p}_{e}" for p in preps for e in engs]


def _ocr_one_engine(
    eng: str,
    prep: dict[str, Any],
    *,
    t_start: float,
) -> tuple[str, dict[str, Any]]:
    """Run a single engine on a prepared preprocess dict."""
    prep_key = str(prep.get("id") or "?")
    vid = f"{prep_key}_{eng}"
    v: dict[str, Any] = {
        "id": vid,
        "preprocess": prep_key,
        "preprocess_name": prep.get("name"),
        "engine": eng,
        "engine_name": _ENGINE_TITLES.get(eng, eng),
        "name": f"{prep.get('name')}+{_ENGINE_TITLES.get(eng, eng)}",
        "ok": False,
        "filename": prep.get("filename"),
        "path": prep.get("path"),
        "preprocess_ms": prep.get("preprocess_ms"),
    }
    if time.perf_counter() - t_start >= OCR_BUDGET_SEC:
        v["error"] = f"OCR budget {OCR_BUDGET_SEC:.0f}s исчерпан"
        v["skipped"] = True
        return vid, v
    if (
        OCR_HEAVY_ONLY_PREPROCESS_A
        and eng in _HEAVY_ENGINES
        and prep_key != "A"
    ):
        v["skipped"] = True
        v["error"] = "Easy/Surya только на preprocess A (скорость)"
        return vid, v
    if not prep.get("ok"):
        v["error"] = prep.get("error") or f"preprocess {prep_key} failed"
        return vid, v
    img = prep.get("image")
    if img is None:
        v["error"] = "нет изображения preprocess"
        return vid, v
    timeout_sec = float(OCR_ENGINE_TIMEOUT_SEC.get(eng, 30))
    try:
        t_ocr = time.perf_counter()
        v["started_ms"] = round((t_ocr - t_start) * 1000, 1)
        # Surya already isolates in a subprocess with its own timeout —
        # do not nest another thread timeout (avoids killing mid-cleanup).
        if eng == "surya":
            text, lines, elapse = _run_surya(img)
        else:
            text, lines, elapse = _run_in_thread(eng, img, timeout_sec)
        ocr_ms = (time.perf_counter() - t_ocr) * 1000
        v.update(
            {
                "ok": True,
                "text": text,
                "text_norm": normalize_ocr_text(text),
                "lines": lines,
                "ocr_ms": round(ocr_ms, 1),
                "engine_elapse": elapse,
                "timed_out": ocr_ms > OCR_TIMEOUT_MS,
            }
        )
    except Exception as exc:  # noqa: BLE001
        v["error"] = str(exc)
        v["timed_out"] = isinstance(exc, TimeoutError) or "лимит" in str(exc)
        if "started_ms" not in v:
            v["started_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return vid, v


def _prepare_one(
    key: str,
    deskewed: np.ndarray,
    label_path: Path,
    *,
    t_origin: float,
) -> dict[str, Any]:
    name, fn = _PREPROCESS[key]
    info: dict[str, Any] = {"id": key, "name": name, "ok": False}
    try:
        t_prep = time.perf_counter()
        info["started_ms"] = round((t_prep - t_origin) * 1000, 1)
        processed = fn(deskewed)
        prep_ms = (time.perf_counter() - t_prep) * 1000
        out_path = label_path.with_name(f"{label_path.stem}_ocr_{key}{label_path.suffix}")
        cv2.imwrite(str(out_path), processed)
        info.update(
            {
                "ok": True,
                "image": processed,
                "path": str(out_path),
                "filename": out_path.name,
                "preprocess_ms": round(prep_ms, 1),
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def run_ocr_variants(label_path: Path) -> dict[str, Any]:
    """Parallel preprocess (B–D) + OCR engines; A OCR starts ASAP after deskew."""
    t_total = time.perf_counter()
    engines = _active_engines()
    prep_keys = _active_preprocess()
    if not engines or not prep_keys:
        return {
            "ok": False,
            "skipped": True,
            "reason": (
                "ocr_engines_empty"
                if not engines
                else "ocr_preprocess_empty"
            ),
            "deskew_angle_deg": 0.0,
            "deskew_ms": 0.0,
            "preprocess": {},
            "preprocess_keys": prep_keys,
            "engines": engines,
            "budget_sec": OCR_BUDGET_SEC,
            "budget_hit": False,
            "parallel": True,
            "variants": {},
            "text": "",
            "text_aggregated": "",
            "text_norm": "",
            "timeout_ms": OCR_TIMEOUT_MS,
            "ms": round((time.perf_counter() - t_total) * 1000, 1),
        }
    bgr0 = _load_bgr(label_path)
    t_deskew = time.perf_counter()
    deskew_started_ms = round((t_deskew - t_total) * 1000, 1)
    deskewed, deskew_angle, deskew_ms = _deskew(bgr0)

    prepared: dict[str, dict[str, Any]] = {}
    variants_out: dict[str, Any] = {}
    budget_hit = False

    # Workers: preprocess × engines (cap reasonably)
    max_workers = min(12, max(4, len(prep_keys) * len(engines) + 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        # 1) Start all preprocess jobs in parallel (incl. A)
        prep_futs: dict[str, concurrent.futures.Future] = {
            key: pool.submit(
                _prepare_one, key, deskewed, label_path, t_origin=t_total
            )
            for key in prep_keys
        }

        ocr_futs: list[concurrent.futures.Future] = []

        # 2) As each preprocess finishes → launch all engines immediately
        pending_prep = set(prep_futs.values())
        while pending_prep:
            done, pending_prep = concurrent.futures.wait(
                pending_prep, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                # find key
                key = next(k for k, f in prep_futs.items() if f is fut)
                try:
                    info = fut.result()
                except Exception as exc:  # noqa: BLE001
                    info = {"id": key, "name": _PREPROCESS[key][0], "ok": False, "error": str(exc)}
                prepared[key] = info
                if time.perf_counter() - t_total >= OCR_BUDGET_SEC:
                    budget_hit = True
                    for eng in engines:
                        vid = f"{key}_{eng}"
                        variants_out[vid] = {
                            "id": vid,
                            "preprocess": key,
                            "engine": eng,
                            "engine_name": _ENGINE_TITLES.get(eng, eng),
                            "name": f"{info.get('name')}+{_ENGINE_TITLES.get(eng, eng)}",
                            "ok": False,
                            "skipped": True,
                            "error": f"OCR budget {OCR_BUDGET_SEC:.0f}s исчерпан",
                            "filename": info.get("filename"),
                            "path": info.get("path"),
                            "preprocess_ms": info.get("preprocess_ms"),
                        }
                    continue
                for eng in engines:
                    ocr_futs.append(
                        pool.submit(_ocr_one_engine, eng, info, t_start=t_total)
                    )

        # 3) Collect OCR results
        for fut in concurrent.futures.as_completed(ocr_futs):
            try:
                vid, v = fut.result()
            except Exception as exc:  # noqa: BLE001
                vid = f"err_{id(fut)}"
                v = {"id": vid, "ok": False, "error": str(exc)}
            variants_out[vid] = v
            if v.get("skipped"):
                budget_hit = True

    # Fill missing variants
    for eng in engines:
        for prep_key in prep_keys:
            vid = f"{prep_key}_{eng}"
            if vid not in variants_out:
                prep = prepared.get(prep_key) or {}
                variants_out[vid] = {
                    "id": vid,
                    "preprocess": prep_key,
                    "engine": eng,
                    "engine_name": _ENGINE_TITLES.get(eng, eng),
                    "name": f"{prep.get('name')}+{_ENGINE_TITLES.get(eng, eng)}",
                    "ok": False,
                    "skipped": True,
                    "error": f"OCR budget {OCR_BUDGET_SEC:.0f}s исчерпан"
                    if budget_hit
                    else "OCR не выполнен",
                    "filename": prep.get("filename"),
                    "path": prep.get("path"),
                    "preprocess_ms": prep.get("preprocess_ms"),
                }

    for prep_key in prep_keys:
        if prep_key in prepared:
            prepared[prep_key].pop("image", None)

    for prep_key in prep_keys:
        rapid_key = f"{prep_key}_rapid"
        if rapid_key in variants_out:
            alias = dict(variants_out[rapid_key])
            alias["id"] = prep_key
            alias["alias_of"] = rapid_key
            variants_out[prep_key] = alias

    order = list(variant_keys(engines, prep_keys)) + list(prep_keys)
    primary = ""
    for key in order:
        t = (variants_out.get(key) or {}).get("text") or ""
        if t.strip():
            primary = t
            if key in ("A_rapid", "A"):
                break

    seen: set[str] = set()
    agg_lines: list[str] = []
    for key in order:
        if key in prep_keys:
            continue
        raw = (variants_out.get(key) or {}).get("text") or ""
        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            key_l = s.casefold()
            if key_l in seen:
                continue
            seen.add(key_l)
            agg_lines.append(s)
    aggregated = "\n".join(agg_lines)

    return {
        "ok": any(
            (variants_out.get(k) or {}).get("ok")
            for k in variant_keys(engines, prep_keys)
        ),
        "deskew_angle_deg": round(deskew_angle, 2),
        "deskew_ms": round(deskew_ms, 1),
        "deskew_started_ms": deskew_started_ms,
        "preprocess": {
            k: {kk: vv for kk, vv in info.items() if kk != "image"}
            for k, info in prepared.items()
        },
        "preprocess_keys": prep_keys,
        "engines": engines,
        "budget_sec": OCR_BUDGET_SEC,
        "budget_hit": budget_hit,
        "parallel": True,
        "variants": variants_out,
        "text": primary,
        "text_aggregated": aggregated,
        "text_norm": normalize_ocr_text(aggregated or primary),
        "timeout_ms": OCR_TIMEOUT_MS,
        "ms": round((time.perf_counter() - t_total) * 1000, 1),
    }


def run_ocr(image_path: Path) -> dict[str, Any]:
    """Backward-compatible single-call API → multi-variant result."""
    return run_ocr_variants(image_path)
