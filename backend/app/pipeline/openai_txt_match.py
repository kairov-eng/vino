"""OpenAI text-vs-text match: query OCR ↔ catalog label texts of candidates.

Prompt template: ``openai_txt_match_prompt.txt`` next to this module.
Placeholders: ``{{ocr_text}}``, ``{{candidates_json}}``.

Env: OPENAI_KEY / OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from app.db import config as app_config

logger = logging.getLogger("vino.openai_txt_match")

_PROMPT_PATH = Path(__file__).with_name("openai_txt_match_prompt.txt")
_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# All candidate ids scored; sorted list kept in ``scores`` / ``top``.
_TOP_PREVIEW = 10


def load_prompt_template(*, path: Path | None = None) -> str:
    src = path or _PROMPT_PATH
    return src.read_text(encoding="utf-8")


def _candidate_lines(wine: dict[str, Any]) -> str:
    """Catalog label text for comparison (prefer label, fallback name)."""
    label = str(wine.get("label") or "").strip()
    if label:
        return label
    return str(wine.get("name") or "").strip()


def build_candidates_payload(
    wines: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """List of {\"<id>\": \"lines\"} for the prompt JSON block."""
    out: list[dict[str, str]] = []
    for w in wines:
        wid = w.get("id")
        if wid is None:
            continue
        lines = _candidate_lines(w)
        if not lines:
            continue
        out.append({str(int(wid)): lines})
    return out


def render_prompt(
    *,
    ocr_text: str,
    wines: list[dict[str, Any]],
    template: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Fill template; return (prompt, candidates_payload)."""
    payload = build_candidates_payload(wines)
    tpl = template if template is not None else load_prompt_template()
    candidates_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = (
        tpl.replace("{{ocr_text}}", str(ocr_text or "").strip())
        .replace("{{candidates_json}}", candidates_json)
    )
    return prompt, payload


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("пустой ответ OpenAI")
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("OpenAI JSON не объект")
    return data


def _normalize_scores(raw: dict[str, Any]) -> dict[int, float]:
    """Parse {id: probability} → {int_id: float 0..1}, keep top by score."""
    scores: list[tuple[int, float]] = []
    for k, v in raw.items():
        try:
            wid = int(str(k).strip())
        except (TypeError, ValueError):
            continue
        try:
            p = float(v)
        except (TypeError, ValueError):
            continue
        if p != p:  # NaN
            continue
        p = max(0.0, min(1.0, p))
        scores.append((wid, p))
    scores.sort(key=lambda t: (-t[1], t[0]))
    # Keep all returned ids (model may return ≤5); still sort for UI/best.
    return {wid: round(p, 6) for wid, p in scores}


_OCR_META_KEYS = {
    "ocr",
    "query_ocr",
    "label_ocr",
    "matches",
    "scores",
    "lines",
    "producer",
    "wine_name",
    "color",
    "type",
    "cupage",
    "coupage",
    "vintage_year",
    "foundation_year",
    "region",
    "alcohol",
    "volume",
    "other",
    "full_text",
    "label_text",
}


def _extract_matches_and_ocr(
    parsed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split response into match map + optional structured OCR block.

    Supports:
    - new: ``{"ocr": {...}, "matches": {"123": 0.9, ...}}``
    - legacy flat: ``{"123": 0.9, "456": 0.1}``
    """
    ocr_obj: dict[str, Any] | None = None
    for key in ("ocr", "query_ocr", "label_ocr"):
        raw = parsed.get(key)
        if isinstance(raw, dict) and raw:
            ocr_obj = raw
            break

    matches_raw: dict[str, Any]
    nested = parsed.get("matches")
    if isinstance(nested, dict):
        matches_raw = nested
    else:
        # Flat id→prob; ignore OCR field names if mixed
        matches_raw = {
            k: v
            for k, v in parsed.items()
            if str(k) not in _OCR_META_KEYS
        }
        # If nested ocr missing but structured fields are top-level — collect them
        if ocr_obj is None:
            maybe = {
                k: parsed[k]
                for k in (
                    "lines",
                    "producer",
                    "wine_name",
                    "color",
                    "type",
                    "cupage",
                    "coupage",
                    "vintage_year",
                    "foundation_year",
                    "region",
                    "alcohol",
                    "volume",
                    "other",
                    "full_text",
                    "label_text",
                )
                if k in parsed
            }
            if maybe:
                ocr_obj = maybe

    return matches_raw, ocr_obj


def _call_openai_text(
    *,
    api_key: str,
    prompt: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    t0 = time.perf_counter()
    url = (app_config.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")
    url = f"{url}/chat/completions"
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You compare wine label OCR texts and also structure the "
                    "query OCR like OpenAI vision OCR. "
                    "Reply with a single JSON object only: "
                    '{"ocr": {"lines": [...], "producer": ..., ...}, '
                    '"matches": {"<wine_id>": <probability 0..1>, ...}} '
                    "with EVERY candidate id under matches."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI: пустой choices")
    message = (choices[0] or {}).get("message") or {}
    raw_text = str(message.get("content") or "").strip()
    parsed = _parse_json_object(raw_text)
    usage = body.get("usage") or {}
    meta = {
        "tokens_input": usage.get("prompt_tokens"),
        "tokens_output": usage.get("completion_tokens"),
        "tokens_total": usage.get("total_tokens"),
        "openai_model": body.get("model") or model,
        "wall_ms": round(wall_ms, 1),
        "finish_reason": (choices[0] or {}).get("finish_reason"),
        "raw_content": raw_text[:4000],
    }
    return parsed, wall_ms, meta


def evaluate_openai_txt_match(
    ocr_text: str,
    wines: list[dict[str, Any]],
    *,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Compare query OCR to candidate label texts via OpenAI.

    ``wines`` must already exclude exclusive-lexicon rejects.
    """
    from app.pipeline.label_detect_llm import text_from_ocr_fields

    t0 = time.perf_counter()
    text = str(ocr_text or "").strip()
    if not text:
        return {
            "ok": False,
            "skipped": True,
            "reason": "empty_ocr",
            "ms": 0,
            "scores": [],
            "by_id": {},
            "top": [],
        }
    payload_wines = [w for w in wines if _candidate_lines(w)]
    if not payload_wines:
        return {
            "ok": False,
            "skipped": True,
            "reason": "no_candidates",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "top": [],
            "n_input": 0,
        }

    api_key = (app_config.OPENAI_KEY or "").strip()
    if not api_key:
        return {
            "ok": False,
            "error": "OPENAI_KEY / OPENAI_API_KEY не задан",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "top": [],
            "n_input": len(payload_wines),
        }

    model = (app_config.OPENAI_MODEL or "gpt-4.1-mini").strip()
    try:
        prompt, cand_payload = render_prompt(ocr_text=text, wines=payload_wines)
        parsed, wall_ms, meta = _call_openai_text(
            api_key=api_key,
            prompt=prompt,
            model=model,
            timeout=timeout,
        )
        matches_raw, ocr_obj = _extract_matches_and_ocr(parsed)
        by_id_scores = _normalize_scores(matches_raw)
    except Exception as exc:  # noqa: BLE001
        logger.exception("openai_txt_match failed")
        return {
            "ok": False,
            "error": str(exc),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "scores": [],
            "by_id": {},
            "top": [],
            "n_input": len(payload_wines),
        }

    allowed = {int(w["id"]) for w in payload_wines if w.get("id") is not None}
    by_id_scores = {i: p for i, p in by_id_scores.items() if i in allowed}
    ranked = sorted(by_id_scores.items(), key=lambda t: (-t[1], t[0]))
    scores_list = [
        {"id": wid, "llm_txt": prob, "rank": i + 1}
        for i, (wid, prob) in enumerate(ranked)
    ]
    by_id = {
        str(wid): {"id": wid, "llm_txt": prob}
        for wid, prob in by_id_scores.items()
    }

    structured_text = ""
    structured_lines: list[Any] = []
    ocr_json: dict[str, Any] | None = None
    if isinstance(ocr_obj, dict) and ocr_obj:
        ocr_json = dict(ocr_obj)
        try:
            structured_text, structured_lines = text_from_ocr_fields(ocr_json)
        except Exception:  # noqa: BLE001
            structured_text = str(
                ocr_json.get("full_text") or ocr_json.get("label_text") or ""
            ).strip()
            raw_lines = ocr_json.get("lines")
            if isinstance(raw_lines, list):
                structured_lines = raw_lines

    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "ok": True,
        "ms": ms,
        "n_input": len(payload_wines),
        "n_scored": len(by_id_scores),
        "ocr_chars": len(text),
        "scores": scores_list,
        "by_id": by_id,
        "top": [
            {"id": wid, "llm_txt": prob}
            for wid, prob in ranked[:_TOP_PREVIEW]
        ],
        "best": (
            {"id": ranked[0][0], "llm_txt": ranked[0][1]} if ranked else None
        ),
        "model": meta.get("openai_model") or model,
        # Structured OCR of query text (same schema as OpenAI vision OCR)
        "text": structured_text or None,
        "lines": structured_lines or None,
        "json": ocr_json,
        "meta": {
            k: v
            for k, v in meta.items()
            if k != "raw_content"
        },
        "candidates_sent": len(cand_payload),
        "explain": {
            "method": "OpenAI сравнение текста",
            "prompt": str(_PROMPT_PATH.name),
            "input": "query OCR vs catalog label (survivors of exclusive lexicon)",
            "output": (
                "ocr{lines+wine fields} + matches{id: probability 0..1} "
                "for all candidates"
            ),
        },
    }

def attach_openai_txt_to_candidate_items(
    candidates: dict[str, Any] | None,
    step: dict[str, Any] | None,
) -> None:
    """Мутирует candidates.*.items: llm_txt."""
    if not isinstance(candidates, dict) or not isinstance(step, dict):
        return
    by_id = step.get("by_id") or {}
    if not isinstance(by_id, dict):
        return
    for branch in ("siglip2", "dinov3"):
        block = candidates.get(branch) or {}
        items = block.get("items") if isinstance(block, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict) or it.get("id") is None:
                continue
            row = by_id.get(str(int(it["id"])))
            if not isinstance(row, dict):
                continue
            val = row.get("llm_txt")
            if val is None:
                continue
            try:
                it["llm_txt"] = round(float(val), 6)
            except (TypeError, ValueError):
                continue
