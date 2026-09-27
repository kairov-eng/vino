"""Cross-Encoder text matcher HTTP service (CPU)."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

MODEL_DIR = Path(os.environ.get("CRENC_MODEL_DIR", "/model")).resolve()
API_KEY = os.environ.get("CRENC_API_KEY") or os.environ.get("EMBED_API_KEY", "")
DEVICE = os.environ.get("DEVICE", "cpu")
TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
MAX_LENGTH = int(os.environ.get("CRENC_MAX_LENGTH", "160"))
DEFAULT_BATCH = int(os.environ.get("CRENC_BATCH_SIZE", "32"))

torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.enabled = False

_state: dict[str, Any] = {}


def _check_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _load_threshold(dir_path: Path) -> float:
    metrics_path = dir_path / "metrics.json"
    if not metrics_path.is_file():
        return 0.26
    try:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        t = raw.get("threshold_best_f1_validation")
        if t is None:
            t = (raw.get("validation") or {}).get("threshold")
        if t is not None:
            return float(t)
    except Exception:  # noqa: BLE001
        return 0.26
    return 0.26


class DocItem(BaseModel):
    text: str
    id: int | str | None = None


class PairItem(BaseModel):
    text1: str
    text2: str
    id: int | str | None = None


class MatchRequest(BaseModel):
    """Либо pairs, либо query + documents."""

    pairs: list[PairItem] | list[list[str]] | None = None
    query: str | None = None
    documents: list[DocItem] | None = None
    batch_size: int | None = Field(default=None, ge=1, le=256)

    @model_validator(mode="after")
    def _require_input(self) -> MatchRequest:
        has_pairs = bool(self.pairs)
        has_qd = self.query is not None and bool(self.documents)
        if has_pairs == has_qd:
            if not has_pairs:
                raise ValueError("provide pairs or query+documents")
            raise ValueError("provide either pairs or query+documents, not both")
        return self


def _resolve_pairs(body: MatchRequest) -> list[tuple[Any, str, str]]:
    """[(id|None, text1, text2), ...]"""
    out: list[tuple[Any, str, str]] = []
    if body.pairs:
        for p in body.pairs:
            if isinstance(p, PairItem):
                out.append((p.id, p.text1, p.text2))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((None, str(p[0]), str(p[1])))
            else:
                raise HTTPException(status_code=400, detail="invalid pair item")
        return out
    assert body.query is not None and body.documents is not None
    q = body.query
    return [(d.id, q, d.text) for d in body.documents]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(1)

    t0 = time.perf_counter()
    if not (MODEL_DIR / "config.json").is_file():
        raise RuntimeError(f"CrossEncoder model not found: {MODEL_DIR}")

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(str(MODEL_DIR), max_length=MAX_LENGTH, device=DEVICE)
    threshold = _load_threshold(MODEL_DIR)

    # warmup до первого HTTP-запроса
    _ = model.predict([["warmup query", "warmup document"]], show_progress_bar=False)

    _state["model"] = model
    _state["threshold"] = threshold
    _state["load_sec"] = round(time.perf_counter() - t0, 3)
    _state["model_dir"] = str(MODEL_DIR)
    print(
        f"loaded CrossEncoder dir={MODEL_DIR} threshold={threshold} "
        f"max_length={MAX_LENGTH} load_sec={_state['load_sec']}",
        flush=True,
    )
    yield
    _state.clear()


app = FastAPI(
    title="CrossEncoder Matcher",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    root_path="/api_cross_encoder_matcher",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if "model" in _state else "loading",
        "model_dir": _state.get("model_dir"),
        "device": DEVICE,
        "threshold": _state.get("threshold"),
        "max_length": MAX_LENGTH,
        "load_sec": _state.get("load_sec"),
        "torch_threads": TORCH_THREADS,
    }


@app.post("/v1/match")
def match(
    body: MatchRequest,
    _: None = Depends(_check_key),
) -> dict[str, Any]:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")

    resolved = _resolve_pairs(body)
    if not resolved:
        return {
            "scores": [],
            "items": [],
            "threshold": _state["threshold"],
            "timings_ms": {"infer": 0.0, "total": 0.0},
            "n": 0,
        }

    batch_size = body.batch_size or DEFAULT_BATCH
    pairs = [[t1, t2] for _, t1, t2 in resolved]

    t_total = time.perf_counter()
    t0 = time.perf_counter()
    try:
        probs = _state["model"].predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"match failed: {exc}") from exc
    infer_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000

    scores = [round(float(p), 6) for p in probs]
    thr = float(_state["threshold"])
    items: list[dict[str, Any]] = []
    for (pid, _t1, _t2), score in zip(resolved, scores):
        row: dict[str, Any] = {
            "score": score,
            "suitable": score >= thr,
        }
        if pid is not None:
            row["id"] = pid
        items.append(row)

    return {
        "scores": scores,
        "items": items,
        "threshold": thr,
        "n": len(scores),
        "timings_ms": {
            "infer": round(infer_ms, 1),
            "total": round(total_ms, 1),
        },
    }
