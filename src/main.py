"""FastAPI backend for stock analysis."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.horizons import DEFAULT_HORIZON, HORIZONS, horizon_for_api
from src.config.settings import settings
from src.pipelines.inference_pipeline import StockAnalysisPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Stock Analyser API",
    description=(
        "AI-powered BUY/SELL/HOLD for globally listed stocks. "
        "ML model trained on ~20-day direction; longer horizons blend trend analysis. "
        "See GET /horizons for honest capability descriptions."
    ),
    version="5.0.0",
)

_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = StockAnalysisPipeline()
_rate_limit: dict[str, list[float]] = {}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_limit.setdefault(client, [])
    window[:] = [t for t in window if now - t < 60]
    if len(window) >= settings.API_RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    window.append(now)
    return await call_next(request)


class AnalyseRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    period: str = "2y"
    horizon_key: str = DEFAULT_HORIZON


class BatchRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=20)
    period: str = "2y"
    horizon_key: str = DEFAULT_HORIZON


@app.get("/health")
def health():
    return {"status": "ok", "version": "5.0.0"}


@app.get("/horizons")
def list_horizons():
    """Horizons with product-engineering honesty copy."""
    return {"horizons": horizon_for_api(), "default": DEFAULT_HORIZON}


@app.post("/analyze")
def analyze(request: AnalyseRequest):
    if request.horizon_key not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon_key. Choose from: {list(HORIZONS.keys())}",
        )
    try:
        return pipeline.run(
            ticker=request.ticker.upper().strip(),
            period=request.period,
            horizon_key=request.horizon_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis failed for %s", request.ticker)
        raise HTTPException(status_code=500, detail="Analysis failed. Check ticker and try again.")


@app.post("/analyze/batch")
def analyze_batch(request: BatchRequest):
    if request.horizon_key not in HORIZONS:
        raise HTTPException(status_code=400, detail="Invalid horizon_key")
    results = []
    for ticker in request.tickers:
        t = ticker.upper().strip()
        try:
            r = pipeline.run(ticker=t, period=request.period, horizon_key=request.horizon_key)
            results.append({"ticker": t, "ok": True, "result": r})
        except Exception as exc:
            results.append({"ticker": t, "ok": False, "error": str(exc)})
    return {"results": results, "count": len(results)}
