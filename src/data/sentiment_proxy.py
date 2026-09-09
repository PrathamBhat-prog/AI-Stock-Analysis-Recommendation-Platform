"""
Training-time sentiment without GDELT historical API calls.

GDELT rate-limits (HTTP 429) make multi-year backfill impractical locally.
Architecture:
  TRAINING  → market-derived proxy + optional 1× yfinance news call per ticker
  INFERENCE → live GDELT (+ yfinance fallback) in sniper_predictor.py

Proxy formula (per row, backward-looking only):
  tanh(5d_return × k) × (1 + volume_zscore)  clipped to [-1, 1]
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.sentiment_scorer import score_headlines

logger = logging.getLogger(__name__)


def _yfinance_headline_score(ticker: str) -> float | None:
    """One yfinance call per ticker — recent headlines only, no GDELT."""
    try:
        import yfinance as yf

        news = yf.Ticker(ticker).news or []
        titles = [n.get("title", "") for n in news[:25] if n.get("title")]
        if not titles:
            return None
        return score_headlines(titles)
    except Exception as exc:
        logger.debug("yfinance news for %s: %s", ticker, exc)
        return None


def attach_proxy_sentiment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Add rolling_sentiment_20d, sent_lag_*, sent_vix_interaction from price/volume proxy.
    Blends in recent yfinance headline score for the last ~20 rows when available.
    """
    out = df.copy()
    close = out["Close"]
    vol = out["Volume"]

    ret5 = close.pct_change(5).fillna(0.0)
    vol_std = vol.rolling(20, min_periods=5).std().replace(0, np.nan)
    vol_z = ((vol - vol.rolling(20, min_periods=5).mean()) / vol_std).fillna(0.0).clip(-3, 3)
    daily = np.tanh(ret5 * 8.0) * (1.0 + 0.15 * vol_z)
    daily = pd.Series(daily, index=out.index).clip(-1.0, 1.0)

    recent = _yfinance_headline_score(ticker)
    if recent is not None:
        n_blend = min(20, len(daily))
        blend_w = np.linspace(0.3, 1.0, n_blend)
        tail_idx = daily.index[-n_blend:]
        daily.loc[tail_idx] = daily.loc[tail_idx] * (1 - blend_w) + recent * blend_w
        logger.info("%s proxy sentiment: blended yfinance score %+.3f into last %d rows", ticker, recent, n_blend)

    out["rolling_sentiment_20d"] = daily.rolling(20, min_periods=1).mean()
    out["sent_lag_1"] = daily.shift(1).fillna(0.0)
    out["sent_lag_3"] = daily.shift(3).fillna(0.0)
    out["sent_lag_5"] = daily.shift(5).fillna(0.0)
    if "VIX" in out.columns:
        out["sent_vix_interaction"] = out["rolling_sentiment_20d"] * out["VIX"]
    else:
        out["sent_vix_interaction"] = 0.0

    return out
