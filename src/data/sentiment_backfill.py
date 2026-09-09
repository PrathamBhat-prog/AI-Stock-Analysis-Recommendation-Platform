"""
Historical GDELT sentiment backfill for training.

GDELT is rate-limited, so we sample every `stride` trading days, score
headlines, write to SQLite, then forward-fill between samples.
Re-runs are cache-hit only.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from src.config.ml_config import SENTIMENT_BACKFILL_STRIDE
from src.data.news_fetcher import fetch_gdelt_headlines_for_range
from src.data.sentiment_cache import (
    apply_sentiment_to_frame,
    get_sentiment,
    save_sentiment,
)
from src.data.sentiment_scorer import score_headlines

logger = logging.getLogger(__name__)


def _query_for_ticker(ticker: str) -> str:
    clean = ticker.replace(".NS", "").replace(".AS", "").replace(".KS", "")
    return f"{clean} stock"


def backfill_ticker_sentiment(
    ticker: str,
    trading_dates: list[date],
    stride: int = SENTIMENT_BACKFILL_STRIDE,
    max_samples: int | None = None,
    recent_years: int | None = None,
) -> int:
    """
    Fetch GDELT for sampled dates. Returns number of new API-backed rows written.
    """
    dates = sorted({d if isinstance(d, date) else pd.Timestamp(d).date() for d in trading_dates})
    if not dates:
        return 0
    if recent_years is not None and recent_years > 0:
        cutoff = dates[-1] - timedelta(days=365 * recent_years)
        dates = [d for d in dates if d >= cutoff]
    query = _query_for_ticker(ticker)
    sampled = dates[:: max(stride, 1)]
    if max_samples is not None and len(sampled) > max_samples:
        step = max(len(sampled) // max_samples, 1)
        sampled = sampled[::step][:max_samples]
    if sampled[-1] != dates[-1]:
        sampled.append(dates[-1])

    written = 0
    for i, day in enumerate(sampled, start=1):
        if get_sentiment(ticker, day) is not None:
            continue
        window_end = day
        window_start = day - timedelta(days=6)
        headlines = fetch_gdelt_headlines_for_range(query, window_start, window_end)
        score = score_headlines(headlines)
        save_sentiment(ticker, day, score, len(headlines))
        written += 1
        if i % 10 == 0 or i == len(sampled):
            logger.info(
                "%s GDELT backfill progress: %d/%d sampled dates (%d new writes)",
                ticker, i, len(sampled), written,
            )
    logger.info("%s GDELT backfill complete: %d new samples / %d sampled dates", ticker, written, len(sampled))
    return written


def attach_sentiment_features(
    df: pd.DataFrame,
    ticker: str,
    backfill: bool = True,
    stride: int = SENTIMENT_BACKFILL_STRIDE,
    max_samples: int | None = None,
    recent_years: int | None = None,
) -> pd.DataFrame:
    """Optionally backfill GDELT, then attach rolling/lag sentiment columns from cache."""
    dates = pd.to_datetime(df["Date"]).dt.date.tolist()
    if backfill:
        backfill_ticker_sentiment(
            ticker, dates, stride=stride,
            max_samples=max_samples, recent_years=recent_years,
        )
    return apply_sentiment_to_frame(df, ticker)
