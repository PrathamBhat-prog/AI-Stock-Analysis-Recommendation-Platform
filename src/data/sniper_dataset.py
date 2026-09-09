"""
Build training dataset for Sniper v5 CatBoost (free data sources only).

Sentiment architecture:
  proxy (default) — price/volume proxy + 1 yfinance news call per ticker
  inference_only — neutral zeros (fast baseline)
  lite / full — optional historical GDELT backfill for research
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.ml_config import (
    DEFAULT_TRAIN_TICKERS,
    SENTIMENT_BACKFILL_STRIDE,
    SENTIMENT_LITE_MAX_SAMPLES,
    SENTIMENT_LITE_RECENT_YEARS,
    SENTIMENT_LITE_STRIDE,
    SNIPER_FORECAST_HORIZON_DAYS,
    TRAIN_PERIOD,
)
from src.data.fetch_data import fetch_stock_data
from src.data.sentiment_backfill import attach_sentiment_features
from src.data.sentiment_proxy import attach_proxy_sentiment
from src.data.validate_data import validate_stock_data

logger = logging.getLogger(__name__)

SNIPER_FEATURE_COLS = [
    "momentum_20d", "dist_52w_high", "rolling_sentiment_20d",
    "vol_ratio_5d", "VIX", "sent_lag_1", "sent_lag_3",
    "sent_lag_5", "sent_vix_interaction", "vix_velocity",
]

# Default training path — no GDELT historical API
DEFAULT_SENTIMENT_MODE = "proxy"


def _fetch_vix_history(period: str = TRAIN_PERIOD) -> pd.DataFrame:
    vix = yf.Ticker("^VIX").history(period=period, auto_adjust=True)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"], utc=True).dt.tz_localize(None)
    vix["VIX"] = vix["Close"].astype(float)
    vix["vix_velocity"] = vix["VIX"].pct_change(5)
    return vix[["Date", "VIX", "vix_velocity"]]


def _attach_sentiment(
    df: pd.DataFrame,
    ticker: str,
    sentiment_mode: str,
) -> pd.DataFrame:
    if sentiment_mode == "proxy":
        return attach_proxy_sentiment(df, ticker)

    if sentiment_mode in ("inference_only", "none"):
        df = df.copy()
        for col in ("rolling_sentiment_20d", "sent_lag_1", "sent_lag_3", "sent_lag_5"):
            df[col] = 0.0
        df["sent_vix_interaction"] = 0.0
        return df

    if sentiment_mode in ("lite", "gdelt_lite"):
        logger.warning("%s: historical GDELT lite mode is slower than proxy", ticker)
        lite_kw = {
            "stride": SENTIMENT_LITE_STRIDE,
            "max_samples": SENTIMENT_LITE_MAX_SAMPLES,
            "recent_years": SENTIMENT_LITE_RECENT_YEARS,
        }
        return attach_sentiment_features(df, ticker=ticker, backfill=True, **lite_kw)

    if sentiment_mode in ("full", "gdelt_full"):
        logger.warning("%s: full historical GDELT backfill is slower than proxy", ticker)
        return attach_sentiment_features(
            df, ticker=ticker, backfill=True, stride=SENTIMENT_BACKFILL_STRIDE,
        )

    return attach_proxy_sentiment(df, ticker)


def _engineer_ticker_features(
    df: pd.DataFrame,
    vix_df: pd.DataFrame,
    ticker: str,
    sentiment_mode: str = DEFAULT_SENTIMENT_MODE,
) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    close = df["Close"]
    vol = df["Volume"]

    df["momentum_20d"] = close.pct_change(20)
    high_52w = close.rolling(252, min_periods=20).max()
    df["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    df["vol_ratio_5d"] = vol / vol.rolling(5).mean().replace(0, np.nan)

    df = df.merge(vix_df, on="Date", how="left")
    df["VIX"] = df["VIX"].ffill().fillna(20.0)
    df["vix_velocity"] = df["vix_velocity"].fillna(0.0)

    df = _attach_sentiment(df, ticker, sentiment_mode)

    horizon = SNIPER_FORECAST_HORIZON_DAYS
    df["target_up"] = (close.shift(-horizon) > close).astype(float)
    df.loc[close.shift(-horizon).isna(), "target_up"] = np.nan

    return df


def build_sniper_dataset(
    tickers: list[str] | None = None,
    period: str = TRAIN_PERIOD,
    backfill_sentiment: bool = True,  # kept for API compat; mode drives behaviour
    sentiment_mode: str = DEFAULT_SENTIMENT_MODE,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TRAIN_TICKERS
    vix_df = _fetch_vix_history(period)
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            raw = fetch_stock_data(ticker=ticker, period=period)
            raw = validate_stock_data(raw)
            feat = _engineer_ticker_features(raw, vix_df, ticker=ticker, sentiment_mode=sentiment_mode)
            feat["ticker"] = ticker
            frames.append(feat)
            logger.info("Sniper dataset: %d rows for %s", len(feat), ticker)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not frames:
        raise ValueError("No Sniper training data loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "Date"]).reset_index(drop=True)
    required = SNIPER_FEATURE_COLS + ["target_up"]
    combined = combined.dropna(subset=required)
    for col in SNIPER_FEATURE_COLS:
        lo, hi = combined[col].quantile([0.01, 0.99])
        combined[col] = combined[col].clip(lo, hi)
    return combined
