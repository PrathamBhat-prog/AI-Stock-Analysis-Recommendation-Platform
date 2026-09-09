"""
Build training dataset for Sniper v5 CatBoost (free data sources only).

Sources: yfinance (OHLCV + VIX), optional cached GDELT sentiment (inference-time
quality). Historical sentiment gaps are filled with 0.0 (neutral) — documented
in INTERVIEW_GUIDE.md as a known limitation with upgrade path to GDELT backfill.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.ml_config import DEFAULT_TRAIN_TICKERS, SNIPER_FORECAST_HORIZON_DAYS, TRAIN_PERIOD
from src.data.fetch_data import fetch_stock_data
from src.data.validate_data import validate_stock_data

logger = logging.getLogger(__name__)

SNIPER_FEATURE_COLS = [
    "momentum_20d", "dist_52w_high", "rolling_sentiment_20d",
    "vol_ratio_5d", "VIX", "sent_lag_1", "sent_lag_3",
    "sent_lag_5", "sent_vix_interaction", "vix_velocity",
]


def _fetch_vix_history(period: str = TRAIN_PERIOD) -> pd.DataFrame:
    vix = yf.Ticker("^VIX").history(period=period, auto_adjust=True)
    vix = vix.reset_index()
    vix["Date"] = pd.to_datetime(vix["Date"], utc=True).dt.tz_localize(None)
    vix["VIX"] = vix["Close"].astype(float)
    vix["vix_velocity"] = vix["VIX"].pct_change(5)
    return vix[["Date", "VIX", "vix_velocity"]]


def _engineer_ticker_features(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    close = df["Close"]
    vol = df["Volume"]

    df["momentum_20d"] = close.pct_change(20)
    high_52w = close.rolling(252, min_periods=20).max()
    df["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    df["vol_ratio_5d"] = vol / vol.rolling(5).mean().replace(0, np.nan)

    # Neutral sentiment prior for historical training (live GDELT at inference)
    df["rolling_sentiment_20d"] = 0.0
    for lag in (1, 3, 5):
        df[f"sent_lag_{lag}"] = 0.0

    df = df.merge(vix_df, on="Date", how="left")
    df["VIX"] = df["VIX"].ffill().fillna(20.0)
    df["vix_velocity"] = df["vix_velocity"].fillna(0.0)
    df["sent_vix_interaction"] = df["rolling_sentiment_20d"] * df["VIX"]

    horizon = SNIPER_FORECAST_HORIZON_DAYS
    df["target_up"] = (close.shift(-horizon) > close).astype(float)
    df.loc[close.shift(-horizon).isna(), "target_up"] = np.nan

    return df


def build_sniper_dataset(
    tickers: list[str] | None = None,
    period: str = TRAIN_PERIOD,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TRAIN_TICKERS
    vix_df = _fetch_vix_history(period)
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            raw = fetch_stock_data(ticker=ticker, period=period)
            raw = validate_stock_data(raw)
            feat = _engineer_ticker_features(raw, vix_df)
            feat["ticker"] = ticker
            frames.append(feat)
            logger.info("Sniper dataset: %d rows for %s", len(feat), ticker)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not frames:
        raise ValueError("No Sniper training data loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    required = SNIPER_FEATURE_COLS + ["target_up"]
    combined = combined.dropna(subset=required)
    return combined
