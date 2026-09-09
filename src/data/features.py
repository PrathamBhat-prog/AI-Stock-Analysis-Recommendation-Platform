"""
Unified feature engineering for training, inference, trend analysis, and charts.

Design choice (interview): one module, explicit column tiers
  - ML_CORE: 12 stationary features for sklearn/LSTM training
  - CHART/TREND: extra columns for UI charts and TrendAgent (no leakage into Sniper v5)
  - Sniper v5 uses separate live features in sniper_predictor.py (sentiment + VIX)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical ML feature list (cross-ticker, stationary)
FEATURE_COLUMNS = [
    "Daily_Return",
    "Return_lag_1",
    "Return_lag_5",
    "Return_std_20",
    "Momentum_20",
    "RSI_14",
    "MACD_Hist",
    "BB_Pct",
    "ATR_14_pct",
    "OBV_ROC_10",
    "Close_vs_MA20",
    "Volume_ratio_5",
]

TARGET_COLUMN = "target_up"


def add_time_series_features(
    df: pd.DataFrame,
    include_chart_columns: bool = True,
) -> pd.DataFrame:
    """
    Compute ML features + optional chart/trend columns.
    Requires: Open, High, Low, Close, Volume, Date.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # --- Core ML features ---
    df["Daily_Return"] = close.pct_change()
    df["Return_lag_1"] = df["Daily_Return"].shift(1)
    df["Return_lag_5"] = df["Daily_Return"].shift(5)
    df["Return_std_20"] = df["Daily_Return"].rolling(20).std()
    df["Momentum_20"] = close.pct_change(20)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_Signal"] = sig
    df["MACD_Hist"] = macd - sig

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_Pct"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    df["BB_Width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    df["ATR_14_pct"] = atr14 / close.replace(0, np.nan)

    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * vol).cumsum()
    df["OBV_ROC_10"] = obv.pct_change(10)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std().replace(0, np.nan)
    df["Close_vs_MA20"] = (close - ma20) / std20
    df["Volume_ratio_5"] = vol / vol.rolling(5).mean().replace(0, np.nan)
    df["Volume_ratio_20"] = vol / vol.rolling(20).mean().replace(0, np.nan)

    # Aliases used by TrendAgent / legacy agents
    df["Close_mean_20"] = ma20
    df["Close_mean_60"] = close.rolling(60).mean()
    df["SMA_20"] = ma20
    df["SMA_50"] = close.rolling(50).mean()
    df["Volatility_20"] = df["Return_std_20"]

    if include_chart_columns:
        df["BB_Upper"] = bb_upper
        df["BB_Lower"] = bb_lower

    return df


# Backward-compatible alias used by dataset builder
add_technical_indicators = add_time_series_features
