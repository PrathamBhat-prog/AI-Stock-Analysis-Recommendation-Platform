import logging

import pandas as pd

from src.config.ml_config import DEFAULT_TRAIN_TICKERS, FORECAST_HORIZON_DAYS, TRAIN_PERIOD
from src.data.fetch_data import fetch_stock_data
from src.data.features import add_technical_indicators
from src.data.labeling import add_direction_label
from src.data.splits import chronological_split, per_ticker_chronological_split
from src.data.validate_data import validate_stock_data
from src.models.feature_columns import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)

__all__ = [
    "build_ticker_dataset",
    "build_training_dataset",
    "chronological_split",
    "per_ticker_chronological_split",
]


def build_ticker_dataset(ticker: str, period: str = TRAIN_PERIOD) -> pd.DataFrame:
    """Fetch, validate, featurize, and label data for one ticker."""
    df = fetch_stock_data(ticker=ticker, period=period)
    df = validate_stock_data(df)
    df = add_technical_indicators(df)
    df = add_direction_label(df, horizon=FORECAST_HORIZON_DAYS, target_col=TARGET_COLUMN)
    df["ticker"] = ticker
    return df


def build_training_dataset(
    tickers: list[str] | None = None,
    period: str = TRAIN_PERIOD,
) -> pd.DataFrame:
    """Build pooled multi-ticker dataset for ML training."""
    tickers = tickers or DEFAULT_TRAIN_TICKERS
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            df = build_ticker_dataset(ticker=ticker, period=period)
            frames.append(df)
            logger.info("Loaded %s rows for %s", len(df), ticker)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not frames:
        raise ValueError("No training data could be loaded for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], utc=True).dt.tz_localize(None)
    combined = combined.sort_values(["ticker", "Date"]).reset_index(drop=True)

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    combined = combined.dropna(subset=required)

    return combined
