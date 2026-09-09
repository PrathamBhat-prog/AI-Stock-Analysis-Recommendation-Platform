"""Train/val/test splitting utilities (no heavy data dependencies)."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global time-ordered split."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Split produced an empty partition; use more data or adjust ratios.")

    return train_df, val_df, test_df


def per_ticker_chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each ticker's own timeline, then pool.
    Prevents newer listings from leaking their full history into test.
    """
    if "ticker" not in df.columns:
        return chronological_split(df, train_ratio=train_ratio, val_ratio=val_ratio)

    trains: list[pd.DataFrame] = []
    vals: list[pd.DataFrame] = []
    tests: list[pd.DataFrame] = []

    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        if train_end < 10 or val_end <= train_end or n - val_end < 5:
            logger.warning("Skipping split for %s — not enough rows (%d)", ticker, n)
            continue
        trains.append(group.iloc[:train_end])
        vals.append(group.iloc[train_end:val_end])
        tests.append(group.iloc[val_end:])

    if not trains or not vals or not tests:
        logger.warning("Per-ticker split failed; falling back to global chronological split")
        return chronological_split(df.sort_values("Date"), train_ratio, val_ratio)

    train_df = pd.concat(trains, ignore_index=True).sort_values("Date")
    val_df = pd.concat(vals, ignore_index=True).sort_values("Date")
    test_df = pd.concat(tests, ignore_index=True).sort_values("Date")
    return train_df, val_df, test_df
