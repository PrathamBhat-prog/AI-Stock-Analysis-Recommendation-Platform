"""
Walk-forward backtest with transaction costs (free, no external services).

Interview: random train/test split leaks future info in time series; walk-forward
re-trains on past windows and tests on the next segment — industry standard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config.ml_config import SNIPER_CATBOOST_PARAMS, SNIPER_CONF_THRESHOLD
from src.data.sniper_dataset import SNIPER_FEATURE_COLS, build_sniper_dataset

logger = logging.getLogger(__name__)

DEFAULT_TRANSACTION_COST = 0.001  # 10 bps per trade (one-way)


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_return_per_trade: float


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-9)
    return float(dd.min())


def run_walk_forward_backtest(
    tickers: list[str] | None = None,
    period: str = "5y",
    n_splits: int = 3,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
    threshold: float = SNIPER_CONF_THRESHOLD,
) -> dict:
    """
    Simple walk-forward: divide timeline into n_splits+1 chunks;
    train on cumulative past, test on next chunk.
    """
    dataset = build_sniper_dataset(tickers=tickers, period=period)
    dataset = dataset.sort_values("Date").reset_index(drop=True)
    n = len(dataset)
    chunk = n // (n_splits + 1)

    all_returns: list[float] = []
    trades = 0
    wins = 0

    params = {**SNIPER_CATBOOST_PARAMS, "verbose": 0}

    for split in range(n_splits):
        train_end = chunk * (split + 1)
        test_end = min(chunk * (split + 2), n)
        if test_end - train_end < 50:
            continue

        train_df = dataset.iloc[:train_end]
        test_df = dataset.iloc[train_end:test_end]

        x_train = train_df[SNIPER_FEATURE_COLS]
        y_train = train_df["target_up"].astype(int)
        x_test = test_df[SNIPER_FEATURE_COLS]
        y_test = test_df["target_up"].astype(int)

        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, verbose=False)

        probas = model.predict_proba(x_test)[:, 1]
        preds = probas >= threshold

        # Strategy: long when BUY signal, flat otherwise
        close = test_df["Close"].values
        forward_ret = np.zeros(len(test_df))
        horizon = 20
        for i in range(len(test_df) - horizon):
            if preds[i]:
                ret = (close[i + horizon] - close[i]) / close[i]
                ret -= 2 * transaction_cost  # round-trip
                all_returns.append(ret)
                trades += 1
                if ret > 0:
                    wins += 1

    if not all_returns:
        return {"error": "Insufficient data for walk-forward backtest", "n_trades": 0}

    rets = np.array(all_returns)
    equity = np.cumprod(1 + rets)
    total_return = float(equity[-1] - 1)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252 / 20)) if rets.std() > 1e-9 else 0.0

    result = BacktestResult(
        total_return=round(total_return, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(_max_drawdown(equity), 4),
        win_rate=round(wins / trades, 4) if trades else 0.0,
        n_trades=trades,
        avg_return_per_trade=round(float(rets.mean()), 6),
    )

    return {
        "walk_forward_splits": n_splits,
        "transaction_cost_bps": transaction_cost * 10000,
        "threshold": threshold,
        **result.__dict__,
    }
