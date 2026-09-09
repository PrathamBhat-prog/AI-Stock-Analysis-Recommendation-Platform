"""
Inverse-volatility position sizing (documented in README).

Formula (simplified, educational):
    suggested_weight = base_allocation * min(max_ratio, target_vol / realized_vol)

Why inverse-vol?
- Equal dollar bets on a volatile meme stock vs a stable utility are not equal risk.
- Scaling down in high-vol regimes reduces drawdowns in backtests (see README claims).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BASE_ALLOCATION = 0.10   # 10% of portfolio per idea
DEFAULT_TARGET_VOL = 0.015      # ~1.5% daily vol target
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.20
MAX_VOL_RATIO = 3.0


def _realized_volatility(df: pd.DataFrame, window: int = 20) -> float:
    if "Daily_Return" in df.columns:
        returns = df["Daily_Return"].dropna()
    else:
        returns = df["Close"].pct_change().dropna()
    if len(returns) < 5:
        return DEFAULT_TARGET_VOL
    vol = float(returns.tail(window).std())
    return vol if vol > 1e-9 else DEFAULT_TARGET_VOL


def compute_position_sizing(
    df: pd.DataFrame,
    decision: str,
    composite_score: float,
    base_allocation: float = DEFAULT_BASE_ALLOCATION,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> dict:
    """
    Return suggested portfolio weight and risk metadata.

    HOLD / SELL still get sizing info (often 0 or reduced) for transparency.
    """
    realized_vol = _realized_volatility(df)
    vol_ratio = min(target_vol / realized_vol, MAX_VOL_RATIO)
    raw_weight = base_allocation * vol_ratio

    if decision == "BUY":
        # Scale conviction: stronger composite → slightly larger (capped)
        conviction = (composite_score - 0.5) * 2.0  # 0 at 50%, 1 at 100%
        conviction = float(np.clip(conviction, 0.0, 1.0))
        suggested = raw_weight * (0.5 + 0.5 * conviction)
    elif decision == "SELL":
        suggested = 0.0
    else:
        suggested = 0.0

    if decision != "BUY":
        suggested = 0.0
    else:
        suggested = float(np.clip(suggested, 0.0, MAX_WEIGHT))
        # Only suggest a position if conviction clears a minimum bar
        if suggested < MIN_WEIGHT:
            suggested = 0.0

    vol_label = (
        "High" if realized_vol > 0.025 else
        "Low" if realized_vol < 0.008 else
        "Normal"
    )

    return {
        "suggested_portfolio_weight": round(suggested, 4),
        "suggested_portfolio_pct": round(suggested * 100, 2),
        "base_allocation": base_allocation,
        "target_daily_vol": target_vol,
        "realized_daily_vol_20d": round(realized_vol, 6),
        "volatility_regime": vol_label,
        "inverse_vol_scalar": round(vol_ratio, 4),
        "method": "inverse_volatility",
        "explanation": (
            f"Suggested allocation {suggested*100:.1f}% of portfolio "
            f"(base {base_allocation*100:.0f}% scaled by inverse vol; "
            f"20d realized vol {realized_vol*100:.2f}%, regime={vol_label})."
        ),
    }
