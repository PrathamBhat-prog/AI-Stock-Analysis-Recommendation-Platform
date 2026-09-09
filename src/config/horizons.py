"""
Investment horizon definitions — single source of truth for API, UI, and decision logic.

IMPORTANT: Keys are TRADING DAYS (market sessions), not calendar days.
  - 126 trading days ≈ 6 months of market activity (~180 calendar days)
  - 252 trading days ≈ 1 year of market activity (~365 calendar days)
"""

from __future__ import annotations

from typing import TypedDict


class HorizonConfig(TypedDict):
    label: str
    days: int
    trading_days: int
    calendar_approx: str
    ml_weight: float
    trend_weight: float
    ml_training_horizon_days: int
    primary_signal: str
    user_expectation: str
    honest_disclaimer: str


HORIZONS: dict[str, HorizonConfig] = {
    "5d": {
        "label": "1 week (5 trading days)",
        "days": 5,
        "trading_days": 5,
        "calendar_approx": "~7 calendar days",
        "ml_weight": 0.80,
        "trend_weight": 0.20,
        "ml_training_horizon_days": 20,
        "primary_signal": "ML (short-horizon model)",
        "user_expectation": "Short-term directional bias over the next week of trading.",
        "honest_disclaimer": (
            "ML is trained on 20 trading-day moves; this view weights that signal heavily."
        ),
    },
    "21d": {
        "label": "1 month (21 trading days)",
        "days": 21,
        "trading_days": 21,
        "calendar_approx": "~30 calendar days",
        "ml_weight": 0.50,
        "trend_weight": 0.50,
        "ml_training_horizon_days": 20,
        "primary_signal": "ML + Trend (balanced)",
        "user_expectation": "Swing outlook over ~1 month of trading sessions.",
        "honest_disclaimer": (
            "Closest to the model's 20 trading-day training horizon."
        ),
    },
    "63d": {
        "label": "3 months (63 trading days)",
        "days": 63,
        "trading_days": 63,
        "calendar_approx": "~90 calendar days",
        "ml_weight": 0.30,
        "trend_weight": 0.70,
        "ml_training_horizon_days": 20,
        "primary_signal": "Trend analysis (ML assist)",
        "user_expectation": "Medium-term trend over ~3 months of market sessions.",
        "honest_disclaimer": (
            "ML was not trained for 63-day forecasts. Trend analysis dominates."
        ),
    },
    "126d": {
        "label": "6 months of trading (126 sessions)",
        "days": 126,
        "trading_days": 126,
        "calendar_approx": "~180 calendar days (not 126 calendar days)",
        "ml_weight": 0.15,
        "trend_weight": 0.85,
        "ml_training_horizon_days": 20,
        "primary_signal": "Trend analysis",
        "user_expectation": "Longer-term trend over ~6 months of market activity.",
        "honest_disclaimer": (
            "126 means 126 trading days (~6 months of market sessions), "
            "NOT 126 calendar days. This is trend extrapolation, not a price target."
        ),
    },
    "252d": {
        "label": "1 year of trading (252 sessions)",
        "days": 252,
        "trading_days": 252,
        "calendar_approx": "~365 calendar days (not 252 calendar days)",
        "ml_weight": 0.10,
        "trend_weight": 0.90,
        "ml_training_horizon_days": 20,
        "primary_signal": "Long-term trend analysis",
        "user_expectation": "Annual trend direction over ~1 year of market sessions.",
        "honest_disclaimer": (
            "252 means 252 trading days (~1 year of market sessions), "
            "NOT 252 calendar days. ~90% trend-based — not an ML annual forecast."
        ),
    },
}

DEFAULT_HORIZON = "21d"

BUY_THRESHOLD = 0.58
SELL_THRESHOLD = 0.42


def horizon_for_api() -> dict:
    return {key: {"key": key, **cfg} for key, cfg in HORIZONS.items()}
