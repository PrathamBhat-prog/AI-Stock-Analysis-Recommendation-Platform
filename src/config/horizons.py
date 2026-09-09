"""
Investment horizon definitions — single source of truth for API, UI, and decision logic.

Product engineering note
------------------------
The ML model (Sniper v5 CatBoost) is trained to predict whether price will be
*higher after ~20 trading days*. Longer horizons (3m–1y) blend that short-horizon
ML signal with a rule-based trend agent. We surface this honestly to users so
they understand what they are (and are not) getting.
"""

from __future__ import annotations

from typing import TypedDict


class HorizonConfig(TypedDict):
    label: str
    days: int
    ml_weight: float
    trend_weight: float
    ml_training_horizon_days: int
    primary_signal: str
    user_expectation: str
    honest_disclaimer: str


HORIZONS: dict[str, HorizonConfig] = {
    "5d": {
        "label": "1 Week (~5 trading days)",
        "days": 5,
        "ml_weight": 0.80,
        "trend_weight": 0.20,
        "ml_training_horizon_days": 20,
        "primary_signal": "ML (short-horizon model)",
        "user_expectation": (
            "Best for traders asking: 'Is there a bullish bias over the next few days?'"
        ),
        "honest_disclaimer": (
            "The ML model is trained on 20-day moves; the 1-week view weights that "
            "signal heavily plus near-term trend confirmation."
        ),
    },
    "21d": {
        "label": "1 Month (~21 trading days)",
        "days": 21,
        "ml_weight": 0.50,
        "trend_weight": 0.50,
        "ml_training_horizon_days": 20,
        "primary_signal": "ML + Trend (balanced)",
        "user_expectation": (
            "Best for swing traders asking: 'Will this stock likely be higher in about a month?'"
        ),
        "honest_disclaimer": (
            "Roughly aligned with the model's 20-day training horizon, blended equally "
            "with medium-term trend analysis."
        ),
    },
    "63d": {
        "label": "3 Months (~63 trading days)",
        "days": 63,
        "ml_weight": 0.30,
        "trend_weight": 0.70,
        "ml_training_horizon_days": 20,
        "primary_signal": "Trend analysis (ML assist)",
        "user_expectation": (
            "Best for investors asking: 'Is the medium-term trend supportive?'"
        ),
        "honest_disclaimer": (
            "ML was not trained for 3-month forecasts. Trend indicators dominate; "
            "ML provides a short-term directional nudge only."
        ),
    },
    "126d": {
        "label": "6 Months (~126 trading days)",
        "days": 126,
        "ml_weight": 0.15,
        "trend_weight": 0.85,
        "ml_training_horizon_days": 20,
        "primary_signal": "Trend analysis",
        "user_expectation": (
            "Best for position investors asking: 'Is the stock in a sustained uptrend or downtrend?'"
        ),
        "honest_disclaimer": (
            "This is a trend-regime estimate, not a precision 6-month price target."
        ),
    },
    "252d": {
        "label": "1 Year (~252 trading days)",
        "days": 252,
        "ml_weight": 0.10,
        "trend_weight": 0.90,
        "ml_training_horizon_days": 20,
        "primary_signal": "Long-term trend analysis",
        "user_expectation": (
            "Best for buy-and-hold investors asking: 'What is the long-term trend direction?'"
        ),
        "honest_disclaimer": (
            "A 1-year market view cannot be predicted with ML precision from daily data alone. "
            "This horizon is ~90% trend-based extrapolation — use for directional context, "
            "not as a guaranteed annual return forecast."
        ),
    },
}

DEFAULT_HORIZON = "21d"

BUY_THRESHOLD = 0.58
SELL_THRESHOLD = 0.42


def horizon_for_api() -> dict:
    """Serialize horizons for GET /horizons (product copy included)."""
    return {
        key: {
            "key": key,
            **cfg,
        }
        for key, cfg in HORIZONS.items()
    }
