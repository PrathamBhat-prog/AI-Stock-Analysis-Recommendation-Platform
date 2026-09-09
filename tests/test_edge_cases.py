import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.config.horizons import DEFAULT_HORIZON, HORIZONS
from src.data.features import add_time_series_features
from src.risk.position_sizer import compute_position_sizing

if importlib.util.find_spec("yfinance") is None:
    pytest.skip("yfinance not installed", allow_module_level=True)

from src.models.sniper_predictor import SniperPredictor, _build_ohlcv_features


def test_default_horizon_is_one_month():
    assert DEFAULT_HORIZON == "21d"
    assert HORIZONS[DEFAULT_HORIZON]["ml_weight"] == 0.50


def test_obv_roc_handles_zero_obv():
    n = 50
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n),
        "Open": np.ones(n),
        "High": np.ones(n) * 1.01,
        "Low": np.ones(n) * 0.99,
        "Close": np.ones(n),
        "Volume": np.zeros(n),  # zero volume → OBV stays 0
    })
    out = add_time_series_features(df)
    assert not np.isinf(out["OBV_ROC_10"].dropna()).any()


def test_sniper_predict_rejects_short_history():
    predictor = SniperPredictor()
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10),
        "Open": range(10),
        "High": range(10),
        "Low": range(10),
        "Close": range(10),
        "Volume": [1_000_000] * 10,
    })
    with pytest.raises(ValueError, match="at least 30 days"):
        predictor.predict("AAPL", df)


def test_ohlcv_features_nan_safe():
    df = pd.DataFrame({
        "Close": [100.0] * 25,
        "Volume": [1_000_000] * 25,
    })
    feats = _build_ohlcv_features(df)
    assert all(feats[k] == feats[k] for k in feats)  # no NaNs


def test_weak_buy_gets_zero_position():
    df = pd.DataFrame({
        "Close": np.linspace(100, 110, 30),
        "Daily_Return": [0.001] * 30,
    })
    # composite barely above 0.5 → low conviction → below MIN_WEIGHT threshold
    r = compute_position_sizing(df, "BUY", 0.51)
    assert r["suggested_portfolio_weight"] == 0.0
