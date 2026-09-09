import numpy as np
import pandas as pd

from src.data.features import add_time_series_features, FEATURE_COLUMNS


def test_feature_columns_present():
    n = 80
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n),
        "Open": np.linspace(100, 130, n),
        "High": np.linspace(101, 131, n),
        "Low": np.linspace(99, 129, n),
        "Close": np.linspace(100, 130, n),
        "Volume": np.random.randint(1e6, 2e6, n),
    })
    out = add_time_series_features(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert "MACD" in out.columns
    assert "Close_mean_20" in out.columns
