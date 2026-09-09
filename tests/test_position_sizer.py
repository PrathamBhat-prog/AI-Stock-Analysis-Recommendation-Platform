import numpy as np
import pandas as pd

from src.risk.position_sizer import compute_position_sizing


def test_buy_gets_positive_weight():
    df = pd.DataFrame({
        "Close": np.linspace(100, 110, 30),
        "Daily_Return": [0.001] * 30,
    })
    r = compute_position_sizing(df, "BUY", 0.65)
    assert r["suggested_portfolio_weight"] > 0


def test_hold_gets_zero_weight():
    df = pd.DataFrame({"Close": [100.0], "Daily_Return": [0.01]})
    r = compute_position_sizing(df, "HOLD", 0.5)
    assert r["suggested_portfolio_weight"] == 0
