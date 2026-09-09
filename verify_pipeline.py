"""Smoke test — no network required for import checks; optional live test with --live."""

import argparse
import sys


def test_imports() -> None:
    from src.config.horizons import HORIZONS, DEFAULT_HORIZON
    from src.data.features import add_time_series_features, FEATURE_COLUMNS
    from src.agents.decision_agent import MLDecisionAgent
    from src.risk.position_sizer import compute_position_sizing
    import pandas as pd
    import numpy as np

    assert DEFAULT_HORIZON in HORIZONS
    assert len(FEATURE_COLUMNS) == 12

    n = 100
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n),
        "Open": np.linspace(100, 120, n),
        "High": np.linspace(101, 121, n),
        "Low": np.linspace(99, 119, n),
        "Close": np.linspace(100, 120, n),
        "Volume": np.random.randint(1_000_000, 2_000_000, n),
    })
    out = add_time_series_features(df)
    assert "RSI_14" in out.columns
    assert "MACD" in out.columns

    agent = MLDecisionAgent()
    decision = agent.decide(
        ml_result={"available": True, "probability_up": 0.65, "model_name": "test", "forecast_horizon_days": 20},
        trend_result={"trend_score": 0.3, "trend_label": "Uptrend", "summary": "ok"},
        horizon_key="252d",
    )
    assert "honest_disclaimer" in decision
    assert decision["horizon_key"] == "252d"

    pos = compute_position_sizing(out, "BUY", 0.65)
    assert "suggested_portfolio_pct" in pos
    print("OK: imports and unit smoke tests passed")


def test_live(ticker: str = "AAPL") -> None:
    from src.pipelines.inference_pipeline import StockAnalysisPipeline

    pipe = StockAnalysisPipeline()
    result = pipe.run(ticker=ticker, period="1y", horizon_key="21d")
    assert result["final_decision"] in ("BUY", "SELL", "HOLD")
    assert "honest_disclaimer" in result
    print(f"OK: live analysis for {ticker} → {result['final_decision']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Run live yfinance analysis")
    p.add_argument("--ticker", default="AAPL")
    args = p.parse_args()
    try:
        test_imports()
        if args.live:
            test_live(args.ticker)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
