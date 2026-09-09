from src.config.horizons import HORIZONS, DEFAULT_HORIZON, horizon_for_api
from src.agents.decision_agent import MLDecisionAgent


def test_one_year_horizon_is_trend_dominant():
    cfg = HORIZONS["252d"]
    assert cfg["trend_weight"] >= 0.85
    assert "trading" in cfg["label"].lower() or "252" in cfg["honest_disclaimer"]


def test_126d_is_not_126_calendar_days():
    cfg = HORIZONS["126d"]
    assert cfg["trading_days"] == 126
    assert "180" in cfg["calendar_approx"]


def test_decision_includes_disclaimer():
    agent = MLDecisionAgent()
    d = agent.decide(
        ml_result={"available": True, "probability_up": 0.6, "model_name": "test", "forecast_horizon_days": 20},
        trend_result={"trend_score": 0.2, "trend_label": "Uptrend", "summary": ""},
        horizon_key="252d",
    )
    assert d["honest_disclaimer"]
    assert d["horizon_key"] == "252d"


def test_horizon_api_payload():
    api = horizon_for_api()
    assert DEFAULT_HORIZON in api
    assert "calendar_approx" in api[DEFAULT_HORIZON]
