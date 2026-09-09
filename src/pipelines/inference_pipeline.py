"""End-to-end inference pipeline for any ticker and investment horizon."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager

import mlflow
import yfinance as yf

from src.agents.decision_agent import MLDecisionAgent, DEFAULT_HORIZON
from src.agents.ml_agent import MLPredictionAgent
from src.agents.risk_agent import RiskAnalysisAgent
from src.agents.trend_agent import analyze_trend
from src.config.ml_config import MIN_INFERENCE_PERIOD, MLFLOW_EXPERIMENT_INFERENCE, SHORT_PERIODS
from src.config.settings import settings
from src.data.fetch_data import fetch_stock_data
from src.data.features import add_time_series_features
from src.data.validate_data import validate_stock_data
from src.risk.position_sizer import compute_position_sizing

logger = logging.getLogger(__name__)


def _get_company_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "company_name": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {
            "company_name": ticker.upper(),
            "sector": "Unknown",
            "industry": "Unknown",
            "currency": "USD",
            "exchange": "",
            "market_cap": None,
            "pe_ratio": None,
            "52w_high": None,
            "52w_low": None,
        }


@contextmanager
def _mlflow_run(ticker: str, horizon_key: str, period: str, fetch_period: str):
    if not settings.ENABLE_INFERENCE_MLFLOW:
        yield
        return
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    mlruns_path = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_INFERENCE)
    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("period", period)
        mlflow.log_param("fetch_period", fetch_period)
        mlflow.log_param("horizon", horizon_key)
        yield


class StockAnalysisPipeline:
    """Fetch → features → ML + trend → risk → decision → position sizing."""

    def __init__(self):
        self.ml_agent = MLPredictionAgent()
        self.decision_agent = MLDecisionAgent()
        self.risk_agent = RiskAnalysisAgent()

    @staticmethod
    def _effective_period(period: str) -> str:
        if period in SHORT_PERIODS:
            return MIN_INFERENCE_PERIOD
        return period

    def run(
        self,
        ticker: str,
        period: str = "2y",
        horizon_key: str = DEFAULT_HORIZON,
    ) -> dict:
        fetch_period = self._effective_period(period)

        with _mlflow_run(ticker, horizon_key, period, fetch_period):
            df = fetch_stock_data(ticker=ticker, period=fetch_period)
            df = validate_stock_data(df)
            df = add_time_series_features(df)

            company_info = _get_company_info(ticker)

            ml_result = self.ml_agent.analyze(
                df=df,
                ticker=ticker,
                company_name=company_info.get("company_name", ticker),
            )

            trend_result = analyze_trend(df)
            risk_result = self.risk_agent.analyze(df)

            decision = self.decision_agent.decide(
                ml_result=ml_result,
                trend_result=trend_result,
                horizon_key=horizon_key,
                risk_overlay=risk_result,
            )

            position = compute_position_sizing(
                df=df,
                decision=decision["final_decision"],
                composite_score=decision["composite_score"],
            )

            explainability = _build_explainability(ml_result)

            if settings.ENABLE_INFERENCE_MLFLOW:
                mlflow.log_param("final_decision", decision["final_decision"])
                mlflow.log_metric("confidence", decision["confidence"])
                mlflow.log_metric("composite", decision["composite_score"])

            full_result = {
                **decision,
                "company": company_info,
                "trend": trend_result,
                "risk": risk_result,
                "position_sizing": position,
                "explainability": explainability,
            }

            if settings.ENABLE_INFERENCE_MLFLOW:
                artifact_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                artifact_path = os.path.join(artifact_dir, f"decision_{ticker}.json")
                with open(artifact_path, "w", encoding="utf-8") as f:
                    json.dump(full_result, f, indent=2, default=str)
                mlflow.log_artifact(artifact_path)

        return full_result


def _build_explainability(ml_result: dict) -> dict:
    drivers = []
    backend = ml_result.get("sentiment_backend", "vader")
    if ml_result.get("sentiment_score") is not None:
        drivers.append({
            "factor": f"News sentiment (GDELT/{backend.upper()})",
            "value": ml_result.get("sentiment_score"),
            "direction": "bullish" if ml_result.get("sentiment_score", 0) > 0 else "bearish",
        })
    if ml_result.get("vix") is not None:
        drivers.append({
            "factor": "VIX (market fear)",
            "value": ml_result.get("vix"),
            "direction": "elevated risk" if ml_result.get("vix", 0) > 25 else "calm",
        })
    if ml_result.get("vix_velocity") is not None:
        drivers.append({
            "factor": "VIX velocity (regime shift)",
            "value": ml_result.get("vix_velocity"),
            "direction": "rising fear" if ml_result.get("vix_velocity", 0) > 0 else "stable",
        })

    return {
        "top_drivers": drivers,
        "model_probability_up": ml_result.get("probability_up"),
        "model_name": ml_result.get("model_name"),
        "model_path": ml_result.get("model_path"),
        "note": (
            "Long horizons lean on trend analysis; ML explains short-term directional bias (~20d)."
        ),
    }
