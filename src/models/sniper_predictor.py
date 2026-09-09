# -*- coding: utf-8 -*-
"""Sniper v5 Production Predictor — CatBoostClassifier + live sentiment/VIX."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

import json
from pathlib import Path

from src.config.ml_config import SNIPER_CONF_THRESHOLD, SNIPER_FORECAST_HORIZON_DAYS
from src.data.news_fetcher import fetch_headlines
from src.data.sentiment_cache import build_sentiment_features, save_sentiment
from src.data.sentiment_scorer import score_headlines, _backend
from src.models.model_io import load_catboost_model, resolve_sniper_model_path

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "momentum_20d", "dist_52w_high", "rolling_sentiment_20d",
    "vol_ratio_5d", "VIX", "sent_lag_1", "sent_lag_3",
    "sent_lag_5", "sent_vix_interaction", "vix_velocity",
]


def _fetch_vix() -> tuple[float, float]:
    try:
        vix_df = yf.Ticker("^VIX").history(period="1mo", auto_adjust=True)
        vix = float(vix_df["Close"].iloc[-1])
        vel = float(vix_df["Close"].pct_change(5).iloc[-1])
        return vix, vel
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return 20.0, 0.0


def _safe_float(val: float, default: float = 0.0) -> float:
    return default if val != val else float(val)


def _build_ohlcv_features(df: pd.DataFrame) -> dict:
    close = df["Close"]
    vol = df["Volume"]
    momentum_20d = _safe_float(close.pct_change(20).iloc[-1])
    high_52w = float(close.rolling(252, min_periods=1).max().iloc[-1])
    dist_52w_high = _safe_float((close.iloc[-1] - high_52w) / high_52w) if high_52w else 0.0
    vol_ma5 = vol.rolling(5).mean().iloc[-1]
    vol_ratio_5d = _safe_float(vol.iloc[-1] / vol_ma5, default=1.0) if vol_ma5 > 0 else 1.0
    return {
        "momentum_20d": momentum_20d,
        "dist_52w_high": dist_52w_high,
        "vol_ratio_5d": vol_ratio_5d,
    }


def _load_threshold() -> float:
    meta_path = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "sniper_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return float(json.load(f).get("threshold", SNIPER_CONF_THRESHOLD))
    return SNIPER_CONF_THRESHOLD


def _load_forecast_horizon() -> int:
    meta_path = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "sniper_metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return int(json.load(f).get("forecast_horizon_days", SNIPER_FORECAST_HORIZON_DAYS))
    return SNIPER_FORECAST_HORIZON_DAYS


class SniperPredictor:
    def __init__(self):
        self._model = None
        self._model_path = resolve_sniper_model_path()

    @property
    def is_available(self) -> bool:
        return self._model_path is not None

    def _load(self):
        if self._model is None:
            self._model = load_catboost_model(self._model_path)
            self._model_path = resolve_sniper_model_path()

    def predict(self, ticker: str, df: pd.DataFrame, company_name: str = "") -> dict:
        if df is None or len(df) < 30:
            raise ValueError(
                f"Need at least 30 days of price history for {ticker}; got {len(df) if df is not None else 0}."
            )
        self._load()
        vix, vix_vel = _fetch_vix()
        headlines = fetch_headlines(ticker, company_name)
        sent_score = score_headlines(headlines)
        today = date.today()
        save_sentiment(ticker, today, sent_score, len(headlines))

        sent_feats = build_sentiment_features(ticker, today, vix)
        sent_feats["rolling_sentiment_20d"] = sent_score
        ohlcv_feats = _build_ohlcv_features(df)

        row = {
            "momentum_20d": ohlcv_feats["momentum_20d"],
            "dist_52w_high": ohlcv_feats["dist_52w_high"],
            "rolling_sentiment_20d": sent_feats["rolling_sentiment_20d"],
            "vol_ratio_5d": ohlcv_feats["vol_ratio_5d"],
            "VIX": vix,
            "sent_lag_1": sent_feats.get("sent_lag_1", 0.0),
            "sent_lag_3": sent_feats.get("sent_lag_3", 0.0),
            "sent_lag_5": sent_feats.get("sent_lag_5", 0.0),
            "sent_vix_interaction": sent_score * vix,
            "vix_velocity": vix_vel,
        }
        X = pd.DataFrame([row])[FEATURE_COLS]
        proba_up = float(self._model.predict_proba(X)[0, 1])
        threshold = _load_threshold()
        horizon = _load_forecast_horizon()

        if proba_up >= threshold:
            signal = "BULLISH"
        elif proba_up <= (1 - threshold):
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "signal": signal,
            "probability_up": round(proba_up, 4),
            "confidence": round(max(proba_up, 1 - proba_up), 4),
            "above_threshold": proba_up >= threshold,
            "threshold": threshold,
            "model_name": "CatBoost Sniper v5",
            "model_path": self._model_path,
            "forecast_horizon_days": horizon,
            "sentiment_score": round(sent_score, 4),
            "sentiment_backend": _backend(),
            "vix": round(vix, 2),
            "vix_velocity": round(vix_vel, 4),
            "headline_count": len(headlines),
            "available": True,
            "feature_vector": row,
            "reason": (
                f"CatBoost Sniper v5: {proba_up:.1%} P(up) in ~{horizon} trading days. "
                f"Sentiment {_backend()} {sent_score:+.3f}, VIX {vix:.1f}."
            ),
        }
