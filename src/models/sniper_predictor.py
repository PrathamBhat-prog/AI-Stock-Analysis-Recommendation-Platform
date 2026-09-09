# -*- coding: utf-8 -*-
"""Sniper v5 Production Predictor — CatBoostClassifier + live sentiment/VIX."""

from __future__ import annotations

import logging
import os
import pickle
from datetime import date

import pandas as pd
import yfinance as yf

from src.config.ml_config import SNIPER_CONF_THRESHOLD
from src.config.settings import settings
from src.data.news_fetcher import fetch_headlines
from src.data.sentiment_cache import build_sentiment_features, save_sentiment
from src.data.sentiment_scorer import score_headlines_vader

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "momentum_20d", "dist_52w_high", "rolling_sentiment_20d",
    "vol_ratio_5d", "VIX", "sent_lag_1", "sent_lag_3",
    "sent_lag_5", "sent_vix_interaction", "vix_velocity",
]


def _resolve_model_path() -> str:
    candidates = [
        settings.SNIPER_MODEL_PATH,
        os.path.join("artifacts", "models", "trading_model_sniper_v5.pkl"),
        "trading_model_sniper_v5.pkl",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return settings.SNIPER_MODEL_PATH


def _fetch_vix() -> tuple[float, float]:
    try:
        vix_df = yf.Ticker("^VIX").history(period="1mo", auto_adjust=True)
        vix = float(vix_df["Close"].iloc[-1])
        vel = float(vix_df["Close"].pct_change(5).iloc[-1])
        return vix, vel
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return 20.0, 0.0


def _build_ohlcv_features(df: pd.DataFrame) -> dict:
    close = df["Close"]
    vol = df["Volume"]
    momentum_20d = float(close.pct_change(20).iloc[-1])
    high_52w = float(close.rolling(252, min_periods=1).max().iloc[-1])
    dist_52w_high = float((close.iloc[-1] - high_52w) / high_52w) if high_52w else 0.0
    vol_ratio_5d = float(
        (vol.iloc[-1] / vol.rolling(5).mean().iloc[-1])
        if vol.rolling(5).mean().iloc[-1] > 0 else 1.0
    )
    return {
        "momentum_20d": momentum_20d,
        "dist_52w_high": dist_52w_high,
        "vol_ratio_5d": vol_ratio_5d,
    }


class SniperPredictor:
    def __init__(self):
        self._model = None
        self._model_path = _resolve_model_path()

    @property
    def is_available(self) -> bool:
        return os.path.exists(self._model_path)

    def _load(self):
        if self._model is None:
            if not self.is_available:
                raise FileNotFoundError(
                    f"Sniper v5 model not found. Tried: {_resolve_model_path()}. "
                    "Run: python train.py --strategy sniper"
                )
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("Sniper v5 loaded from %s", self._model_path)

    def predict(self, ticker: str, df: pd.DataFrame, company_name: str = "") -> dict:
        self._load()
        vix, vix_vel = _fetch_vix()
        headlines = fetch_headlines(ticker, company_name)
        sent_score = score_headlines_vader(headlines)
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
        threshold = SNIPER_CONF_THRESHOLD

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
            "forecast_horizon_days": 20,
            "sentiment_score": round(sent_score, 4),
            "vix": round(vix, 2),
            "vix_velocity": round(vix_vel, 4),
            "headline_count": len(headlines),
            "available": True,
            "feature_vector": row,
            "reason": (
                f"CatBoost Sniper v5: {proba_up:.1%} P(up) in ~20 trading days. "
                f"Sentiment {sent_score:+.3f}, VIX {vix:.1f}."
            ),
        }
