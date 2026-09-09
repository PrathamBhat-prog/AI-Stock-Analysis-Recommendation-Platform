# -*- coding: utf-8 -*-
"""
Headline sentiment: VADER by default, optional FinBERT (HuggingFace, free).

SENTIMENT_BACKEND=auto  → FinBERT if transformers is installed, else VADER
SENTIMENT_BACKEND=vader
SENTIMENT_BACKEND=finbert  → FinBERT with VADER fallback on error
"""

from __future__ import annotations

import logging

import numpy as np

from src.config.settings import settings

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _vader = None
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")

_finbert_pipeline = None
_finbert_failed = False
FINBERT_MODEL = "ProsusAI/finbert"


def finbert_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _load_finbert():
    global _finbert_pipeline, _finbert_failed
    if _finbert_pipeline is not None or _finbert_failed:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            truncation=True,
            max_length=64,
        )
        logger.info("FinBERT loaded (%s)", FINBERT_MODEL)
    except Exception as exc:
        _finbert_failed = True
        logger.warning("FinBERT unavailable, using VADER: %s", exc)
        _finbert_pipeline = None
    return _finbert_pipeline


def score_headlines_vader(headlines: list[str]) -> float:
    """Mean VADER compound score in [-1, +1]."""
    if not VADER_AVAILABLE or not headlines:
        return 0.0
    scores = [_vader.polarity_scores(h)["compound"] for h in headlines if h]
    return float(np.mean(scores)) if scores else 0.0


def score_headlines_finbert(headlines: list[str]) -> float:
    """
    Map FinBERT labels to a signed score: P(positive) - P(negative) in [-1, 1].
    Neutral headlines contribute 0.
    """
    pipe = _load_finbert()
    if pipe is None or not headlines:
        return score_headlines_vader(headlines)
    signed: list[float] = []
    for h in headlines:
        if not h:
            continue
        try:
            out = pipe(h[:512])[0]
            label = str(out.get("label", "")).lower()
            score = float(out.get("score", 0.0))
            if "pos" in label:
                signed.append(score)
            elif "neg" in label:
                signed.append(-score)
            else:
                signed.append(0.0)
        except Exception:
            signed.append(score_headlines_vader([h]))
    return float(np.mean(signed)) if signed else 0.0


def _backend() -> str:
    choice = (settings.SENTIMENT_BACKEND or "auto").lower()
    if choice == "finbert":
        return "finbert"
    if choice == "vader":
        return "vader"
    return "finbert" if finbert_available() else "vader"


def score_headlines(headlines: list[str]) -> float:
    """Primary scorer used by inference and GDELT backfill."""
    backend = _backend()
    if backend == "finbert":
        return score_headlines_finbert(headlines)
    return score_headlines_vader(headlines)


def compute_sentiment(headlines: list[str]) -> dict:
    score = score_headlines(headlines)
    logger.info("Sentiment score: %+.4f (%d headlines, backend=%s)", score, len(headlines), _backend())
    return {
        "rolling_sentiment_20d": score,
        "sent_positive": max(0.0, score),
        "sent_negative": min(0.0, score),
        "headline_count": len(headlines),
        "backend": _backend(),
    }
