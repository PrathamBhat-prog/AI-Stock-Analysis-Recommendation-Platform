from src.data.sentiment_scorer import score_headlines_vader, _backend, VADER_AVAILABLE
import pytest


def test_vader_positive_headline():
    if not VADER_AVAILABLE:
        pytest.skip("vaderSentiment not installed")
    score = score_headlines_vader(["Apple stock surges on strong earnings beat"])
    assert score > 0


def test_vader_empty_headlines():
    assert score_headlines_vader([]) == 0.0


def test_backend_defaults_to_vader_without_transformers():
    # In CI without transformers, auto should resolve to vader
    assert _backend() in ("vader", "finbert")
