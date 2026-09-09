import os
import tempfile

import pytest

pytest.importorskip("catboost")
from catboost import CatBoostClassifier

from src.models.model_io import load_catboost_model, resolve_sniper_model_path, save_catboost_model


def test_save_and_load_cbm_roundtrip():
    model = CatBoostClassifier(iterations=5, verbose=0, random_seed=1)
    X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y = [0, 1, 0, 1]
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.cbm")
        save_catboost_model(model, path)
        loaded = load_catboost_model(path)
        assert loaded.predict_proba(X).shape == (4, 2)


def test_resolve_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr(
        "src.models.model_io.sniper_model_paths",
        lambda: ["/nonexistent/model.cbm"],
    )
    assert resolve_sniper_model_path() is None
