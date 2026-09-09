import importlib.util

import pytest

if importlib.util.find_spec("yfinance") is None:
    pytest.skip("yfinance not installed", allow_module_level=True)

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_horizons_endpoint():
    r = client.get("/horizons")
    assert r.status_code == 200
    data = r.json()
    assert "252d" in data["horizons"]
    assert "honest_disclaimer" in data["horizons"]["252d"]
