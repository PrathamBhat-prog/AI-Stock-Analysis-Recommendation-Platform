# AI Stock Analysis & Recommendation Platform

Production-style stock analyser: **CatBoost Sniper v5** + **GDELT sentiment** + **VIX macro** + **trend analysis**, with honest multi-horizon messaging.

> **Disclaimer:** Research/education only. Not financial advice.

[![CI](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml)

---

## Highlights

| Feature | Implementation |
|---------|----------------|
| 20-day CatBoost production model (`.cbm`) | `src/models/sniper_trainer.py` → `artifacts/models/trading_model_sniper_v5.cbm` |
| Historical GDELT sentiment backfill (training) | `src/data/sentiment_backfill.py`, SQLite `artifacts/sentiment.db` |
| Optional FinBERT sentiment (local, no API key) | `SENTIMENT_BACKEND=finbert` when `transformers` installed |
| Per-ticker chronological train/val/test splits | `src/data/splits.py` — 70% / 15% / 15% per ticker |
| Walk-forward backtest + transaction costs | `src/backtest/walk_forward.py` — default 10 bps one-way |
| Inverse-vol position sizing + volatility risk overlay | `position_sizer.py` + `risk_agent.py` (BUY→HOLD in HIGH vol) |
| Honest horizon disclaimers (API + UI) | `src/config/horizons.py`, exposed at `GET /horizons` |
| FastAPI + Gradio + Docker Compose | `src/main.py`, `src/ui/gradio_app.py`, `docker-compose.yml` |
| CI | GitHub Actions — `verify_pipeline.py` + `pytest` |
| No paid third-party APIs | yfinance, GDELT, VADER/FinBERT, CatBoost — internet required |

**Metrics honesty:** UI model panel and `artifacts/models/sniper_metadata.json` report **actual** held-out test metrics from the last training run. We do not hard-code marketing numbers in the UI.

---

## What can this actually predict?

Horizon keys are **trading days** (market sessions), not calendar days.

| Horizon key | Trading days | ~Calendar equivalent | ML weight | Trend weight | Honest answer |
|-------------|--------------|----------------------|-----------|--------------|---------------|
| `5d` | 5 | ~1 week | 80% | 20% | Short-term directional bias |
| `21d` | 21 | ~1 month | 50% | 50% | **Default** — closest to ML training (20 trading days) |
| `63d` | 63 | ~3 months | 30% | 70% | Trend regime |
| `126d` | 126 | ~6 months of sessions (~180 calendar days) | 15% | 85% | Trend extrapolation |
| `252d` | 252 | ~1 year of sessions (~365 calendar days) | 10% | 90% | Trend direction — **not** an annual price forecast |

The CatBoost model label: **P(price higher in ~20 trading days)**.

Decision thresholds (blended ML+trend score): BUY ≥ 0.58, SELL ≤ 0.42 (`src/config/horizons.py`).

---

## Quick start

```powershell
git clone https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform.git
cd AI-Stock-Analysis-Recommendation-Platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Option A — copy bundled legacy pickle (inference works; .cbm preferred)
python scripts/setup_model.py

# Option B — production train (recommended)
python train.py --strategy sniper --period 10y

python verify_pipeline.py
pytest tests/ -v
```

### Run locally

```powershell
# API — http://localhost:8000/docs
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# UI — http://localhost:7860
python -m src.ui.gradio_app
```

### Docker (API + UI + MLflow UI)

```powershell
docker compose up --build
# API → :8000   Gradio → :7860   MLflow UI → :5000
```

Mount `./artifacts` and `./.cache` so trained `.cbm` models and GDELT caches persist.

---

## Training

| Command | Description |
|---------|-------------|
| `python scripts/run_production_pipeline.py` | **Full production** — resumable GDELT cache + 10y train |
| `python scripts/backfill_all_sentiment.py` | Pre-warm GDELT/SQLite cache only (resumable) |
| `python train.py --strategy sniper --period 10y` | Train only — 32 tickers, GDELT backfill ON |
| `python train.py --strategy sniper --no-gdelt-backfill` | Fast dev (neutral sentiment features) |
| `python train.py --strategy sklearn --period 10y` | Compare **8** model candidates (7 tabular + LSTM), 20d labels |
| `python train.py --strategy backtest --period 10y` | Walk-forward backtest (10 bps costs) |
| `python scripts/backfill_sentiment.py AAPL --period 10y` | Pre-warm GDELT cache for one ticker |

**Outputs:** `artifacts/models/trading_model_sniper_v5.cbm`, `sniper_metadata.json`, `sniper_feature_importance.csv`

**Universe (default):** 24 US large-caps + 8 Indian `.NS` tickers (`DEFAULT_TRAIN_TICKERS` in `ml_config.py`).

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `GET` | `/horizons` | Horizon keys + weights + disclaimers |
| `POST` | `/analyze` | Single ticker (`ticker`, `horizon_key`, optional `period`) |
| `POST` | `/analyze/batch` | Up to 20 tickers |

```bash
curl http://localhost:8000/horizons

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon_key": "21d"}'
```

Default API `period` is `2y` (minimum enforced for short periods via `MIN_INFERENCE_PERIOD`).

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SNIPER_MODEL_PATH` | `artifacts/models/trading_model_sniper_v5.cbm` | Model file |
| `SENTIMENT_BACKEND` | `auto` | `vader` \| `finbert` \| `auto` |
| `ENABLE_INFERENCE_MLFLOW` | `false` | Log each API call to MLflow |
| `CORS_ORIGINS` | `*` | API CORS |
| `API_RATE_LIMIT_PER_MINUTE` | `60` | Per-IP rate limit |

### Optional FinBERT

```powershell
pip install transformers torch
$env:SENTIMENT_BACKEND = "finbert"
```

---

## Project structure

```
src/
  config/horizons.py, ml_config.py, settings.py
  data/          features, sniper_dataset, sentiment_backfill, news_fetcher
  models/        sniper_trainer, sniper_predictor, model_io
  agents/        ml, trend, risk, decision
  risk/          position_sizer.py
  backtest/      walk_forward.py
  pipelines/     inference_pipeline, training_pipeline
  ui/            gradio_app.py, model_info.py
  main.py
scripts/         setup_model.py, backfill_sentiment.py, start.sh
tests/
docs/ARCHITECTURE.md
```

---

## Stack (free / open source)

| Layer | Tools |
|-------|-------|
| Data | yfinance, GDELT, ^VIX |
| Sentiment | VADER (in `requirements.txt`), FinBERT (optional) |
| ML | CatBoost (production), scikit-learn, XGBoost, LightGBM, PyTorch LSTM (research pipeline) |
| Serving | FastAPI, Gradio |
| MLOps | MLflow (training; inference logging opt-in) |
| CI | GitHub Actions |
| Deploy | Docker Compose (`docker-compose.yml`) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model not found | `python train.py --strategy sniper --period 10y` or `python scripts/setup_model.py` |
| GDELT slow on first train | Expected — cached in `.cache/news/` and `artifacts/sentiment.db` |
| GDELT 429 rate limits | Backfill samples every 20 sessions; re-runs use cache |
| Re-train without GDELT | `--no-gdelt-backfill` |
| FinBERT OOM on CPU | `SENTIMENT_BACKEND=vader` |
| Enable inference MLflow | `ENABLE_INFERENCE_MLFLOW=true` |

---

## Docs

- [Architecture](docs/ARCHITECTURE.md)

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)

## License

MIT — see [LICENSE](LICENSE)
