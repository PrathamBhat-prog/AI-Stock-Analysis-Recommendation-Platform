# AI Stock Analysis & Recommendation Platform

Production-style stock analyser: **CatBoost Sniper v5** + **GDELT sentiment** + **VIX macro** + **trend analysis**, with honest multi-horizon messaging.

> **Disclaimer:** Research/education only. Not financial advice.

[![CI](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml)

---

## Highlights

| Feature | Status |
|---------|--------|
| 20-day CatBoost production model (`.cbm`) | ✅ |
| Historical GDELT sentiment backfill for training | ✅ |
| Optional FinBERT sentiment (free, local) | ✅ |
| Per-ticker chronological train/val/test splits | ✅ |
| Walk-forward backtest with transaction costs | ✅ |
| Inverse-vol position sizing + risk overlay | ✅ |
| Honest horizon disclaimers (API + UI) | ✅ |
| FastAPI + Gradio + Docker Compose | ✅ |
| CI tests (pytest + verify_pipeline) | ✅ |
| No paid APIs required | ✅ |

---

## What can this actually predict?

Horizon keys are **trading days** (market sessions), not calendar days.  
126 trading days ≈ 6 months of market activity (~180 calendar days).  
252 trading days ≈ 1 year of market activity (~365 calendar days).

| Horizon key | Trading days | ~Calendar equivalent | ML weight | Trend weight | Honest answer |
|-------------|--------------|----------------------|-----------|--------------|---------------|
| `5d` | 5 | ~1 week | 80% | 20% | Short-term directional bias |
| `21d` | 21 | ~1 month | 50% | 50% | **Default** — closest to ML training (20 trading days) |
| `63d` | 63 | ~3 months | 30% | 70% | Trend regime |
| `126d` | 126 | ~6 months | 15% | 85% | Trend extrapolation (not 126 calendar days) |
| `252d` | 252 | ~1 year | 10% | 90% | **Trend direction — not an annual price forecast** |

The CatBoost model is trained on **20 trading-day** forward returns.

---

## Quick start

```powershell
git clone https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform.git
cd AI-Stock-Analysis-Recommendation-Platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_model.py

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

### Docker (API + UI)

```powershell
docker compose up --build
# API → :8000   UI → :7860   MLflow → :5000
```

---

## Training

| Command | Description |
|---------|-------------|
| `python train.py --strategy sniper --period 5y` | Train Sniper v5 with GDELT backfill |
| `python train.py --strategy sniper --no-gdelt-backfill` | Fast train (neutral sentiment) |
| `python train.py --strategy sklearn --period 5y` | Compare 8 models + LSTM (20d labels) |
| `python train.py --strategy backtest --period 5y` | Walk-forward backtest |
| `python scripts/backfill_sentiment.py AAPL` | Backfill GDELT cache for one ticker |

Output model: `artifacts/models/trading_model_sniper_v5.cbm` (native CatBoost)

---

## API

```bash
curl http://localhost:8000/horizons

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon_key": "21d"}'

curl -X POST http://localhost:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL","MSFT"], "horizon_key": "21d"}'
```

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SNIPER_MODEL_PATH` | `artifacts/models/trading_model_sniper_v5.cbm` | Model file |
| `SENTIMENT_BACKEND` | `auto` | `vader` \| `finbert` \| `auto` |
| `ENABLE_INFERENCE_MLFLOW` | `false` | Log each API call to MLflow |
| `CORS_ORIGINS` | `*` | API CORS |
| `API_RATE_LIMIT_PER_MINUTE` | `60` | Per-IP rate limit |

### Optional FinBERT (free, no API key)

```powershell
pip install transformers torch
$env:SENTIMENT_BACKEND = "finbert"
```

---

## Project structure

```
src/
  config/horizons.py       # Horizon weights + honest disclaimers
  data/
    features.py            # Unified feature engineering
    sentiment_backfill.py  # GDELT historical backfill
    sentiment_scorer.py    # VADER + optional FinBERT
    sniper_dataset.py      # Sniper training data
  models/
    model_io.py            # .cbm save/load
    sniper_predictor.py    # Production inference
    sniper_trainer.py      # In-repo training
  agents/                  # ML, trend, risk, decision
  risk/position_sizer.py   # Inverse-vol sizing
  backtest/walk_forward.py
  pipelines/
  ui/gradio_app.py
  main.py
scripts/
  setup_model.py
  backfill_sentiment.py
  start.sh                 # Docker entrypoint
tests/
docs/
  ARCHITECTURE.md
```

---

## Stack (100% free / open source)

| Layer | Tools |
|-------|-------|
| Data | yfinance, GDELT, ^VIX |
| Sentiment | VADER (default), FinBERT (optional) |
| ML | CatBoost, scikit-learn, XGBoost, LightGBM, PyTorch LSTM |
| Serving | FastAPI, Gradio |
| MLOps | MLflow (training; inference logging optional) |
| CI | GitHub Actions |
| Deploy | Docker Compose, AWS ECS (optional) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model not found | `python scripts/setup_model.py` or `python train.py --strategy sniper` |
| GDELT slow on first train | Normal — cached in `.cache/news/` and `artifacts/sentiment.db` |
| Re-train without GDELT | `--no-gdelt-backfill` |
| FinBERT OOM on CPU | Use `SENTIMENT_BACKEND=vader` |
| Enable inference logging | `ENABLE_INFERENCE_MLFLOW=true` |

---

## Docs

- [Architecture](docs/ARCHITECTURE.md)

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)

## License

MIT — see [LICENSE](LICENSE)
