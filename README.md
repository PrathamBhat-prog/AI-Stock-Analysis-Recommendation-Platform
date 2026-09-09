# AI Stock Analysis & Recommendation Platform

Production-style stock analyser combining **CatBoost ML (Sniper v5)**, **GDELT news sentiment**, **VIX macro features**, and **rule-based trend analysis** — with **honest time-horizon messaging** for end users.

> **Disclaimer:** Research and education only. Not financial advice. Past performance does not guarantee future results.

[![CI](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml)

---

## What this project does

| Layer | Role |
|-------|------|
| **Sniper v5 (CatBoost)** | P(price higher in **~20 trading days**) using sentiment + VIX + technicals |
| **Trend Agent** | RSI, MACD, MA alignment, Bollinger, momentum — works on **any ticker** |
| **Risk Agent** | Volatility regime; can downgrade BUY → HOLD in high-vol environments |
| **Decision Agent** | Horizon-weighted BUY / SELL / HOLD with plain-English explanations |
| **Position Sizer** | Inverse-volatility suggested portfolio weight |

### What can it actually predict? (product truth)

| User selects | ML weight | Trend weight | Honest answer |
|--------------|-----------|--------------|---------------|
| **1 week** (`5d`) | 80% | 20% | Short-term directional bias |
| **1 month** (`21d`) | 50% | 50% | **Default** — closest to ML training horizon |
| **3 months** (`63d`) | 30% | 70% | Trend regime, ML assists |
| **6 months** (`126d`) | 15% | 85% | Trend extrapolation |
| **1 year** (`252d`) | 10% | 90% | **Trend direction only — not an annual price forecast** |

Every API/UI response includes `honest_disclaimer` and `user_expectation` fields.

---

## Quick start

### 1. Setup

```powershell
git clone https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform.git
cd AI-Stock-Analysis-Recommendation-Platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_model.py   # copies trading_model_sniper_v5.pkl → artifacts/models/
```

### 2. Verify (no network required)

```powershell
python verify_pipeline.py
pytest tests/ -v
```

### 3. Optional live test

```powershell
python verify_pipeline.py --live --ticker AAPL
```

### 4. Run the app

```powershell
# API — http://localhost:8000/docs
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# UI — http://localhost:7860
python -m src.ui.gradio_app

# Or both via helper script
.\run_app.ps1
```

---

## Training & backtesting

| Command | Description |
|---------|-------------|
| `python train.py --strategy sniper --period 5y` | Train **production** CatBoost Sniper v5 (20-day labels) |
| `python train.py --strategy sklearn --period 5y` | Compare 8 models + LSTM (5-day labels, research) |
| `python train.py --strategy backtest --period 5y` | Walk-forward backtest with transaction costs |

```powershell
# Full production retrain (slow — downloads 10y data for 32 tickers)
python train.py --strategy sniper --period 10y
```

**MLflow UI:** `mlflow ui --backend-store-uri mlruns` → http://localhost:5000

---

## API reference

### `GET /health`
Health check.

### `GET /horizons`
All horizons with product copy (`user_expectation`, `honest_disclaimer`, weights).

### `POST /analyze`
```json
{
  "ticker": "AAPL",
  "period": "2y",
  "horizon_key": "21d"
}
```

**Response highlights:** `final_decision`, `confidence`, `honest_disclaimer`, `position_sizing`, `explainability`, `risk`, `trend`.

### `POST /analyze/batch`
```json
{
  "tickers": ["AAPL", "MSFT", "RELIANCE.NS"],
  "horizon_key": "21d"
}
```

**Horizon keys:** `5d` | `21d` | `63d` | `126d` | `252d`

---

## Project structure

```
src/
  config/
    horizons.py          # Single source of truth for horizon weights + disclaimers
    ml_config.py         # Training hyperparameters, paths, benchmarks
    settings.py          # Env vars (no paid API keys)
  data/
    features.py          # Unified feature engineering (ML + chart + trend columns)
    sniper_dataset.py    # Sniper v5 training data builder
    news_fetcher.py      # GDELT + yfinance news (cached)
    sentiment_cache.py   # SQLite sentiment history
  models/
    sniper_predictor.py  # Production inference
    sniper_trainer.py    # In-repo CatBoost training
    trainer.py           # sklearn/LSTM comparison pipeline
  agents/
    ml_agent.py          # Sniper primary, sklearn fallback
    trend_agent.py       # Rule-based trend analysis
    risk_agent.py        # Volatility overlay
    decision_agent.py    # Horizon-weighted BUY/SELL/HOLD
  risk/
    position_sizer.py    # Inverse-volatility sizing
  backtest/
    walk_forward.py      # Walk-forward backtest with costs
  pipelines/
    inference_pipeline.py
    training_pipeline.py
  ui/gradio_app.py
  main.py                # FastAPI v5
scripts/setup_model.py   # Model path setup
tests/                   # pytest suite
docs/
  ARCHITECTURE.md
  INTERVIEW_GUIDE.md
train.py
verify_pipeline.py
```

---

## Two ML systems (important)

| | **Sniper v5** (production) | **sklearn/LSTM** (research) |
|--|---------------------------|----------------------------|
| Model | CatBoost | XGBoost, LightGBM, LSTM, etc. |
| Label horizon | **20 days** | 5 days |
| Features | Sentiment + VIX + OHLCV | 12 technical indicators |
| Train command | `--strategy sniper` | `--strategy sklearn` |
| Inference | Primary in `ml_agent` | Fallback if Sniper missing |

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SNIPER_MODEL_PATH` | `artifacts/models/trading_model_sniper_v5.pkl` | Production model location |
| `CORS_ORIGINS` | `*` | API CORS (set to your domain in production) |
| `API_RATE_LIMIT_PER_MINUTE` | `60` | Per-IP rate limit |
| `ENVIRONMENT` | `development` | Environment label |

No OpenAI, Gemini, or paid API keys required.

---

## Docker

```powershell
docker build -t stock-analyser .
docker run -p 7860:7860 stock-analyser
```

Requires `trading_model_sniper_v5.pkl` at repo root for the Docker build context (copied into the image).

---

## Known limitations

1. **Historical sentiment in Sniper training** uses a neutral (0.0) prior; live GDELT is used at inference.
2. **yfinance** data quality varies by exchange; some tickers may fail.
3. **1-year horizon** is trend-heavy by design — not an ML annual forecast.
4. **Pooled multi-ticker training** uses chronological row splits (documented trade-off vs per-ticker splits).
5. **Not investment advice** — educational/research project.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Sniper v5 model not found` | Run `python scripts/setup_model.py` or `python train.py --strategy sniper` |
| `No data found for ticker` | Check symbol (e.g. `RELIANCE.NS` for India NSE) |
| GDELT rate limit (429) | Cached automatically; wait and retry |
| Port 7860 busy | Gradio auto-selects another port |
| `train.py` sklearn path slow | PyTorch LSTM is heavy; use `--models catboost xgboost` to subset |
| Tests skip `test_api` | Install full `requirements.txt` (needs `yfinance`) |

---

## Documentation

- [Architecture & design decisions](docs/ARCHITECTURE.md)
- [Interview guide — "why X not Y"](docs/INTERVIEW_GUIDE.md)

---

## Stack (100% free / open source)

| Category | Tools |
|----------|-------|
| Data | yfinance, GDELT API, ^VIX |
| Sentiment | VADER |
| ML | CatBoost, scikit-learn, XGBoost, LightGBM, PyTorch LSTM |
| API / UI | FastAPI, Gradio |
| MLOps | MLflow |
| CI | GitHub Actions |
| Deploy | Docker, AWS ECS (optional) |

---

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)  
Email: prathambhat75@gmail.com

## License

MIT — see [LICENSE](LICENSE)
