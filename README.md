# AI Stock Analysis & Recommendation Platform

**Sniper v5** — CatBoost + GDELT sentiment + VIX macro features for **~20-day** directional signals, blended with trend analysis for longer horizons.

> **Disclaimer:** Research/education only. Not financial advice.

## What can this actually predict?

| Horizon | Primary signal | Honest answer |
|---------|----------------|---------------|
| 1 week | ML (80%) + trend | Short-term bullish/bearish bias |
| 1 month | ML + trend (50/50) | Closest to model training (~20 days) |
| 3–6 months | Trend dominant | Regime/trend context, not ML-precision |
| 1 year | ~90% trend | **Directional trend estimate — not an annual price forecast** |

See `GET /horizons` or the **Horizon Guide** tab in the UI for full product copy.

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_model.py

# Smoke tests (no network)
python verify_pipeline.py
pytest tests/ -v

# Train production model (free data: yfinance + VIX)
python train.py --strategy sniper --period 5y

# Walk-forward backtest with transaction costs
python train.py --strategy backtest --period 5y
```

### Run the app

**API (port 8000):**
```powershell
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**UI (port 7860):**
```powershell
python -m src.ui.gradio_app
```

Or: `.\run_app.ps1`

## API

```bash
# Default: 1-month horizon
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon_key": "21d"}'

# 1-year view (trend-heavy — read honest_disclaimer in response)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon_key": "252d"}'

curl http://localhost:8000/horizons
```

## Training strategies

| Command | Purpose |
|---------|---------|
| `python train.py --strategy sniper` | CatBoost Sniper v5 (production) |
| `python train.py --strategy sklearn` | Compare 8 models + LSTM (research) |
| `python train.py --strategy backtest` | Walk-forward backtest |

## Architecture

```
yfinance + GDELT + VIX → Features → CatBoost Sniper v5
                                  → Trend Agent
                                  → Risk Agent → Decision → Position sizing
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)  
Interview prep: [docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md)

## Stack (all free / open source)

- **Data:** yfinance, GDELT API, VIX
- **ML:** CatBoost, scikit-learn, PyTorch LSTM
- **Sentiment:** VADER (no paid NLP APIs)
- **Serving:** FastAPI + Gradio
- **MLOps:** MLflow
- **Deploy:** Docker + AWS ECS (optional)

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)

## License

MIT — see [LICENSE](LICENSE)
