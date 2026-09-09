# Architecture

## What the user gets (product truth)

| User selects | What we actually predict |
|--------------|--------------------------|
| **1 week** | ~80% ML (20-day-trained) + 20% trend |
| **1 month** | ~50/50 ML + trend (closest to model training horizon) |
| **3–6 months** | Mostly trend; ML is a small nudge |
| **1 year** | ~90% trend extrapolation — **not** a precision annual forecast |

The CatBoost **Sniper v5** model is trained to estimate P(price higher in **~20 trading days**).

## Data flow

```
yfinance (OHLCV) ──┬──► Feature engineering (unified features.py)
GDELT / yfinance   │         │
news (free)        │         ├──► Trend Agent
^VIX               │         ├──► Risk Agent
                   │         └──► Sniper v5 CatBoost (.cbm)
                   │
Historical GDELT ──┴──► SQLite sentiment cache (training backfill)
                              │
                              ├──► Decision Agent (horizon-weighted blend)
                              ├──► Position sizing (inverse vol)
                              └──► FastAPI / Gradio
```

## Resolved design items

| Item | Implementation |
|------|----------------|
| Unified 20-day labels | `FORECAST_HORIZON_DAYS = 20` for sklearn + Sniper |
| Per-ticker train/val/test split | `per_ticker_chronological_split()` |
| Historical GDELT training | `sentiment_backfill.py` + SQLite cache |
| FinBERT upgrade | Optional `SENTIMENT_BACKEND=finbert` (free, local) |
| CatBoost native format | `.cbm` via `model_io.py`; pickle legacy supported |
| Inference MLflow overhead | `ENABLE_INFERENCE_MLFLOW=false` by default |
| Docker API + UI | `docker-compose.yml` runs both services |

## Training commands

```bash
python train.py --strategy sniper --period 5y
python train.py --strategy sniper --no-gdelt-backfill   # fast dev
python train.py --strategy sklearn --period 5y
python train.py --strategy backtest --period 5y
python scripts/backfill_sentiment.py AAPL --period 5y
```

## Why these choices (interview)

- **CatBoost over XGBoost/LightGBM for Sniper:** robust defaults, strong tabular performance.
- **VADER default / FinBERT optional:** free, reproducible; FinBERT when `transformers` installed.
- **GDELT sampled backfill:** respects rate limits; forward-fill between samples.
- **Per-ticker splits:** prevents newer listings from leaking entire history into test.
- **`.cbm` over pickle:** safer versioning, native CatBoost format.

## Remaining optional upgrades

- Sector-specific models
- Real-time alerting
- Portfolio upload mode (intentionally skipped)
