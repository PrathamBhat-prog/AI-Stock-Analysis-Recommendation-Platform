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
                   │         └──► Sniper v5 CatBoost (10 features)
                   │
                   └──► Decision Agent (horizon-weighted blend)
                              │
                              ├──► Position sizing (inverse vol)
                              └──► API / Gradio
```

## Why these choices (interview)

- **CatBoost over XGBoost/LightGBM for Sniper:** robust defaults, handles mixed features, strong on tabular finance data without heavy tuning.
- **VADER over FinBERT/GPT:** free, CPU-only, no API cost; upgrade path documented.
- **GDELT:** free global news; cached to respect rate limits.
- **Trend blend for long horizons:** honest UX — don't pretend a 20-day model forecasts 1 year.
- **Chronological split / walk-forward:** prevents lookahead leakage in time series.
- **MLflow:** experiment tracking without paid MLOps platforms.

## Training commands

```bash
python train.py --strategy sniper --period 5y    # production model
python train.py --strategy sklearn --period 5y   # research comparison
python train.py --strategy backtest --period 5y  # walk-forward with costs
```

## Audit notes (code review)

| Item | Status |
|------|--------|
| Unified `features.py` for train + inference + trend | Done |
| Horizon honesty in API/UI | Done |
| Risk + position sizing wired | Done |
| In-repo Sniper training | Done |
| Walk-forward backtest (non-overlapping trades) | Done |
| CI + pytest | Done |
| Gradio default horizon = 1 month (`21d`) | Done |
| Historical GDELT backfill for training | Future work |
| FinBERT sentiment upgrade | Future work |
| Per-ticker chronological splits | Future work |
