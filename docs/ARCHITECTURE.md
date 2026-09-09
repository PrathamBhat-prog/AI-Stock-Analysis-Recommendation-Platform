# Architecture

## Product truth (what the user selects vs what we compute)

Horizon keys are **trading days** (market sessions). `126d` ≈ six months of market activity (~180 calendar days); `252d` ≈ one year of sessions (~365 calendar days).

| User-facing horizon | ML weight | Trend weight | What we actually do |
|---------------------|-----------|--------------|---------------------|
| ~1 week (`5d`) | 80% | 20% | ML (20d-trained) + light trend |
| ~1 month (`21d`) | 50% | 50% | Balanced — closest to training horizon |
| ~3 months (`63d`) | 30% | 70% | Trend-led |
| ~6 months of sessions (`126d`) | 15% | 85% | Trend extrapolation |
| ~1 year of sessions (`252d`) | 10% | 90% | Trend direction — **not** an ML annual forecast |

The CatBoost **Sniper v5** model estimates **P(price higher in ~10 trading days)** (v5.1: was 20d).

## Data flow

```
yfinance (OHLCV) ──┬──► features.py (12 ML features + chart/trend columns)
GDELT / yfinance   │         │
news (live)        │         ├──► Trend Agent (6 signal groups, any ticker)
^VIX               │         ├──► Risk Agent (20d vol → LOW/MED/HIGH)
                   │         └──► Sniper v5 CatBoost (.cbm, 10 features)
                   │
Historical GDELT ──┴──► SQLite sentiment_cache (training backfill, stride=20d)
                              │
                              ├──► Decision Agent (horizon-weighted blend)
                              ├──► Position sizing (inverse vol, 10% base)
                              └──► FastAPI / Gradio
```

## Training pipeline (Sniper v5)

1. Fetch 10y OHLCV for `DEFAULT_TRAIN_TICKERS` (32 symbols)
2. Optional GDELT backfill → `artifacts/sentiment.db`
3. Build 20-day forward-return binary labels
4. `per_ticker_chronological_split()` — 70/15/15 per ticker
5. CatBoost with early stopping on validation AUC
6. Save `.cbm`, `sniper_metadata.json`, feature importance CSV, MLflow run

## Inference pipeline

1. Fetch OHLCV (`period` default `2y`; short periods bumped to `2y` minimum)
2. Live GDELT headlines (+ yfinance fallback) → VADER/FinBERT score
3. Sniper predict → ML probability; trend + risk agents run in parallel
4. Decision agent blends by horizon; risk can downgrade BUY → HOLD
5. Inverse-vol position sizing on final decision

## Key modules

| Item | Location |
|------|----------|
| Horizon weights + disclaimers | `src/config/horizons.py` |
| 20-day label | `FORECAST_HORIZON_DAYS = 20` in `ml_config.py` |
| Per-ticker split | `src/data/splits.py` |
| GDELT backfill | `src/data/sentiment_backfill.py` |
| Walk-forward + 10 bps costs | `src/backtest/walk_forward.py` |
| Metrics in UI (no hard-coded marketing numbers) | `src/ui/model_info.py` |

## Training commands

```bash
python train.py --strategy sniper --period 10y
python train.py --strategy sniper --no-gdelt-backfill   # fast dev
python train.py --strategy sklearn --period 10y
python train.py --strategy backtest --period 10y
python scripts/backfill_sentiment.py AAPL --period 10y
```

## Design rationale

- **CatBoost for Sniper:** strong tabular defaults, native `.cbm` export.
- **VADER default / FinBERT optional:** reproducible, no API keys.
- **GDELT sampled backfill:** rate-limit safe; forward-fill between samples.
- **Per-ticker splits:** prevents listing-date leakage across tickers.
- **Honest horizons:** longer views are trend-dominated by design.

## Optional future work

- Sector-specific models
- Real-time alerting
- Portfolio upload mode (intentionally out of scope)
