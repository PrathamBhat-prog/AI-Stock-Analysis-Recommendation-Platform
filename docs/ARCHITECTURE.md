# Architecture

## Sentiment: train/serve split (solves GDELT HTTP 429)

GDELT's free API rate-limits historical backfill (HTTP 429). We do **not** call GDELT during training loops.

| Phase | Sentiment source | API calls |
|-------|------------------|-----------|
| **Training** (`sentiment_mode=proxy`) | Price/volume proxy + optional 1× yfinance headlines per ticker | ~32 yfinance calls total |
| **Inference** | Live GDELT → yfinance fallback → VADER/FinBERT | 1–2 per user analysis |

Proxy formula: `tanh(5d_return × 8) × (1 + 0.15 × volume_zscore)` — backward-looking, no leakage.

`gdelt_lite` / `gdelt_full` modes remain in code but are **deprecated** for local use.

## Product truth (horizons)

Horizon keys are **trading days**. Sniper v5 label: **P(up in ~10 trading days)**.

## Data flow

```
yfinance (OHLCV) ──┬──► features + proxy sentiment (training)
^VIX               │         ├──► Trend Agent
                   │         ├──► Risk Agent
                   │         └──► Sniper v5 CatBoost (.cbm)
                   │
Live GDELT ────────┴──► sniper_predictor (inference only)
                              │
                              ├──► Decision Agent
                              ├──► Position sizing
                              └──► FastAPI / Gradio
```

## Training commands

```bash
python scripts/run_production_pipeline.py          # proxy, ~10 min
python train.py --strategy sniper --sentiment-mode proxy
```
