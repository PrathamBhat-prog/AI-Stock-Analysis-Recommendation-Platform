# Interview Guide — "Why did you use X and not Y?"

## Problem framing

**Q: What problem does this solve?**  
A: Help retail investors get a **directional bias** (BUY/SELL/HOLD) with **explicit time horizon context**, not fake price targets. We separate short-horizon ML (~20 days) from long-horizon trend views.

## Model choices

| Choice | Why | Alternative considered |
|--------|-----|------------------------|
| CatBoost (Sniper v5) | Strong tabular performance, `.cbm` native format | XGBoost (also in repo) |
| 20-day label (all pipelines) | Signal vs noise; aligned research + production | 5-day (removed — caused confusion) |
| VADER default | Free, fast, no GPU | FinBERT optional locally |
| GDELT backfill (sampled) | Free historical news; cached | Paid Bloomberg feeds |
| Per-ticker chronological split | Prevents cross-ticker leakage | Global row sort (old approach) |
| Trend agent (rules) | Any ticker, no retrain | ARIMA/Prophet |

## Horizons / product

**Q: Can users predict the market 1 year ahead?**  
A: They get a **trend-regime estimate** (90% trend, 10% ML). We show `honest_disclaimer` in every response.

## Evaluation

**Q: How do you avoid overfitting?**  
A: Per-ticker chronological splits, walk-forward backtest with transaction costs, non-overlapping trades.

## Engineering

**Q: Why `.cbm` not pickle?**  
A: Native CatBoost format — version-safe, no arbitrary code execution risk from pickle.

**Q: Why is MLflow off at inference by default?**  
A: Latency — logging every API call adds I/O. Enable with `ENABLE_INFERENCE_MLFLOW=true` for demos.

**Q: FinBERT without paid API?**  
A: `transformers` downloads ProsusAI/finbert once; runs locally on CPU/GPU.

## Known limitations (honest)

1. GDELT backfill is **sampled** (every ~20 trading days) — not every single day.
2. yfinance data quality varies by exchange.
3. Not investment advice.
