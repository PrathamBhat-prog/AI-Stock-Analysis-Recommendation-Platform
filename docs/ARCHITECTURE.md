# Architecture

## Dual-path sentiment design

| Phase | Source | Rationale |
|-------|--------|-----------|
| **Training** | Price/volume sentiment proxy + optional yfinance headlines | Stable, reproducible features across 10 years without per-day news API dependency |
| **Inference** | Live GDELT (+ yfinance fallback) scored with VADER or FinBERT | Real-time news for user-facing analysis |

Proxy: `tanh(5d_return × 8) × (1 + 0.15 × volume_zscore)` — backward-looking only.

Implementation: `src/data/sentiment_proxy.py` (train), `src/data/news_fetcher.py` (inference).

## ML model

- **CatBoost Sniper v5** — P(up in ~10 trading days)
- 10 features: momentum, 52w distance, sentiment family, volume, VIX, interactions
- Per-ticker chronological split; early stopping; winsorization

## Agents & serving

```
OHLCV + VIX → Features → CatBoost / Trend / Risk → Decision (horizon blend) → Sizing → API + Gradio
```

Horizon keys are **trading days**, using the standard ~252-session year (so 126d ≈ six months of sessions, 252d ≈ one year of sessions). Long horizons are trend-dominated by design.

## Overfitting controls

1. Chronological split per ticker (no leakage across time or listings)
2. Validation-based early stopping (CatBoost `use_best_model=True`)
3. Feature winsorization before fit
4. Threshold tuned on validation; test set untouched until final metrics
