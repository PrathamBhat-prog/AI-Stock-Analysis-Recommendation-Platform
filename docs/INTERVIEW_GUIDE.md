# Interview Guide — "Why did you use X and not Y?"

## Problem framing

**Q: What problem does this solve?**  
A: Help retail investors get a **directional bias** (BUY/SELL/HOLD) with **explicit time horizon context**, not fake price targets. We separate short-horizon ML (~20 days) from long-horizon trend views.

## Model choices

| Choice | Why | Alternative considered |
|--------|-----|------------------------|
| CatBoost (Sniper v5) | Strong tabular performance, minimal preprocessing | XGBoost (also in repo for comparison) |
| 20-day label | Signal vs noise; 1-day is mostly random | 5-day (legacy sklearn pipeline) |
| VADER sentiment | Free, fast, no GPU/API | FinBERT (better but heavier) |
| GDELT news | Free, global coverage | Paid Bloomberg/Reuters feeds |
| LSTM in train.py | Captures sequences for research | Too data-hungry for production alone |
| Trend agent (rules) | Works on any ticker without retraining | ARIMA/Prophet (extra deps, fragile) |

## Horizons / product

**Q: Can users predict the market 1 year ahead?**  
A: They get a **trend-regime estimate** (90% trend, 10% ML). We show an `honest_disclaimer` in API/UI. The ML model was **not** trained for 252-day forecasts — claiming that would be misleading.

**Q: Why blend ML and trend?**  
A: ML excels at short horizons with rich features; long horizons need slower-moving structure (MA slopes, MACD). Blending avoids retraining five models while being transparent about weights.

## Evaluation

**Q: Why ROC-AUC over accuracy?**  
A: Direction labels are often imbalanced; AUC measures ranking quality across thresholds.

**Q: How do you avoid overfitting?**  
A: Chronological splits, walk-forward backtest, transaction costs, held-out tickers/time.

## Risk

**Q: Inverse-vol sizing?**  
A: Equal dollar bets ≠ equal risk. Scale down when 20d realized vol is high. Implemented in `src/risk/position_sizer.py`.

## Engineering

**Q: FastAPI + Gradio?**  
A: FastAPI for programmatic clients/integrations; Gradio for fast investor-facing UI.

**Q: No paid APIs?**  
A: yfinance + GDELT + VADER + local CatBoost — reproducible for reviewers and CI.

## Known limitations (shows maturity)

1. Historical sentiment in Sniper training uses neutral prior (live GDELT at inference).
2. yfinance data quality varies by exchange.
3. Not investment advice; educational/research project.
