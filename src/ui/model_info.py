"""Build Gradio model copy from on-disk training artifacts (no fabricated metrics)."""

from __future__ import annotations

import json
from pathlib import Path

from src.config.ml_config import SNIPER_CATBOOST_PARAMS, SNIPER_CONF_THRESHOLD

_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "models"
_META = _ARTIFACTS / "sniper_metadata.json"
_IMPORTANCE = _ARTIFACTS / "sniper_feature_importance.csv"


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _load_metadata() -> dict:
    if not _META.exists():
        return {}
    with open(_META, encoding="utf-8") as f:
        return json.load(f)


def _load_feature_importance() -> list[tuple[str, float]]:
    if not _IMPORTANCE.exists():
        return []
    import pandas as pd

    df = pd.read_csv(_IMPORTANCE)
    if df.empty:
        return []
    total = float(df["importance"].sum()) or 1.0
    rows = []
    for _, row in df.sort_values("importance", ascending=False).iterrows():
        rows.append((str(row["feature"]), 100.0 * float(row["importance"]) / total))
    return rows


def build_model_info_markdown() -> str:
    meta = _load_metadata()
    test = meta.get("test_metrics", {})
    val = meta.get("val_metrics", {})
    threshold = meta.get("threshold", SNIPER_CONF_THRESHOLD)
    backfill = meta.get("sentiment_backfill")
    rows = meta.get("train_rows", "—")
    tickers = meta.get("tickers", "default")
    split = meta.get("split_method", "per_ticker_chronological")

    importance_lines = ""
    for name, pct in _load_feature_importance()[:6]:
        importance_lines += f"| **{name}** | {pct:.1f}% |\n"

    if not importance_lines:
        importance_lines = "| *(train model to populate)* | — |\n"

    params = SNIPER_CATBOOST_PARAMS
    metrics_note = (
        "Metrics below come from the latest `artifacts/models/sniper_metadata.json` "
        "produced by `python train.py --strategy sniper`."
    )
    if not meta:
        metrics_note += " **No metadata file found yet — run training first.**"

    backfill_note = (
        "GDELT historical sentiment backfill: **enabled** during last train."
        if backfill
        else "GDELT historical sentiment backfill: **disabled** on last train (neutral sentiment features)."
        if backfill is False
        else "GDELT backfill status: see training metadata."
    )

    return f"""
## Sniper v5 — CatBoost production model

**Objective:** Estimate P(stock price higher in ~**20 trading days**).

{metrics_note}

### Latest training run (held-out test set)
| Metric | Test | Validation |
|--------|------|------------|
| **ROC-AUC** | {test.get('roc_auc', 'N/A')} | {val.get('roc_auc', 'N/A')} |
| **Accuracy** | {_pct(test.get('accuracy'))} | {_pct(val.get('accuracy'))} |
| **Precision @ {threshold}** | {_pct(test.get('precision'))} | {_pct(val.get('precision'))} |
| **Recall @ {threshold}** | {_pct(test.get('recall'))} | {_pct(val.get('recall'))} |

- Train rows: {rows} | Tickers: {tickers} | Split: {split}
- {backfill_note}
- Probability threshold: **{threshold}** (higher → fewer BUY signals, often higher precision)

### CatBoost hyperparameters (from `ml_config.py`)
```
iterations = {params.get('iterations')}
learning_rate = {params.get('learning_rate')}
depth = {params.get('depth')}
l2_leaf_reg = {params.get('l2_leaf_reg')}
eval_metric = {params.get('eval_metric')}
```

### Feature importance (last trained model)
| Feature | Share of total importance |
|---------|---------------------------|
{importance_lines}

### Horizon blending (trading days, not calendar days)
| Horizon key | ML weight | Trend weight |
|-------------|-----------|--------------|
| 5d (~1 week) | 80% | 20% |
| 21d (~1 month) | 50% | 50% |
| 63d (~3 months) | 30% | 70% |
| 126d (~6 months of sessions) | 15% | 85% |
| 252d (~1 year of sessions) | 10% | 90% |

Long horizons are **trend-dominated**; the ML model is not trained for annual price targets.

### Risk & sizing
- **Risk agent:** 20-day realized volatility → LOW / MEDIUM / HIGH; can downgrade BUY to HOLD in HIGH vol.
- **Position sizing:** inverse-volatility scaling from a 10% base allocation (capped at 20%).

---
**Disclaimer:** Research/education only. Not financial advice. Past metrics do not guarantee future results.
"""


def build_glossary_markdown() -> str:
    threshold = _load_metadata().get("threshold", SNIPER_CONF_THRESHOLD)
    return f"""
## Signals & definitions

### BUY / SELL / HOLD
| Signal | Meaning |
|--------|---------|
| **BUY** | Horizon-weighted composite score ≥ buy threshold |
| **SELL** | Composite score ≤ sell threshold |
| **HOLD** | Between thresholds or risk-downgraded |

Default buy threshold: **{threshold}** on the blended score (not raw ML probability alone).

### Trading days vs calendar days
Horizon keys like `126d` and `252d` mean **trading sessions** (~180 and ~365 calendar days respectively), not calendar-day counts.

### Trend agent (6 signal groups)
1. Moving-average alignment & slopes  
2. RSI momentum  
3. MACD / signal crossovers  
4. Bollinger band position  
5. 20-day price momentum  
6. Volume confirmation  

### Position sizing
`weight ≈ base_allocation × min(target_vol / realized_vol, 3.0) × conviction`, capped at 20% of portfolio.

### Sentiment
- **Inference:** live GDELT (+ yfinance fallback) scored with VADER or optional FinBERT.  
- **Training:** historical GDELT sampled every ~20 sessions, cached in `artifacts/sentiment.db`, forward-filled.
"""
