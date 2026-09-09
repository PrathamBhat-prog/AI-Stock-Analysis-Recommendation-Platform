# AI Stock Analysis & Recommendation Platform

> ## ⚠️ IMPORTANT DISCLAIMER
>
> **This project is for EDUCATIONAL and RESEARCH purposes only.**
>
> It is **NOT financial advice**. It is **NOT** a recommendation to buy or sell any security.
> Past model metrics do not guarantee future results. Do not use this tool as the sole basis
> for investment decisions. Consult a qualified financial professional before investing.

**CatBoost Sniper v5** estimates the probability that a stock’s close is higher after about **10 trading sessions**, using price, volume, VIX, and sentiment features. A decision layer mixes that probability with trend analysis and a volatility-risk overlay, then suggests a position size. Training uses a market-derived sentiment proxy; live analysis scores **GDELT** headlines with VADER or FinBERT. The stack is FastAPI, Gradio, Docker, MLflow, and GitHub Actions.

[![CI](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml)

![Gradio dashboard — AI Stock Analysis Recommendation Platform](docs/images/stock-analyser.png)

---

## Highlights

| Capability | Details |
|------------|---------|
| **Sniper v5** (CatBoost `.cbm`) | Binary classifier: P(close higher in ~10 trading sessions) |
| **Sentiment** | Training: price/volume proxy (+ optional yfinance headlines). Inference: live GDELT + VADER or FinBERT |
| **Decision stack** | ML score + trend signals + realized-vol risk + horizon mix + inverse-vol sizing (10% base, 20% cap) |
| **Horizons** | 5 / 21 / 63 / 126 / 252 **trading sessions** (standard equity calendar; see below) |
| **MLOps** | MLflow on train, optional inference logging, Docker Compose, pytest CI |
| **Evaluation** | Per-ticker chronological splits, winsorization, validation-only threshold, walk-forward backtest script |
| **Data & libraries** | yfinance, GDELT, CatBoost, FastAPI, Gradio — no paid market-data APIs |

**Held-out test** (32 tickers, 10 years, threshold 0.43): accuracy **53.8%**, precision **53.8%**, ROC-AUC **0.50**. After each training run these numbers are stored in `artifacts/models/sniper_metadata.json` and shown on the Gradio model panel.

---

## What the model predicts

Sniper v5 answers: *is the close likely higher in about **10 trading sessions**?*

The API and dashboard also expose **5, 21, 63, 126, and 252 session** views. Those views share the **same** classifier. Each view mixes the 10-session probability with trend analysis; as the requested look-ahead gets longer, trend receives more weight (`src/config/horizons.py`).

### Horizon keys (trading sessions)

Equity work counts **sessions**, not calendar dates. A typical US cash-equity year has about **252** trading days (weekends and holidays excluded). The same 252 is used when annualizing volatility. The other keys follow that calendar:

| Key | Sessions | About | Derivation |
|-----|----------|--------|------------|
| `5d` | 5 | 1 week | Five consecutive sessions |
| `21d` | 21 | 1 month | 252 / 12 |
| `63d` | 63 | 1 quarter | 252 / 4 |
| `126d` | 126 | 6 months | 252 / 2 |
| `252d` | 252 | 1 year | Full trading year |

`126d` is six months of **market days** (~180 calendar days), not 126 calendar days. `252d` is one year of **market days** (~365 calendar days), not 252 calendar days.

| Key | ML weight | Trend weight | Use |
|-----|-----------|--------------|-----|
| `5d` | 80% | 20% | Near-term bias |
| `21d` | 50% | 50% | Default (closest product view to the 10-session label) |
| `63d` | 30% | 70% | Quarter |
| `126d` | 15% | 85% | Half-year of sessions |
| `252d` | 10% | 90% | Full trading year |

**Decision rule:** BUY if the blended score ≥ 0.58, SELL if ≤ 0.42, otherwise HOLD. Elevated realized volatility can change BUY to HOLD.

---

## Architecture

```
Training:  yfinance OHLCV + ^VIX + sentiment proxy  →  CatBoost (.cbm)
Inference: live OHLCV + VIX + GDELT headlines       →  same 10 features → P(up)
Agents:    ML + Trend + Risk  →  horizon mix  →  inverse-vol size  →  FastAPI / Gradio
```

**Sentiment.** Historical news coverage from public APIs is incomplete across a 10-year daily panel. Training therefore uses a backward-looking price/volume proxy (and at most one yfinance headline blend per ticker). At inference, current GDELT articles are scored (yfinance news as fallback). Set `SENTIMENT_BACKEND` to `vader`, `finbert`, or `auto`.

**Features (10):** 20-session momentum, distance from 52-week high, 20-session rolling sentiment, 5-session volume ratio, VIX, sentiment lags 1/3/5, sentiment × VIX, 5-session VIX velocity.

**Universe:** 32 listings (24 US + 8 India `.NS`) in `DEFAULT_TRAIN_TICKERS`.

See [Architecture](docs/ARCHITECTURE.md).

---

## Quick start

```powershell
git clone https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform.git
cd AI-Stock-Analysis-Recommendation-Platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/setup_model.py
python train.py --strategy sniper --period 10y
python verify_pipeline.py
pytest tests/ -v
```

```powershell
uvicorn src.main:app --reload --port 8000    # OpenAPI: :8000/docs
python -m src.ui.gradio_app                  # dashboard: :7860
```

```powershell
docker compose up --build
```

Model files are produced by training (`python train.py --strategy sniper --period 10y`). They are not committed. `scripts/setup_model.py` copies a local `.cbm` if one is already on disk.

---

## Training

| Command | Purpose |
|---------|---------|
| `python scripts/run_production_pipeline.py` | Train and print held-out metrics |
| `python train.py --strategy sniper --period 10y` | Production CatBoost (`sentiment_mode=proxy`) |
| `python train.py --strategy sklearn --period 10y` | Research: 8 candidates (7 tabular + LSTM), 20-session labels |
| `python train.py --strategy backtest --period 10y` | Walk-forward backtest (10 bps costs) |

**Outputs:** `artifacts/models/trading_model_sniper_v5.cbm`, `sniper_metadata.json`.

### Split and regularisation

- Per-ticker **chronological** 70/15/15 split (no shuffle)
- CatBoost **early stopping** on validation AUC (`use_best_model=True`)
- Feature **winsorization** at the 1st and 99th percentiles
- Classification threshold maximises **validation F1**; the test set is scored once at the end

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/horizons` | Horizon definitions and mix weights |
| POST | `/analyze` | Single ticker |
| POST | `/analyze/batch` | Up to 20 tickers |

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SNIPER_MODEL_PATH` | `artifacts/models/trading_model_sniper_v5.cbm` | Model file |
| `SENTIMENT_BACKEND` | `auto` | `vader` \| `finbert` \| `auto` |
| `ENABLE_INFERENCE_MLFLOW` | `false` | Per-request MLflow logging |

---

## Repository map

| Path | Role |
|------|------|
| `src/data/sniper_dataset.py` | Features, labels, dataset build |
| `src/data/sentiment_proxy.py` | Training-time sentiment |
| `src/data/news_fetcher.py` | Live GDELT |
| `src/models/sniper_trainer.py` / `sniper_predictor.py` | Train / load `.cbm` |
| `src/agents/` | ML, trend, risk, decision mix |
| `src/risk/position_sizer.py` | Inverse-vol sizing |
| `src/config/horizons.py` | Session-based horizons |
| `src/main.py` | FastAPI |
| `src/ui/gradio_app.py` | Dashboard |
| `tests/` | CI |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Training and inference paths |

---

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)

## License

MIT — see [LICENSE](LICENSE)
