# AI Stock Analysis & Recommendation Platform

> ## ⚠️ IMPORTANT DISCLAIMER
>
> **This project is for EDUCATIONAL and RESEARCH purposes only.**
>
> It is **NOT financial advice**. It is **NOT** a recommendation to buy or sell any security.
> Past model metrics do not guarantee future results. Do not use this tool as the sole basis
> for investment decisions. Consult a qualified financial professional before investing.

Production-style stock analyser: **CatBoost Sniper v5** + **dual-path sentiment** (training proxy + live GDELT) + **VIX macro** + **trend analysis**, with honest multi-horizon messaging.

[![CI](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform/actions/workflows/ci.yml)

---

## Highlights

| Capability | Details |
|------------|---------|
| **Sniper v5** (CatBoost `.cbm`) | 10-trading-day direction classifier; native CatBoost export |
| **Dual-path sentiment** | Training: market-derived proxy + yfinance headlines. Inference: live GDELT + VADER/FinBERT |
| **Multi-agent pipeline** | ML + trend (6 signals) + risk (vol overlay) + horizon-weighted decision |
| **Position sizing** | Inverse-volatility scaling with 10% base allocation (capped 20%) |
| **Honest horizons** | Trading-day keys with API/UI disclaimers (`GET /horizons`) |
| **MLOps** | MLflow training logs, optional inference logging, Docker Compose |
| **Quality** | Per-ticker chronological splits, walk-forward backtest, CI (pytest) |
| **Stack** | 100% free/open source — yfinance, GDELT, CatBoost, FastAPI, Gradio |

**Latest held-out test metrics** are stored in `artifacts/models/sniper_metadata.json` after each training run (reported honestly in the Gradio model panel).

---

## What does the model predict?

Horizon keys are **trading days** (market sessions), not calendar days.

| Key | Sessions | ~Calendar | ML wt | Trend wt | Role |
|-----|----------|-----------|-------|----------|------|
| `5d` | 5 | ~1 week | 80% | 20% | Short-term bias |
| `21d` | 21 | ~1 month | 50% | 50% | **Default** — nearest to ~10d ML training |
| `63d` | 63 | ~3 months | 30% | 70% | Trend-led |
| `126d` | 126 | ~6 mo sessions | 15% | 85% | Trend extrapolation |
| `252d` | 252 | ~1 yr sessions | 10% | 90% | Trend only — not an annual forecast |

**ML label:** P(price higher in ~**10 trading days**).  
**Decision layer:** BUY if blended score ≥ 0.58, SELL if ≤ 0.42 (`src/config/horizons.py`).

---

## Architecture (summary)

```
Training:  yfinance OHLCV + VIX + sentiment proxy → CatBoost (.cbm)
Inference: live GDELT headlines + same price/VIX features → prediction
Agents:    ML + Trend + Risk → Decision → Position sizing → API / Gradio
```

See [Architecture](docs/ARCHITECTURE.md) and [Project Guide PDF](docs/deliverables/AI_Stock_Analyser_Project_Guide.pdf).

---

## Quick start

```powershell
git clone https://github.com/PrathamBhat-prog/AI-Stock-Analysis-Recommendation-Platform.git
cd AI-Stock-Analysis-Recommendation-Platform
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/setup_model.py          # or train below
python train.py --strategy sniper --period 10y
python verify_pipeline.py
pytest tests/ -v
```

```powershell
uvicorn src.main:app --reload --port 8000    # API docs :8000/docs
python -m src.ui.gradio_app                  # UI :7860
```

```powershell
docker compose up --build
```

---

## Training

| Command | Purpose |
|---------|---------|
| `python scripts/run_production_pipeline.py` | Train + verify + regenerate docs (~10 min) |
| `python train.py --strategy sniper --period 10y` | Production CatBoost (default `sentiment_mode=proxy`) |
| `python train.py --strategy sklearn --period 10y` | Research: 8 model candidates (7 tabular + LSTM) |
| `python train.py --strategy backtest --period 10y` | Walk-forward backtest (10 bps costs) |

**Outputs:** `artifacts/models/trading_model_sniper_v5.cbm`, `sniper_metadata.json`, `overfitting_report.json`

**Universe:** 32 tickers (24 US + 8 India `.NS`) — `DEFAULT_TRAIN_TICKERS` in `ml_config.py`.

### Anti-overfitting measures

- Per-ticker **chronological** 70/15/15 split (no random shuffle)
- CatBoost **early stopping** on validation AUC
- **Winsorization** (1st–99th percentile) on features
- Threshold tuned on **validation F1** only, evaluated on held-out test
- Run `python scripts/generate_deliverables.py` to refresh overfitting analysis in the project PDF

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/horizons` | Horizon weights + disclaimers |
| POST | `/analyze` | Single ticker analysis |
| POST | `/analyze/batch` | Up to 20 tickers |

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SNIPER_MODEL_PATH` | `artifacts/models/trading_model_sniper_v5.cbm` | Model file |
| `SENTIMENT_BACKEND` | `auto` | `vader` \| `finbert` \| `auto` |
| `ENABLE_INFERENCE_MLFLOW` | `false` | Per-request MLflow logging |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design |
| [Project Guide (PDF)](docs/deliverables/AI_Stock_Analyser_Project_Guide.pdf) | Full technical write-up + code snippets + overfitting analysis |
| [Interview Cheat Sheet (DOCX)](docs/deliverables/Interview_Cheat_Sheet.docx) | Q&A for technical interviews |

---

## Author

**Pratham Bhat** — [PrathamBhat-prog](https://github.com/PrathamBhat-prog)

## License

MIT — see [LICENSE](LICENSE)
