"""
Generate final deliverables (NOT markdown):
  - artifacts/deliverables/AI_Stock_Analyser_Project_Guide.pdf
  - artifacts/deliverables/Interview_Cheat_Sheet.docx

Run automatically after production training, or manually:
  python scripts/generate_deliverables.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "deliverables"
META_PATH = ROOT / "artifacts" / "models" / "sniper_metadata.json"


def _load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _run_verification() -> tuple[bool, str]:
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "verify_pipeline.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out.strip()
    except Exception as exc:
        return False, str(exc)


def _snippet(path: str, start: int, end: int) -> str:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    chunk = lines[start - 1 : end]
    return "\n".join(chunk)


def _build_project_pdf(meta: dict, verify_ok: bool, verify_log: str) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "AI_Stock_Analyser_Project_Guide.pdf"
    test = meta.get("test_metrics", {})
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
    code = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=10, backColor=colors.HexColor("#f4f4f4"))

    story = []
    story.append(Paragraph("AI Stock Analysis &amp; Recommendation Platform", h1))
    story.append(Paragraph(
        f"Author: Pratham Bhat | Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        body,
    ))
    story.append(Spacer(1, 0.3 * cm))

    sections = [
        ("1. Executive Summary", """
Production-style stock analyser using CatBoost Sniper v5, GDELT news sentiment, VIX macro features,
and a trend analysis agent. Serves BUY/SELL/HOLD via FastAPI and Gradio with honest multi-horizon
messaging (trading days, not calendar days). 100% free stack: yfinance, GDELT, VADER/FinBERT, CatBoost.
        """),
        ("2. Problem &amp; Goals", """
Build an end-to-end ML pipeline that predicts short-term stock direction (~20 trading days) while
being honest about longer horizons (trend-dominated). No paid APIs. Reproducible training, caching,
CI, and Docker deployment for portfolio/interview use.
        """),
        ("3. Architecture", """
Data: yfinance OHLCV + ^VIX + GDELT headlines.
Features: unified features.py (12 sklearn features + chart columns); Sniper uses 10 dedicated features.
Agents: ML (CatBoost), Trend (6 signal groups), Risk (20d vol), Decision (horizon-weighted blend).
Output: position sizing (inverse vol) + plain-English explanation.
Serving: FastAPI :8000, Gradio :7860, optional MLflow :5000.
        """),
        ("4. ML Model — CatBoost Sniper v5", f"""
Target: P(price higher in ~20 trading days).
Features: momentum_20d, dist_52w_high, rolling_sentiment_20d, vol_ratio_5d, VIX, sent_lag_1/3/5,
sent_vix_interaction, vix_velocity.
Training: 32 tickers, per-ticker chronological 70/15/15 split, GDELT historical backfill (SQLite cache).
Threshold: {meta.get('threshold', 0.52)} on ML probability; decision layer uses composite ≥0.58 for BUY.
        """),
        ("5. Latest Held-Out Test Metrics", f"""
ROC-AUC: {test.get('roc_auc', 'N/A')}
Accuracy: {test.get('accuracy', 'N/A')}
Precision @ threshold: {test.get('precision', 'N/A')}
Recall @ threshold: {test.get('recall', 'N/A')}
Train rows: {meta.get('train_rows', 'N/A')} | GDELT backfill: {meta.get('sentiment_backfill', 'N/A')}
Verification smoke test: {'PASSED' if verify_ok else 'FAILED'}
        """),
        ("6. Multi-Horizon Honesty", """
Horizon keys are TRADING DAYS. 126d ≈ 6 months of market sessions (~180 calendar days).
252d ≈ 1 year of sessions (~365 calendar days). ML weight drops as horizon lengthens; 252d is ~90% trend.
The model is NOT trained for annual price forecasts.
        """),
        ("7. Caching &amp; Production Pipeline", """
GDELT backfill: artifacts/sentiment.db + .cache/news/hist_*.json (resumable).
Command: python scripts/run_production_pipeline.py --period 10y
Re-runs skip cached dates/tickers.
        """),
        ("8. Limitations", """
• Directional classifier, not price target or return magnitude.
• GDELT rate limits → sampled backfill (every ~20 sessions) + forward-fill.
• Indian tickers may have sparser GDELT coverage.
• Past metrics ≠ future performance. Research/education only.
        """),
    ]

    for title, text in sections:
        story.append(Paragraph(title, h2))
        story.append(Paragraph(text.strip().replace("\n", " "), body))

    story.append(Paragraph("9. Key Code Snippets", h2))

    snippets = [
        ("Horizon weights (src/config/horizons.py)", _snippet("src/config/horizons.py", 27, 45)),
        ("Decision blend (src/agents/decision_agent.py)", _snippet("src/agents/decision_agent.py", 55, 77)),
        ("Inverse-vol sizing (src/risk/position_sizer.py)", _snippet("src/risk/position_sizer.py", 47, 67)),
        ("GDELT backfill (src/data/sentiment_backfill.py)", _snippet("src/data/sentiment_backfill.py", 49, 64)),
    ]
    for title, code_text in snippets:
        story.append(Paragraph(title, body))
        story.append(Preformatted(code_text, code))
        story.append(Spacer(1, 0.2 * cm))

    if verify_log:
        story.append(Paragraph("10. Verification Log", h2))
        story.append(Preformatted(verify_log[:2000], code))

    doc.build(story)
    return pdf_path


def _build_interview_docx(meta: dict) -> Path:
    from docx import Document
    from docx.shared import Pt

    OUT.mkdir(parents=True, exist_ok=True)
    docx_path = OUT / "Interview_Cheat_Sheet.docx"
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    def h(text: str, level: int = 1):
        doc.add_heading(text, level=level)

    def p(text: str):
        doc.add_paragraph(text)

    test = meta.get("test_metrics", {})

    h("Interview Cheat Sheet — AI Stock Analyser", 0)
    p(f"Metrics snapshot: Test ROC-AUC {test.get('roc_auc', 'N/A')}, "
      f"GDELT backfill {meta.get('sentiment_backfill', 'N/A')}")

    qa = [
        ("Tell me about this project in 60 seconds.",
         "End-to-end stock analyser: CatBoost predicts 20-day direction using price, volume, VIX, "
         "and GDELT sentiment. Trend and risk agents blend in for longer horizons. FastAPI + Gradio, "
         "Docker, MLflow, CI. Honest about what ML can and cannot predict."),
        ("Why CatBoost over XGBoost/LightGBM for production?",
         "Robust defaults, strong tabular performance, native .cbm export, handles mixed features well. "
         "XGB/LGBM remain in sklearn research pipeline for comparison."),
        ("Why 20-day forecast horizon?",
         "1-day direction is mostly noise (~50% accuracy). 20 trading days captures swing moves "
         "while keeping enough labels per ticker. Aligns sklearn research pipeline with production."),
        ("Why trading days vs calendar days in the UI?",
         "Finance uses session counts: 252 ≈ 1 year of market activity, not 252 calendar days. "
         "We document this explicitly to avoid misleading users."),
        ("Why GDELT instead of paid news APIs?",
         "Free, global, no API key. Trade-off: rate limits and sparse historical depth → "
         "sample every 20 sessions, SQLite cache, disk cache, resumable backfill script."),
        ("Why VADER default and FinBERT optional?",
         "VADER is fast, CPU-friendly, reproducible. FinBERT is better semantically but needs "
         "transformers/torch and more compute. SENTIMENT_BACKEND=auto picks FinBERT when available."),
        ("Why per-ticker chronological splits?",
         "Pooling all tickers and random-splitting leaks future data across symbols and listing dates. "
         "Per-ticker 70/15/15 preserves time order within each stock."),
        ("Why walk-forward backtest instead of single train/test?",
         "Time series violates IID. Walk-forward retrains on expanding windows — industry standard. "
         "Includes 10 bps transaction costs per trade."),
        ("What are the biggest limitations?",
         "• Not a price target model — binary direction only.\n"
         "• Long horizons are trend extrapolation, not ML forecasts.\n"
         "• GDELT sentiment is sampled and forward-filled.\n"
         "• Modest AUC (~0.52–0.55) — realistic for retail-grade features, not hedge-fund alpha."),
        ("What pain points did you hit while building?",
         "• Two incompatible ML systems (5d sklearn vs 20d Sniper) — unified to 20d.\n"
         "• README claimed features code didn't implement — full truth audit.\n"
         "• GDELT 429 rate limits — hours-long backfill, caching layer, resumable pipeline.\n"
         "• Fake metrics in Gradio UI — replaced with live sniper_metadata.json.\n"
         "• Trading-day vs calendar-day labeling errors in docs."),
        ("How do you handle HIGH volatility?",
         "Risk agent uses 20d realized vol. Decision agent downgrades BUY → HOLD in HIGH regime. "
         "Position sizer uses inverse-vol scaling (10% base, 20% cap)."),
        ("Why composite score 0.58 for BUY not 0.52?",
         "0.52 is ML probability threshold inside CatBoost signals. 0.58 is the blended ML+trend "
         "composite after horizon weighting — stricter final gate."),
        ("Can this predict a stock for 1 year?",
         "No honestly. At 252d, ~90% weight is trend analysis. We show disclaimers in API and UI. "
         "ML was trained on 20-day moves only."),
        ("How would you improve it next?",
         "Sector-specific models, better sentiment (full FinBERT pipeline), options/implied vol features, "
         "proper purged cross-validation, paper trading loop. Portfolio upload was intentionally skipped."),
        ("Tricky: Why might precision be 0% at 0.52 threshold?",
         "Class imbalance + high threshold → few positive predictions on test set. "
         "ROC-AUC still informative. Threshold tuning trades recall for precision."),
        ("Tricky: Is this financial advice?",
         "No. Research/education disclaimer everywhere. Sizing is illustrative inverse-vol, not execution."),
        ("Tricky: Why not LSTM for production?",
         "LSTM in research pipeline; needs more data, harder to debug, slower inference. "
         "CatBoost on tabular features won for production simplicity and .cbm deployment."),
        ("Tricky: How do you prevent data leakage?",
         "Chronological splits per ticker, backward-only features, GDELT cache for past dates only, "
         "walk-forward backtest, no future returns in feature rows."),
        ("System design: How does caching work?",
         "SQLite sentiment_cache for scores; .cache/news for raw GDELT JSON. "
         "backfill_all_sentiment.py tracks progress in sentiment_backfill_progress.json. "
         "Re-runs are cache-hit only."),
        ("System design: How would you deploy?",
         "Docker Compose (API + Gradio + MLflow). Mount artifacts/ and .cache/. "
         "SNIPER_MODEL_PATH env var. ENABLE_INFERENCE_MLFLOW=false by default for latency."),
    ]

    for q, a in qa:
        h(q, 2)
        p(a)

    h("Quick Reference — Stack", 1)
    p("yfinance, GDELT, VADER, CatBoost, FastAPI, Gradio, MLflow, Docker, GitHub Actions, pytest.")

    doc.save(docx_path)
    return docx_path


def main() -> int:
    try:
        import docx  # noqa: F401
        import reportlab  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "python-docx", "-q"])

    meta = _load_meta()
    verify_ok, verify_log = _run_verification()
    pdf = _build_project_pdf(meta, verify_ok, verify_log)
    docx_file = _build_interview_docx(meta)
    print(f"PDF:  {pdf}")
    print(f"DOCX: {docx_file}")
    print(f"Verification: {'OK' if verify_ok else 'FAILED'}")
    return 0 if verify_ok else 1


if __name__ == "__main__":
    sys.exit(main())
