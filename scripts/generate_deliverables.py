"""
Generate project PDF + interview DOCX (includes overfitting analysis).

Outputs:
  docs/deliverables/AI_Stock_Analyser_Project_Guide.pdf
  docs/deliverables/Interview_Cheat_Sheet.docx
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "deliverables"
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
        return r.returncode == 0, ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as exc:
        return False, str(exc)


def _overfitting_report() -> dict:
    try:
        from src.models.overfitting_analysis import run_overfitting_analysis
        return run_overfitting_analysis()
    except Exception as exc:
        return {"error": str(exc), "summary": "Overfitting analysis could not be run."}


def _snippet(path: str, start: int, end: int) -> str:
    return "\n".join((ROOT / path).read_text(encoding="utf-8").splitlines()[start - 1 : end])


def _build_project_pdf(meta: dict, of_report: dict, verify_ok: bool) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "AI_Stock_Analyser_Project_Guide.pdf"
    test = meta.get("test_metrics", {})
    val = meta.get("val_metrics", {})
    of_m = of_report.get("metrics_by_split", {})
    train_m = of_m.get("train", {})
    val_m = of_m.get("validation", val)
    test_m = of_m.get("test", test)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=5)
    warn = ParagraphStyle("Warn", parent=body, textColor=colors.HexColor("#B45309"), fontSize=11, spaceAfter=8)
    code = ParagraphStyle("Code", parent=styles["Code"], fontSize=7.5, leading=9, backColor=colors.HexColor("#f5f5f5"))

    story = [
        Paragraph("AI Stock Analysis &amp; Recommendation Platform", h1),
        Paragraph(f"Pratham Bhat | {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", body),
        Paragraph(
            "<b>EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE.</b> Do not use for real trading decisions.",
            warn,
        ),
    ]

    blocks = [
        ("1. Executive Summary", """
End-to-end stock analysis platform: CatBoost Sniper v5 predicts 10-trading-day direction using
price, volume, VIX, and sentiment features. Dual-path sentiment: proxy at training time, live
GDELT at inference. Multi-agent pipeline (ML + trend + risk) with horizon-weighted BUY/SELL/HOLD,
inverse-vol position sizing, FastAPI, Gradio, Docker, MLflow, and CI.
        """),
        ("2. System Architecture", """
Data: yfinance OHLCV, ^VIX, GDELT (inference), VADER/FinBERT.
Training: sentiment proxy (sentiment_proxy.py) + 32 tickers × 10y, per-ticker chronological split.
Model: CatBoost .cbm, 10 features, 10-day binary label.
Serving: inference_pipeline.py → decision_agent.py → position_sizer.py → API/UI.
        """),
        ("3. Held-Out Test Metrics (from sniper_metadata.json)", f"""
Threshold (val-tuned): {meta.get('threshold', 'N/A')}
ROC-AUC: {test.get('roc_auc', 'N/A')} | Accuracy: {test.get('accuracy', 'N/A')}
Precision: {test.get('precision', 'N/A')} | Recall: {test.get('recall', 'N/A')}
Train rows: {meta.get('train_rows', 'N/A')} | Sentiment mode: {meta.get('sentiment_mode', 'N/A')}
        """),
        ("4. Overfitting Analysis — Methodology", """
How we check for overfitting (run after each training via overfitting_analysis.py):

1. Rebuild the dataset with identical settings (proxy sentiment, 10y, 32 tickers).
2. Apply per-ticker chronological 70/15/15 split — test set never seen during training.
3. Load the saved .cbm model; score train, validation, and test partitions.
4. Tune probability threshold on validation F1 only; apply same threshold to all splits.
5. Compare ROC-AUC across splits. Flag if train AUC − val AUC &gt; 0.05 (overfitting risk).

Additional safeguards during training:
• CatBoost early stopping on validation AUC (best iteration retained, not max iterations).
• Feature winsorization at 1st/99th percentile.
• No random train/test shuffle (time-series safe).
        """),
        ("5. Overfitting Analysis — Results", f"""
Verdict: {of_report.get('verdict', 'N/A')}
{of_report.get('summary', '')}

AUC by split:
  Train:      {train_m.get('roc_auc', 'N/A')}
  Validation: {val_m.get('roc_auc', 'N/A')}
  Test:       {test_m.get('roc_auc', 'N/A')}

Gaps: train−val AUC = {of_report.get('gaps', {}).get('train_minus_val_auc', 'N/A')};
      val−test AUC = {of_report.get('gaps', {}).get('val_minus_test_auc', 'N/A')}

CatBoost best iteration: {of_report.get('catboost_best_iteration', 'N/A')} (early stop vs 1200 max).

Interpretation: Fast training (~2–10 min) is expected — CatBoost on ~55k tabular rows with
early stopping is CPU-efficient. Speed alone does not imply overfitting; the val/test AUC gap does.
        """),
        ("6. Limitations", """
• Modest predictive power (AUC ~0.50–0.52) — realistic for public retail features.
• Not a price-target or portfolio optimiser; directional research tool only.
• Long UI horizons are trend-dominated, not ML annual forecasts.
• EDUCATIONAL PURPOSE ONLY — NOT FINANCIAL ADVICE.
        """),
    ]
    for title, text in blocks:
        story.append(Paragraph(title, h2))
        story.append(Paragraph(text.strip().replace("\n", " "), body))

    story.append(Paragraph("7. Key Code Snippets", h2))
    for title, code_text in [
        ("sentiment_proxy.py", _snippet("src/data/sentiment_proxy.py", 1, 25)),
        ("Per-ticker split", _snippet("src/data/splits.py", 1, 35)),
        ("Decision blend", _snippet("src/agents/decision_agent.py", 55, 77)),
        ("Early stopping params", _snippet("src/config/ml_config.py", 72, 85)),
    ]:
        story.append(Paragraph(title, body))
        story.append(Preformatted(code_text, code))
        story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph(f"8. Verification: {'PASSED' if verify_ok else 'FAILED'}", h2))
    doc.build(story)
    return pdf_path


def _build_interview_docx(meta: dict, of_report: dict) -> Path:
    from docx import Document
    from docx.shared import Pt, RGBColor

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Interview_Cheat_Sheet.docx"
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    p = doc.add_paragraph()
    r = p.add_run("EDUCATIONAL ONLY — NOT FINANCIAL ADVICE")
    r.bold = True
    r.font.color.rgb = RGBColor(0xB4, 0x53, 0x09)

    doc.add_heading("Interview Cheat Sheet", 0)
    test = meta.get("test_metrics", {})
    doc.add_paragraph(
        f"Test metrics: AUC {test.get('roc_auc', 'N/A'):.4f}, "
        f"Acc {test.get('accuracy', 'N/A'):.2%}, Prec {test.get('precision', 'N/A'):.2%}"
    )

    qa = [
        ("60-second pitch?", "ML stock analyser: CatBoost predicts 10-day direction from price, VIX, "
         "sentiment. Agents blend ML+trend+risk by horizon. FastAPI+Gradio+Docker+CI. Honest disclaimers."),
        ("Why CatBoost?", "Strong tabular defaults, .cbm export, early stopping, handles mixed features."),
        ("Why 10-day horizon?", "1-day is noise; 10 sessions balances signal vs label count for 32 tickers."),
        ("Dual-path sentiment?", "Train on reproducible proxy; inference uses live GDELT — standard train/serve split."),
        ("How prevent overfitting?", "Per-ticker chronological split, val early stopping, winsorization, "
         "threshold on val only, compare train/val/test AUC in overfitting_report.json."),
        ("Is fast training suspicious?", "No — ~55k rows + CatBoost early stop finishes in minutes. "
         "Check val vs test AUC gap, not wall-clock time."),
        ("Trading days vs calendar?", "252 sessions ≈ 1 year of market activity, not 252 calendar days."),
        ("Limitations?", "AUC ~0.50, not financial advice, long horizons are trend-only."),
        ("Why not LSTM in production?", "CatBoost simpler to deploy (.cbm), faster inference; LSTM in research pipeline."),
        ("Data leakage prevention?", "Chronological per-ticker split, backward-only features, walk-forward backtest."),
    ]
    for q, a in qa:
        doc.add_heading(q, 2)
        doc.add_paragraph(a)

    doc.add_heading("Overfitting check summary", 1)
    doc.add_paragraph(of_report.get("summary", ""))
    doc.save(path)
    return path


def main() -> int:
    try:
        import docx  # noqa: F401
        import reportlab  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "python-docx", "-q"])

    sys.path.insert(0, str(ROOT))
    meta = _load_meta()
    of_report = _overfitting_report()
    report_path = ROOT / "artifacts" / "models" / "overfitting_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(of_report, f, indent=2)

    verify_ok, _ = _run_verification()
    pdf = _build_project_pdf(meta, of_report, verify_ok)
    docx_file = _build_interview_docx(meta, of_report)

    # Mirror to artifacts for local pipeline use
    art = ROOT / "artifacts" / "deliverables"
    art.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, art / pdf.name)
    shutil.copy2(docx_file, art / docx_file.name)

    print(f"PDF:  {pdf}")
    print(f"DOCX: {docx_file}")
    print(f"Overfitting: {of_report.get('verdict')} — {of_report.get('summary', '')[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
