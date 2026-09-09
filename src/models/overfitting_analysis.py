"""Evaluate train vs validation vs test metrics to assess overfitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config.ml_config import SNIPER_CBM_PATH, SNIPER_FORECAST_HORIZON_DAYS
from src.data.splits import per_ticker_chronological_split
from src.data.sniper_dataset import SNIPER_FEATURE_COLS, build_sniper_dataset
from src.models.sniper_trainer import _tune_threshold


def _metrics(y_true: np.ndarray, probas: np.ndarray, threshold: float) -> dict:
    preds = (probas >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probas)),
    }


def run_overfitting_analysis(
    sentiment_mode: str = "proxy",
    period: str = "10y",
    model_path: str | None = None,
) -> dict:
    """
    Compare held-out validation vs test performance and report overfitting indicators.

    Methodology:
      1. Rebuild dataset with same settings as training
      2. Per-ticker chronological 70/15/15 split (no random shuffle)
      3. Load saved CatBoost model; score train / val / test
      4. Tune threshold on validation only; apply to all splits
      5. Flag if train >> val or val >> test (AUC gap > 0.03)
    """
    model_path = model_path or SNIPER_CBM_PATH
    model = CatBoostClassifier()
    model.load_model(model_path)

    dataset = build_sniper_dataset(period=period, sentiment_mode=sentiment_mode)
    train_df, val_df, test_df = per_ticker_chronological_split(dataset)

    splits = {
        "train": (train_df, "train"),
        "validation": (val_df, "validation"),
        "test": (test_df, "test"),
    }
    probas = {}
    y_true = {}
    for name, (df, _) in splits.items():
        x = df[SNIPER_FEATURE_COLS]
        y_true[name] = df["target_up"].astype(int).values
        probas[name] = model.predict_proba(x)[:, 1]

    threshold = _tune_threshold(y_true["validation"], probas["validation"])
    metrics = {name: _metrics(y_true[name], probas[name], threshold) for name in splits}

    train_auc = metrics["train"]["roc_auc"]
    val_auc = metrics["validation"]["roc_auc"]
    test_auc = metrics["test"]["roc_auc"]
    gap_train_val = train_auc - val_auc
    gap_val_test = val_auc - test_auc

    if gap_train_val > 0.05:
        verdict = "possible_overfitting"
        summary = (
            f"Train AUC ({train_auc:.4f}) exceeds validation AUC ({val_auc:.4f}) by "
            f"{gap_train_val:.4f} — monitor generalization."
        )
    elif gap_val_test > 0.03:
        verdict = "mild_validation_test_drift"
        summary = (
            f"Validation AUC ({val_auc:.4f}) vs test AUC ({test_auc:.4f}) gap "
            f"{gap_val_test:.4f} — acceptable for financial time series."
        )
    else:
        verdict = "no_severe_overfitting"
        summary = (
            f"Train/val/test AUC ({train_auc:.4f} / {val_auc:.4f} / {test_auc:.4f}) are "
            "close — no large overfitting gap detected."
        )

    best_iter = getattr(model, "best_iteration_", None) or model.get_best_iteration()

    return {
        "threshold": threshold,
        "metrics_by_split": metrics,
        "gaps": {
            "train_minus_val_auc": round(gap_train_val, 4),
            "val_minus_test_auc": round(gap_val_test, 4),
        },
        "verdict": verdict,
        "summary": summary,
        "methodology": [
            "Per-ticker chronological split (70/15/15) — no future leakage",
            "Threshold tuned on validation set only, then applied to train and test",
            "CatBoost early stopping on validation AUC (best_iteration saved)",
            "Feature winsorization at 1st/99th percentile before training",
            f"Label horizon: {SNIPER_FORECAST_HORIZON_DAYS} trading days",
        ],
        "catboost_best_iteration": best_iter,
        "row_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
    }


def save_overfitting_report(path: Path | None = None) -> dict:
    report = run_overfitting_analysis()
    out = path or Path(SNIPER_CBM_PATH).parent / "overfitting_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
