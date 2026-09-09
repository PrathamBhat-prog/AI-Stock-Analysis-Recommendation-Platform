"""
Train Sniper v5 CatBoost in-repo (no Colab, no paid APIs).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config.ml_config import (
    MLFLOW_EXPERIMENT_SNIPER,
    SNIPER_CATBOOST_PARAMS,
    SNIPER_CBM_PATH,
    SNIPER_CONF_THRESHOLD,
    SNIPER_FORECAST_HORIZON_DAYS,
    SNIPER_MODEL_PATH,
)
from src.data.splits import per_ticker_chronological_split
from src.data.sniper_dataset import SNIPER_FEATURE_COLS, build_sniper_dataset
from src.models.model_io import save_catboost_model

logger = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, probas: np.ndarray, threshold: float) -> dict:
    preds = (probas >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probas)),
        "threshold": threshold,
    }


def _tune_threshold(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Pick threshold on validation set that maximises F1."""
    best_t = SNIPER_CONF_THRESHOLD
    best_f1 = -1.0
    for t in np.linspace(0.42, 0.58, 33):
        f1 = f1_score(y_true, (probas >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    logger.info("Tuned threshold=%.3f (val F1=%.4f)", best_t, best_f1)
    return best_t


def train_sniper(
    tickers: list[str] | None = None,
    period: str = "10y",
    threshold: float | None = None,
    backfill_sentiment: bool = True,
    sentiment_mode: str = "proxy",
) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    mlruns = project_root / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_SNIPER)

    logger.info(
        "Building Sniper v5 dataset (sentiment_mode=%s, horizon=%dd) ...",
        sentiment_mode, SNIPER_FORECAST_HORIZON_DAYS,
    )
    dataset = build_sniper_dataset(
        tickers=tickers,
        period=period,
        backfill_sentiment=backfill_sentiment,
        sentiment_mode=sentiment_mode,
    )
    train_df, val_df, test_df = per_ticker_chronological_split(dataset)

    x_train = train_df[SNIPER_FEATURE_COLS]
    y_train = train_df["target_up"].astype(int)
    x_val = val_df[SNIPER_FEATURE_COLS]
    y_val = val_df["target_up"].astype(int)
    x_test = test_df[SNIPER_FEATURE_COLS]
    y_test = test_df["target_up"].astype(int)

    model = CatBoostClassifier(**SNIPER_CATBOOST_PARAMS)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

    val_proba = model.predict_proba(x_val)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]
    tuned_t = threshold if threshold is not None else _tune_threshold(y_val.values, val_proba)
    val_m = _metrics(y_val.values, val_proba, tuned_t)
    test_m = _metrics(y_test.values, test_proba, tuned_t)

    cbm_path = save_catboost_model(model, SNIPER_CBM_PATH)

    importance = pd.DataFrame({
        "feature": SNIPER_FEATURE_COLS,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    imp_path = Path(SNIPER_CBM_PATH).parent / "sniper_feature_importance.csv"
    importance.to_csv(imp_path, index=False)

    metadata = {
        "model_name": "CatBoost Sniper v5",
        "model_path": cbm_path,
        "legacy_pickle_path": SNIPER_MODEL_PATH,
        "feature_columns": SNIPER_FEATURE_COLS,
        "forecast_horizon_days": SNIPER_FORECAST_HORIZON_DAYS,
        "threshold": tuned_t,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "tickers": tickers or "default",
        "split_method": "per_ticker_chronological",
        "sentiment_mode": sentiment_mode,
        "sentiment_backfill": sentiment_mode in ("lite", "full", "gdelt_lite", "gdelt_full"),
        "note": (
            f"{sentiment_mode}: {SNIPER_FORECAST_HORIZON_DAYS}d labels, winsorized features, "
            "val-tuned threshold. Training uses proxy sentiment; live GDELT at inference."
        ),
    }
    meta_path = Path(SNIPER_CBM_PATH).parent / "sniper_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with mlflow.start_run(run_name="sniper-v5"):
        mlflow.log_params(SNIPER_CATBOOST_PARAMS)
        mlflow.log_param("split_method", "per_ticker_chronological")
        mlflow.log_param("sentiment_mode", sentiment_mode)
        mlflow.log_param("forecast_horizon_days", SNIPER_FORECAST_HORIZON_DAYS)
        mlflow.log_param("threshold", tuned_t)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(cbm_path)

    logger.info(
        "Sniper v5 saved — test AUC=%.4f acc=%.3f prec=%.3f @ threshold=%.3f",
        test_m["roc_auc"], test_m["accuracy"], test_m["precision"], tuned_t,
    )
    return metadata
