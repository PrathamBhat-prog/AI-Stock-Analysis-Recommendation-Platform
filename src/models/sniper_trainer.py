"""
Train Sniper v5 CatBoost in-repo (no Colab, no paid APIs).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.config.ml_config import (
    MLFLOW_EXPERIMENT_SNIPER,
    SNIPER_CATBOOST_PARAMS,
    SNIPER_CONF_THRESHOLD,
    SNIPER_MODEL_PATH,
)
from src.data.sniper_dataset import SNIPER_FEATURE_COLS, build_sniper_dataset
from src.data.dataset import chronological_split

logger = logging.getLogger(__name__)


def _metrics(y_true: np.ndarray, probas: np.ndarray, threshold: float) -> dict:
    preds = (probas >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probas)),
        "threshold": threshold,
    }


def train_sniper(
    tickers: list[str] | None = None,
    period: str = "10y",
    threshold: float = SNIPER_CONF_THRESHOLD,
) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    mlruns = project_root / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_SNIPER)

    logger.info("Building Sniper v5 dataset ...")
    dataset = build_sniper_dataset(tickers=tickers, period=period)
    train_df, val_df, test_df = chronological_split(dataset)

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
    val_m = _metrics(y_val.values, val_proba, threshold)
    test_m = _metrics(y_test.values, test_proba, threshold)

    os.makedirs(os.path.dirname(SNIPER_MODEL_PATH), exist_ok=True)
    with open(SNIPER_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    importance = pd.DataFrame({
        "feature": SNIPER_FEATURE_COLS,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    imp_path = Path(SNIPER_MODEL_PATH).parent / "sniper_feature_importance.csv"
    importance.to_csv(imp_path, index=False)

    metadata = {
        "model_name": "CatBoost Sniper v5",
        "feature_columns": SNIPER_FEATURE_COLS,
        "forecast_horizon_days": 20,
        "threshold": threshold,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "tickers": tickers or "default",
        "note": (
            "Historical sentiment filled with neutral prior during training; "
            "live GDELT used at inference."
        ),
    }
    meta_path = Path(SNIPER_MODEL_PATH).parent / "sniper_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with mlflow.start_run(run_name="sniper-v5"):
        mlflow.log_params(SNIPER_CATBOOST_PARAMS)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_artifact(SNIPER_MODEL_PATH)

    logger.info("Sniper v5 saved → %s (test AUC=%.4f)", SNIPER_MODEL_PATH, test_m["roc_auc"])
    return metadata
