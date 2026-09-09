"""CatBoost model save/load — native .cbm preferred, pickle legacy fallback."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

from catboost import CatBoostClassifier

from src.config.ml_config import SNIPER_CBM_PATH, SNIPER_MODEL_PATH

logger = logging.getLogger(__name__)


def sniper_model_paths() -> list[str]:
    return [
        os.environ.get("SNIPER_MODEL_PATH", SNIPER_CBM_PATH),
        SNIPER_CBM_PATH,
        os.environ.get("SNIPER_MODEL_PATH", SNIPER_MODEL_PATH),
        SNIPER_MODEL_PATH,
        "trading_model_sniper_v5.pkl",
        "artifacts/models/trading_model_sniper_v5.cbm",
        "artifacts/models/trading_model_sniper_v5.pkl",
    ]


def resolve_sniper_model_path() -> str | None:
    for path in sniper_model_paths():
        if path and os.path.exists(path):
            return path
    return None


def save_catboost_model(model: CatBoostClassifier, cbm_path: str = SNIPER_CBM_PATH) -> str:
    os.makedirs(os.path.dirname(cbm_path), exist_ok=True)
    model.save_model(cbm_path)
    logger.info("Saved CatBoost native model to %s", cbm_path)
    return cbm_path


def load_catboost_model(path: str | None = None) -> CatBoostClassifier:
    path = path or resolve_sniper_model_path()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "Sniper model not found. Train with: python train.py --strategy sniper"
        )
    if path.endswith(".cbm"):
        model = CatBoostClassifier()
        model.load_model(path)
        logger.info("Loaded CatBoost .cbm from %s", path)
        return model
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded legacy pickle model from %s", path)
    return model
