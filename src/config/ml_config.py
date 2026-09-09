"""
ML training and inference configuration.

Horizon alignment:
  - SNIPER_FORECAST_HORIZON_DAYS = 20 (production CatBoost model)
  - FORECAST_HORIZON_DAYS = 5 (legacy sklearn/LSTM pipeline for comparison)
"""

# Expanded universe for training rows
DEFAULT_TRAIN_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "V", "MA", "JNJ", "UNH", "WMT", "HD", "PG",
    "XOM", "KO", "PEP", "DIS", "NFLX", "AMD", "INTC", "CSCO",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
]

FORECAST_HORIZON_DAYS = 5
SNIPER_FORECAST_HORIZON_DAYS = 20
TRAIN_PERIOD = "10y"
TARGET_MIN_ROWS = 25_000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
PRIMARY_METRIC = "roc_auc"

MODEL_CANDIDATES = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "lstm",
]

SEQUENCE_LENGTH = 60
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.35
LSTM_EPOCHS = 300
LSTM_BATCH_SIZE = 512
LSTM_LEARNING_RATE = 3e-4
LSTM_PATIENCE = 30

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = "artifacts/models"
BEST_MODEL_PATH = "artifacts/models/best_model.joblib"
BEST_LSTM_PATH = "artifacts/models/best_model.pt"
LSTM_SCALER_PATH = "artifacts/models/lstm_scaler.joblib"
MODEL_METADATA_PATH = "artifacts/models/model_metadata.json"
FEATURE_IMPORTANCE_PATH = "artifacts/models/feature_importance.csv"
BENCHMARK_REPORT_PATH = "artifacts/models/benchmark_comparison.json"
SNIPER_MODEL_PATH = "artifacts/models/trading_model_sniper_v5.pkl"

MLFLOW_EXPERIMENT_TRAINING = "stock-ml-training"
MLFLOW_EXPERIMENT_INFERENCE = "stock-analysis-pipeline"
MLFLOW_EXPERIMENT_SNIPER = "sniper-v5-training"

MIN_INFERENCE_PERIOD = "2y"
SHORT_PERIODS = {"1d", "5d", "1mo", "3mo"}

# CatBoost Sniper v5 defaults (reproducible training)
SNIPER_CATBOOST_PARAMS = {
    "iterations": 1500,
    "learning_rate": 0.015,
    "depth": 7,
    "l2_leaf_reg": 8,
    "random_seed": 42,
    "verbose": 100,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}

SNIPER_CONF_THRESHOLD = 0.52

MARKET_BENCHMARKS = {
    "random_guess": {"accuracy": 0.50, "f1": 0.50, "roc_auc": 0.50},
    "buy_and_hold_majority": {"accuracy": 0.52, "f1": 0.55, "roc_auc": 0.52},
    "arima_directional_proxy": {"accuracy": 0.51, "f1": 0.52, "roc_auc": 0.53},
    "xgboost_gbm_industry": {"accuracy": 0.55, "f1": 0.56, "roc_auc": 0.57},
    "lstm_daily_direction_lit": {"accuracy": 0.54, "f1": 0.58, "roc_auc": 0.56},
    "transformer_finance_lit": {"accuracy": 0.56, "f1": 0.60, "roc_auc": 0.58},
}
