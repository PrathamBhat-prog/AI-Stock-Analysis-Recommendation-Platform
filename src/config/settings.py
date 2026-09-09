"""Application settings (environment variables). No paid API keys required."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    STOCK_DATA_PROVIDER: str = os.getenv("STOCK_DATA_PROVIDER", "yfinance")
    SNIPER_MODEL_PATH: str = os.getenv(
        "SNIPER_MODEL_PATH",
        os.path.join("artifacts", "models", "trading_model_sniper_v5.cbm"),
    )
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    API_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "60"))
    ENABLE_INFERENCE_MLFLOW: bool = os.getenv("ENABLE_INFERENCE_MLFLOW", "false").lower() in {
        "1", "true", "yes", "on",
    }
    SENTIMENT_BACKEND: str = os.getenv("SENTIMENT_BACKEND", "auto")  # vader | finbert | auto


settings = Settings()
