"""
Train ML models for stock direction prediction.

Usage:
    python train.py --strategy sklearn              # 8 models + LSTM (20d labels)
    python train.py --strategy sniper               # CatBoost Sniper v5 (production)
    python train.py --strategy sniper --no-gdelt-backfill  # fast dev (neutral sentiment)
    python train.py --strategy backtest             # walk-forward backtest
    python train.py --tickers AAPL MSFT --period 5y
"""
import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _train_sklearn(args) -> int:
    from src.pipelines.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline()
    result = pipeline.run(
        tickers=args.tickers,
        period=args.period,
        model_names=args.models,
    )
    meta = result["metadata"]
    ds = meta["dataset_size"]
    print("\n=== Training Complete (sklearn/LSTM, 20d labels) ===")
    print(f"Dataset: {ds['total_rows']} rows from {ds['tickers']} tickers")
    print(f"Best model: {result['best_model']}")
    pm = result["primary_metric"]
    print(f"Test {pm}: {result['best_test_metrics'][pm]:.4f}")
    return 0


def _train_sniper(args) -> int:
    from src.models.sniper_trainer import train_sniper

    mode = "inference_only" if args.no_gdelt_backfill else args.sentiment_mode
    meta = train_sniper(
        tickers=args.tickers,
        period=args.period,
        backfill_sentiment=mode != "inference_only",
        sentiment_mode="full" if mode == "full" else mode,
    )
    print("\n=== Sniper v5 Training Complete ===")
    print(json.dumps(meta, indent=2))
    return 0


def _run_backtest(args) -> int:
    from src.backtest.walk_forward import run_walk_forward_backtest

    result = run_walk_forward_backtest(
        tickers=args.tickers,
        period=args.period,
        backfill_sentiment=not args.no_gdelt_backfill,
    )
    print("\n=== Walk-Forward Backtest ===")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train stock ML models")
    parser.add_argument("--strategy", choices=["sklearn", "sniper", "backtest"], default="sklearn")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--period", default="10y", help="yfinance history window (production default: 10y)")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--no-gdelt-backfill",
        action="store_true",
        help="Alias for --sentiment-mode inference_only (fast production default)",
    )
    parser.add_argument(
        "--sentiment-mode",
        choices=["inference_only", "lite", "full"],
        default="inference_only",
        help="inference_only=fast (~30min, live GDELT at runtime); lite=~8 samples/ticker; full=hours",
    )
    args = parser.parse_args()

    try:
        if args.strategy == "sniper":
            return _train_sniper(args)
        if args.strategy == "backtest":
            return _run_backtest(args)
        return _train_sklearn(args)
    except Exception as exc:
        logger.exception("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
