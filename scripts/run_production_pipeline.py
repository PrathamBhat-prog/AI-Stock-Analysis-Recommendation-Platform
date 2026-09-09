"""
Production pipeline (fast default ~30-45 min):
  1. Train Sniper v5 on 10y price/VIX/momentum (sentiment_mode=inference_only)
  2. Live GDELT sentiment applied at inference time (no multi-hour backfill)
  3. Verify + generate PDF/DOCX deliverables

Optional slow path: --sentiment-mode lite (~1-2h) or full (overnight+)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
LOG_PATH = ROOT / "artifacts" / "production_pipeline.log"


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Fast production train + deliverables")
    parser.add_argument("--period", default="10y")
    parser.add_argument(
        "--sentiment-mode",
        choices=["inference_only", "lite", "full"],
        default="inference_only",
        help="Default inference_only: ~30min train, GDELT live at runtime",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    logger.info("=== Production pipeline (sentiment_mode=%s, period=%s) ===", args.sentiment_mode, args.period)

    if not args.skip_train:
        from src.models.sniper_trainer import train_sniper

        logger.info("Training Sniper v5 (no multi-hour GDELT backfill unless mode=full)")
        meta = train_sniper(
            period=args.period,
            backfill_sentiment=args.sentiment_mode != "inference_only",
            sentiment_mode=args.sentiment_mode,
        )
        logger.info("Training complete — test AUC=%.4f", meta["test_metrics"]["roc_auc"])
        print(json.dumps(meta, indent=2))

    logger.info("Verification + deliverables (PDF + DOCX)")
    import importlib.util

    del_path = ROOT / "scripts" / "generate_deliverables.py"
    spec = importlib.util.spec_from_file_location("generate_deliverables", del_path)
    del_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(del_mod)
    del_mod.main()

    logger.info("Done. Log: %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Production pipeline failed")
        sys.exit(1)
