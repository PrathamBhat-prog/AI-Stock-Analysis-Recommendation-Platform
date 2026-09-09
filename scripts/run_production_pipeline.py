"""
Production pipeline (~10 min):
  1. Train Sniper v5 (proxy sentiment — no GDELT 429)
  2. Live GDELT at inference only
  3. Verify + PDF/DOCX deliverables
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    parser = argparse.ArgumentParser(description="Production train + deliverables")
    parser.add_argument("--period", default="10y")
    parser.add_argument(
        "--sentiment-mode",
        choices=["proxy", "inference_only", "lite", "full"],
        default="proxy",
        help="proxy=default (no GDELT 429); lite/full deprecated",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if args.sentiment_mode in ("lite", "full"):
        logger.warning(
            "GDELT %s mode often hits HTTP 429 — use --sentiment-mode proxy instead",
            args.sentiment_mode,
        )

    logger.info("=== Production pipeline (sentiment_mode=%s) ===", args.sentiment_mode)

    if not args.skip_train:
        from src.models.sniper_trainer import train_sniper

        meta = train_sniper(period=args.period, sentiment_mode=args.sentiment_mode)
        logger.info(
            "Done — test AUC=%.4f acc=%.3f prec=%.3f",
            meta["test_metrics"]["roc_auc"],
            meta["test_metrics"]["accuracy"],
            meta["test_metrics"]["precision"],
        )
        print(json.dumps(meta, indent=2))

    import importlib.util

    del_path = ROOT / "scripts" / "generate_deliverables.py"
    spec = importlib.util.spec_from_file_location("generate_deliverables", del_path)
    del_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(del_mod)
    del_mod.main()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
