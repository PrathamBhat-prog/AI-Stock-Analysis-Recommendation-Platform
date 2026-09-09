"""
Production pipeline: resumable GDELT backfill → Sniper v5 train (10y).

Caches persist under artifacts/sentiment.db and .cache/news/.
Re-runs skip completed tickers and cached dates.
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
    parser = argparse.ArgumentParser(description="GDELT backfill + Sniper production train")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-backfill", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    logger.info("=== Production pipeline start (period=%s) ===", args.period)

    if not args.skip_backfill:
        import importlib.util

        bf_path = ROOT / "scripts" / "backfill_all_sentiment.py"
        spec = importlib.util.spec_from_file_location("backfill_all_sentiment", bf_path)
        bf_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(bf_mod)
        backfill_all = bf_mod.backfill_all

        logger.info("Phase 1/2: GDELT sentiment backfill (resumable)")
        bf = backfill_all(period=args.period, force=args.force_backfill)
        logger.info("Backfill phase complete: %s", json.dumps(bf, indent=2))
    else:
        logger.info("Phase 1/2: skipped (--skip-backfill)")

    if not args.skip_train:
        from src.models.sniper_trainer import train_sniper

        logger.info("Phase 2/2: Sniper v5 training (uses SQLite + disk cache)")
        meta = train_sniper(period=args.period, backfill_sentiment=True)
        logger.info("Training complete — test AUC=%.4f", meta["test_metrics"]["roc_auc"])
        print("\n=== Production pipeline complete ===")
        print(json.dumps(meta, indent=2))
    else:
        logger.info("Phase 2/2: skipped (--skip-train)")

    logger.info("Phase 3: verification + deliverables (PDF + DOCX)")
    import importlib.util

    del_path = ROOT / "scripts" / "generate_deliverables.py"
    spec = importlib.util.spec_from_file_location("generate_deliverables", del_path)
    del_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(del_mod)
    rc = del_mod.main()
    if rc != 0:
        logger.warning("Deliverables generation completed with verification warnings (see log)")

    logger.info("Log file: %s", LOG_PATH)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Production pipeline failed")
        sys.exit(1)
