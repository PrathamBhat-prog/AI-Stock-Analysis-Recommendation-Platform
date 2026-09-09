"""
Resumable GDELT sentiment backfill for all training tickers.

Caches to:
  - artifacts/sentiment.db   (SQLite — used by training/inference)
  - .cache/news/hist_*.json    (raw GDELT headline windows)

Safe to re-run: already-cached dates are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.ml_config import DEFAULT_TRAIN_TICKERS, SENTIMENT_BACKFILL_STRIDE
from src.data.fetch_data import fetch_stock_data
from src.data.sentiment_backfill import backfill_ticker_sentiment
from src.data.sentiment_cache import DB_PATH, _get_conn
from src.data.validate_data import validate_stock_data

logger = logging.getLogger(__name__)
PROGRESS_PATH = ROOT / "artifacts" / "sentiment_backfill_progress.json"


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_tickers": [], "runs": []}


def _save_progress(state: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _cache_stats() -> dict:
    news_files = list((ROOT / ".cache" / "news").glob("hist_*.json"))
    rows = 0
    if DB_PATH.exists():
        with _get_conn() as conn:
            rows = conn.execute("SELECT COUNT(*) FROM sentiment_cache").fetchone()[0]
    return {
        "sqlite_rows": rows,
        "gdelt_disk_cache_files": len(news_files),
        "sqlite_path": str(DB_PATH),
        "news_cache_dir": str(ROOT / ".cache" / "news"),
    }


def backfill_all(
    tickers: list[str] | None = None,
    period: str = "10y",
    stride: int = SENTIMENT_BACKFILL_STRIDE,
    pause_between_tickers: float = 10.0,
    force: bool = False,
) -> dict:
    tickers = tickers or DEFAULT_TRAIN_TICKERS
    state = _load_progress()
    completed = set(state.get("completed_tickers", []))
    if force:
        completed = set()

    run_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "stride": stride,
        "tickers_total": len(tickers),
        "per_ticker": {},
    }

    for i, ticker in enumerate(tickers, start=1):
        if ticker in completed and not force:
            logger.info("[%d/%d] %s — already complete (skip)", i, len(tickers), ticker)
            continue

        logger.info("[%d/%d] Backfilling %s (%s) ...", i, len(tickers), ticker, period)
        t0 = time.time()
        try:
            df = fetch_stock_data(ticker, period=period)
            df = validate_stock_data(df)
            dates = df["Date"].dt.date.tolist()
            written = backfill_ticker_sentiment(ticker, dates, stride=stride)
            elapsed = time.time() - t0
            run_meta["per_ticker"][ticker] = {
                "new_samples": written,
                "trading_days": len(dates),
                "elapsed_sec": round(elapsed, 1),
            }
            completed.add(ticker)
            state["completed_tickers"] = sorted(completed)
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            state["cache_stats"] = _cache_stats()
            state["runs"] = (state.get("runs") or [])[-4:] + [run_meta]
            _save_progress(state)
            logger.info(
                "[%d/%d] %s done — %d new writes in %.0fs | cache: %s",
                i, len(tickers), ticker, written, elapsed, _cache_stats(),
            )
        except Exception as exc:
            logger.exception("Backfill failed for %s: %s", ticker, exc)
            run_meta["per_ticker"][ticker] = {"error": str(exc)}
            _save_progress(state)
            raise

        if i < len(tickers):
            time.sleep(pause_between_tickers)

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    state["cache_stats"] = _cache_stats()
    _save_progress(state)
    return {"completed": len(completed), "cache_stats": _cache_stats(), "progress_file": str(PROGRESS_PATH)}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Backfill GDELT sentiment for all training tickers")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--stride", type=int, default=SENTIMENT_BACKFILL_STRIDE)
    parser.add_argument("--pause", type=float, default=10.0, help="Seconds between tickers")
    parser.add_argument("--force", action="store_true", help="Re-run even if ticker marked complete")
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    result = backfill_all(
        tickers=args.tickers,
        period=args.period,
        stride=args.stride,
        pause_between_tickers=args.pause,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
