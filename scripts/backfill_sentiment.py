"""CLI: backfill GDELT sentiment for one ticker into SQLite cache."""

import argparse
import sys

from src.data.fetch_data import fetch_stock_data
from src.data.sentiment_backfill import backfill_ticker_sentiment
from src.data.validate_data import validate_stock_data


def main() -> int:
    parser = argparse.ArgumentParser(description="GDELT sentiment backfill")
    parser.add_argument("ticker", help="e.g. AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()

    df = fetch_stock_data(args.ticker, period=args.period)
    df = validate_stock_data(df)
    dates = df["Date"].dt.date.tolist()
    n = backfill_ticker_sentiment(args.ticker, dates, stride=args.stride)
    print(f"Backfill complete: {n} new GDELT samples for {args.ticker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
