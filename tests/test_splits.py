import pandas as pd

from src.data.splits import per_ticker_chronological_split


def test_per_ticker_split_preserves_all_tickers():
    rows = []
    for ticker in ("AAA", "BBB"):
        for i, d in enumerate(pd.date_range("2020-01-01", periods=100)):
            rows.append({"ticker": ticker, "Date": d, "x": i})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    train, val, test = per_ticker_chronological_split(df)
    assert set(train["ticker"]) == {"AAA", "BBB"}
    assert set(val["ticker"]) == {"AAA", "BBB"}
    assert set(test["ticker"]) == {"AAA", "BBB"}
    assert len(train) == 140  # 70 per ticker
    assert len(val) == 30
    assert len(test) == 30
