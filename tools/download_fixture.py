"""Download a small BTCUSDT 1h CSV for integration tests.

From repo root::

    python tools/download_fixture.py

Writes ``tests/fixtures/btcusdt_1h_2024_jan.csv`` (~14 days). Safe to commit.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUTPUT = REPO_ROOT / "tests" / "fixtures" / "btcusdt_1h_2024_jan.csv"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 14


def fetch_klines(symbol, interval, start, end):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol, "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df[["time", "open", "high", "low", "close", "volume"]].astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float,
    })


def main():
    end = datetime(2024, 2, 1)
    start = end - timedelta(days=DAYS)
    df = fetch_klines(SYMBOL, INTERVAL, start, end)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT}")
    print(f"Time range: {df['time'].min()} -> {df['time'].max()}")
    print(f"Close range: ${df['close'].min():.0f} -> ${df['close'].max():.0f}")


if __name__ == "__main__":
    main()
