"""
Data pipeline: download, validate, normalize, store as Parquet.
"""
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import requests
import time
from datetime import datetime, timezone

DATA_DIR = Path("data/parquet")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "MATICUSDT", "ADAUSDT", "DOTUSDT",
]


def download_binance_klines(
    symbol: str,
    interval: str = "1m",
    start_ms: int = None,
    end_ms: int = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Download OHLCV klines from Binance REST API."""
    url = "https://api.binance.com/api/v3/klines"
    params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_vol", "taker_buy_quote_vol", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)
    df.set_index("open_time", inplace=True)
    return df


def download_full_history(symbol: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Download full historical klines in batches, handling pagination."""
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    all_dfs: list = []
    current_ms = start_ms

    while current_ms < end_ms:
        df = download_binance_klines(symbol, "1d", start_ms=current_ms, limit=1000)
        if df.empty:
            break
        all_dfs.append(df)
        last_ts = int(df.index[-1].timestamp() * 1000)
        if last_ts <= current_ms:
            break
        current_ms = last_ts + 1
        time.sleep(0.1)  # rate limit

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs).drop_duplicates().sort_index()
    return combined


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: returns, log returns, rolling vol, VWAP."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol_21"] = df["log_returns"].rolling(21).std() * np.sqrt(252)
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = (
        (df["typical_price"] * df["volume"]).rolling(21).sum()
        / df["volume"].rolling(21).sum()
    )
    return df


def validate_and_clean(df: pd.DataFrame, max_forward_fill: int = 3) -> pd.DataFrame:
    """Validate data: no backfill, max 3-bar forward fill, drop remaining NaN."""
    df = df.copy()
    df = df.ffill(limit=max_forward_fill)
    df = df.dropna()
    if not (df["close"] > 0).all():
        raise ValueError("Negative or zero prices detected")
    if not (df["volume"] >= 0).all():
        raise ValueError("Negative volumes detected")
    return df


def save_parquet(df: pd.DataFrame, symbol: str, data_dir: Path = DATA_DIR) -> None:
    """Persist a DataFrame to a Parquet file using snappy compression."""
    path = data_dir / f"{symbol}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path), compression="snappy")


def load_parquet(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a previously saved Parquet file for *symbol*."""
    path = data_dir / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(str(path)).to_pandas()


def run_pipeline(symbols: list = SYMBOLS, start_date: str = "2020-01-01") -> None:
    """Download, process, and save all symbols."""
    for sym in symbols:
        print(f"Downloading {sym}...")
        df = download_full_history(sym, start_date)
        if df.empty:
            print(f"  WARNING: No data for {sym}")
            continue
        df = compute_features(df)
        df = validate_and_clean(df)
        save_parquet(df, sym)
        print(f"  Saved {len(df)} rows for {sym}")


if __name__ == "__main__":
    run_pipeline()
