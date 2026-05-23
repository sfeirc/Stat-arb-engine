"""
Binance WebSocket (real-time) + REST historical data client.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import websocket  # websocket-client

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Kline:
    symbol: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


@dataclass
class BookTicker:
    symbol: str
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------

REST_BASE = "https://api.binance.com"


def get_exchange_info(symbols: List[str]) -> Dict:
    """Return exchange-info for a list of symbols."""
    url = f"{REST_BASE}/api/v3/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    info = resp.json()
    sym_set = set(symbols)
    return {
        s["symbol"]: s
        for s in info["symbols"]
        if s["symbol"] in sym_set
    }


def get_book_ticker(symbol: str) -> BookTicker:
    """Fetch current best bid/ask from REST."""
    url = f"{REST_BASE}/api/v3/ticker/bookTicker"
    resp = requests.get(url, params={"symbol": symbol}, timeout=5)
    resp.raise_for_status()
    d = resp.json()
    return BookTicker(
        symbol=d["symbol"],
        bid_price=float(d["bidPrice"]),
        ask_price=float(d["askPrice"]),
        bid_qty=float(d["bidQty"]),
        ask_qty=float(d["askQty"]),
    )


def get_recent_trades(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Fetch recent trades from REST."""
    url = f"{REST_BASE}/api/v3/trades"
    resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows)[["time", "price", "qty", "isBuyerMaker"]]
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["price"] = df["price"].astype(float)
    df["qty"] = df["qty"].astype(float)
    df.set_index("time", inplace=True)
    return df


def get_avg_daily_volume(symbol: str, lookback_days: int = 30) -> float:
    """Compute 30-day average daily volume in USDT."""
    url = f"{REST_BASE}/api/v3/klines"
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - lookback_days * 86_400_000
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": lookback_days,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return 0.0
    quote_volumes = [float(row[7]) for row in data]  # index 7 = quote asset volume
    return float(np.mean(quote_volumes))


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

WS_BASE = "wss://stream.binance.com:9443/stream"


def _parse_kline(msg: dict) -> Optional[Kline]:
    """Parse a raw kline WebSocket message into a Kline dataclass."""
    if msg.get("e") != "kline":
        return None
    k = msg["k"]
    return Kline(
        symbol=k["s"],
        open_time=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        is_closed=k["x"],
    )


class BinanceWebSocketClient:
    """
    Subscribe to multiple kline streams simultaneously.
    Closed klines are forwarded to *on_kline* callback or the internal queue.
    """

    def __init__(
        self,
        symbols: List[str],
        interval: str = "1m",
        on_kline: Optional[Callable[[Kline], None]] = None,
    ) -> None:
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.on_kline = on_kline
        self._q: queue.Queue[Kline] = queue.Queue(maxsize=10_000)
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # --- Public API --------------------------------------------------------

    def start(self) -> None:
        """Open the WebSocket in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ws is not None:
            self._ws.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get(self, block: bool = True, timeout: float = 1.0) -> Optional[Kline]:
        """Retrieve the next closed kline from the queue."""
        try:
            return self._q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    # --- Internals ---------------------------------------------------------

    def _stream_url(self) -> str:
        streams = "/".join(
            f"{sym}@kline_{self.interval}" for sym in self.symbols
        )
        return f"{WS_BASE}?streams={streams}"

    def _run(self) -> None:
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self._stream_url(),
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:  # noqa: BLE001
                print(f"[BinanceWS] connection error: {exc}; reconnecting in 3s")
                time.sleep(3)

    def _on_message(self, _ws: websocket.WebSocketApp, raw: str) -> None:
        msg = json.loads(raw)
        data = msg.get("data", msg)  # combined stream wraps in {"data": ...}
        kline = _parse_kline(data)
        if kline is None or not kline.is_closed:
            return
        if self.on_kline is not None:
            self.on_kline(kline)
        else:
            try:
                self._q.put_nowait(kline)
            except queue.Full:
                pass  # drop oldest if backlogged

    def _on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        print(f"[BinanceWS] error: {error}")

    def _on_close(self, _ws: websocket.WebSocketApp, code: int, msg: str) -> None:
        print(f"[BinanceWS] closed (code={code}): {msg}")


# ---------------------------------------------------------------------------
# Convenience: live OHLCV buffer
# ---------------------------------------------------------------------------

class LiveOHLCVBuffer:
    """
    Maintains a rolling buffer of the last *maxlen* closed klines per symbol.
    Thread-safe for single-writer / multiple-reader use.
    """

    def __init__(self, symbols: List[str], maxlen: int = 1440) -> None:
        self.maxlen = maxlen
        self._lock = threading.Lock()
        self._data: Dict[str, List[Kline]] = {s: [] for s in symbols}

    def push(self, kline: Kline) -> None:
        with self._lock:
            buf = self._data.setdefault(kline.symbol, [])
            buf.append(kline)
            if len(buf) > self.maxlen:
                buf.pop(0)

    def to_dataframe(self, symbol: str) -> pd.DataFrame:
        """Return a pandas DataFrame for the symbol's current buffer."""
        with self._lock:
            rows = list(self._data.get(symbol, []))
        if not rows:
            return pd.DataFrame()
        records = [
            {
                "open_time": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in rows
        ]
        df = pd.DataFrame(records).set_index("open_time")
        return df


if __name__ == "__main__":
    # Quick smoke test: print 5 closed 1-min klines for BTC and ETH
    symbols = ["BTCUSDT", "ETHUSDT"]
    client = BinanceWebSocketClient(symbols, interval="1m")
    client.start()
    received = 0
    while received < 5:
        k = client.get(timeout=90)
        if k is not None:
            print(f"[{k.symbol}] {k.open_time}  close={k.close:.2f}  vol={k.volume:.2f}")
            received += 1
    client.stop()
