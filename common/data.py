"""Data loaders. Bundled CSVs under data/ are the default so every backtest reproduces
offline; the fetchers refresh them from public endpoints when a network is available.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ----------------------------------------------------------------------------- bundled

def morpho_algo_history() -> pd.DataFrame:
    """Daily series used by the loop allocator replay: price (collateral exchange rate, 1e18),
    supply/borrow assets (token units), borrow APR, live flag."""
    df = pd.read_csv(DATA / "algo_history.csv", parse_dates=["date"])
    return df.sort_values(["market", "date"]).reset_index(drop=True)


def eurc_usdc_aero_state() -> pd.DataFrame:
    """Tick-level state log of the EURC/USDC Aerodrome CL backtest (ts, tick, equity, range, fees)."""
    df = pd.read_csv(DATA / "eurc_usdc_aero_backtest.csv")
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


# ----------------------------------------------------------------------------- fetchers

def _get(url: str, params: dict | None = None, timeout: int = 30):
    import urllib.parse
    import urllib.request

    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "portfolio-backtests/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Spot klines, paginated. Columns: open_time, open, high, low, close, volume."""
    rows = []
    cur = start_ms
    while cur < end_ms:
        batch = _get(
            "https://api.binance.com/api/v3/klines",
            {"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": 1000},
        )
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "n", "tbb", "tbq", "ignore",
    ])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["time", "open", "high", "low", "close", "volume"]]


def hyperliquid_funding(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Hourly funding history from Hyperliquid's info endpoint."""
    import urllib.request

    rows, cur = [], start_ms
    while cur < end_ms:
        body = json.dumps({"type": "fundingHistory", "coin": coin, "startTime": cur, "endTime": end_ms}).encode()
        req = urllib.request.Request(
            "https://api.hyperliquid.xyz/info", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1]["time"] + 1
        if len(batch) < 500:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["premium"] = df["premium"].astype(float)
    return df[["time", "fundingRate", "premium"]]


def hyperliquid_candles(coin: str, start_ms: int, end_ms: int, interval: str = "1h") -> pd.DataFrame:
    """Hourly candles from Hyperliquid's info endpoint (used for HYPE, which has no Binance spot market).
    Returned in the same shape as binance_klines so the loaders are source-agnostic."""
    import urllib.request

    rows, cur = [], start_ms
    while cur < end_ms:
        body = json.dumps({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": cur, "endTime": end_ms}}).encode()
        req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1]["t"]) + 1
        if nxt <= cur or len(batch) < 2:
            break
        cur = nxt
        time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates("t")
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["t"].astype(int), unit="ms", utc=True)
    out = pd.DataFrame({"time": df["time"], "open": df["o"].astype(float), "high": df["h"].astype(float),
                        "low": df["l"].astype(float), "close": df["c"].astype(float), "volume": df["v"].astype(float)})
    return out.sort_values("time").reset_index(drop=True)


def cached(name: str, builder, refresh: bool = False) -> pd.DataFrame:
    """Use data/<name>.csv if present (or refresh=False), else build and cache it."""
    path = DATA / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["time"])
    df = builder()
    df.to_csv(path, index=False)
    return df
