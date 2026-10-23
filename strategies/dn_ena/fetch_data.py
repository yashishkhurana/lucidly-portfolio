"""Pull the dnENA inputs into data/ (run from a shell with network):

    python -m strategies.dn_ena.fetch_data --from 2025-09-01

Writes three CSVs with a fixed layout; the backtest reads them and nothing else:

  data/pendle_pt_sena.csv      time (UTC, hourly or daily), market (address), maturity (date), pt_price_ena
                               (PT price in sENA/ENA terms, < 1 before maturity), implied_apy, underlying_apy
  data/hl_funding_ENA.csv      time, fundingRate (hourly), premium           
  data/binance_ENAUSDT_1h.csv  time, open, high, low, close, volume          

Pendle: the public API (api-v2.pendle.finance) lists markets per chain and serves a market history
(`/core/v2/{chain}/markets/{address}/history`, 1440 rows a call) with implied APY and PT discount; sENA markets live on Ethereum
(chain 1). The endpoint shapes have changed over time, so this script is written to degrade: it prints the
raw keys it finds and leaves a market out rather than guessing. If an endpoint has moved, the CSV layout
above is the contract — produce it by any route (Pendle's UI export, a subgraph, DefiLlama) and the backtest runs.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from common.data import DATA, binance_klines, hyperliquid_funding

PENDLE = "https://api-v2.pendle.finance/core/v1"
PENDLE_V2 = "https://api-v2.pendle.finance/core/v2"
CHAIN = 1


def _get(url: str, params: dict | None = None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "portfolio-backtests/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def list_sena_markets() -> list[dict]:
    out = []
    page, limit = 0, 100
    while True:
        res = _get(f"{PENDLE}/{CHAIN}/markets", {"skip": page * limit, "limit": limit, "is_active": "false"})
        rows = res.get("results", res if isinstance(res, list) else [])
        for m in rows:
            # every Ethena market carries "ethena" in its protocol field, so match on the underlying symbol
            # (sENA, or plain ENA for the 2024 markets); rsENA and the USDe family are excluded
            ua = m.get("underlyingAsset") or {}
            sym = (ua.get("symbol") if isinstance(ua, dict) else str(ua)) or ""
            if sym in ("sENA", "ENA"):
                out.append(m)
        if len(rows) < limit:
            break
        page += 1
    return out


def market_history(address: str, start: datetime, time_frame: str = "hour") -> pd.DataFrame:
    """Market history from `start` to now. The v1 `/history` route is gone (404, 2026-10); v2 `/history`
    returns at most 1440 rows per call with `timestamp_start`/`timestamp_end`, so page in 59-day windows.
    Rows carry impliedApy, underlyingApy and ptDiscount (PT price in underlying terms = 1 - ptDiscount)."""
    rows: list[dict] = []
    t0, now = start, datetime.now(timezone.utc)
    step = pd.Timedelta(days=59)
    while t0 < now:
        t1 = min(t0 + step, now)
        fmt = lambda d: d.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        res = _get(f"{PENDLE_V2}/{CHAIN}/markets/{address}/history",
                   {"time_frame": time_frame, "timestamp_start": fmt(t0), "timestamp_end": fmt(t1)})
        if "results" in res:
            rows += res["results"]
        elif "timestamps" in res:  # older array shape
            rows += [{"timestamp": t, "impliedApy": a, "ptDiscount": None, "underlyingApy": u}
                     for t, a, u in zip(res["timestamps"], res.get("impliedApys", []), res.get("underlyingApys", []))]
        else:
            print("unrecognised history shape; keys:", list(res)[:12])
            return pd.DataFrame()
        t0 = t1
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["implied_apy"] = pd.to_numeric(df.get("impliedApy"), errors="coerce")
    df["underlying_apy"] = pd.to_numeric(df.get("underlyingApy"), errors="coerce")
    df["pt_price"] = 1.0 - pd.to_numeric(df.get("ptDiscount"), errors="coerce")
    df = df[["time", "implied_apy", "underlying_apy", "pt_price"]].dropna(subset=["time"]).drop_duplicates("time")
    return df.sort_values("time")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2025-06-01")
    a = ap.parse_args()
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    start_ms, end_ms = int(start.timestamp() * 1000), int(time.time() * 1000)

    # 1. Pendle PT-sENA markets, every maturity
    frames = []
    for m in list_sena_markets():
        addr = m.get("address") or m.get("id")
        expiry = m.get("expiry") or m.get("maturity")
        name = m.get("name") or m.get("symbol") or addr
        print("market", name, addr, "expiry", expiry)
        h = market_history(addr, start)
        if h.empty:
            continue
        h["market"] = addr
        h["maturity"] = pd.to_datetime(expiry, utc=True, errors="coerce")
        frames.append(h)
        time.sleep(0.3)
    if frames:
        pt = pd.concat(frames).sort_values(["market", "time"])
        # ptPrice from the API is in underlying (sENA) terms; keep the column name explicit
        pt = pt.rename(columns={"pt_price": "pt_price_ena"})
        pt = pt[pt["time"] >= pd.Timestamp(start)]
        pt.to_csv(DATA / "pendle_pt_sena.csv", index=False)
        print("wrote", len(pt), "rows for", pt["market"].nunique(), "markets ->", DATA / "pendle_pt_sena.csv")
    else:
        print("no Pendle history written — see the docstring for the CSV layout to supply by another route")

    # 2. HL ENA funding, 3. Binance ENA spot
    f = hyperliquid_funding("ENA", start_ms, end_ms)
    f.to_csv(DATA / "hl_funding_ENA.csv", index=False)
    k = binance_klines("ENAUSDT", "1h", start_ms, end_ms)
    k.to_csv(DATA / "binance_ENAUSDT_1h.csv", index=False)
    print("funding rows", len(f), "candles", len(k))
