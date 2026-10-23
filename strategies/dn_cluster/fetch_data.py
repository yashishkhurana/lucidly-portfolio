"""Pull 90 days of Hyperliquid hourly funding and Binance 1h spot for the cluster assets into data/.
Run on a machine with network access:  python -m strategies.dn_cluster.fetch_data
"""
from __future__ import annotations

import time

from common.data import DATA, binance_klines, hyperliquid_candles, hyperliquid_funding

COINS = {"SOL": "SOLUSDT", "BTC": "BTCUSDT", "HYPE": "HYPEUSDT", "ETH": "ETHUSDT"}

if __name__ == "__main__":
    end = int(time.time() * 1000)
    start = end - 90 * 24 * 3600 * 1000
    for coin, sym in COINS.items():
        f = hyperliquid_funding(coin, start, end)
        f.to_csv(DATA / f"hl_funding_{coin}.csv", index=False)
        src = "binance"
        try:
            if coin == "HYPE":
                raise ValueError("no Binance spot market")
            k = binance_klines(sym, "1h", start, end)
        except Exception as e:
            print(coin, "Binance spot unavailable (", e, ") — using Hyperliquid candles")
            k = hyperliquid_candles(coin, start, end)
            src = "hyperliquid"
        if k.empty:
            print(coin, "no candles fetched; the backtest will stay synthetic for this coin")
            continue
        # the loader looks for binance_<SYM>_1h.csv; the 'source' column records where it really came from
        k["source"] = src
        k.to_csv(DATA / f"binance_{sym}_1h.csv", index=False)
        print(coin, len(f), "funding rows,", len(k), "candles from", src)
