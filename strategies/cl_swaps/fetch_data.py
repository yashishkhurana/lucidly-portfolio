"""Pull Uniswap v3 / Aerodrome Slipstream `Swap` events for a pool over a time window via plain JSON-RPC,
so the CL backtests can charge fees from MEASURED volume and pool liquidity instead of a modelled rate.

Run (needs an RPC with eth_getLogs; Alchemy works):
    ARBITRUM_RPC_URL=https://arb-mainnet.g.alchemy.com/v2/<key> \
    python -m strategies.cl_swaps.fetch_data --pool wbtc_weth_arb
    BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/<key> \
    python -m strategies.cl_swaps.fetch_data --pool eurc_usdc_base
    python -m strategies.cl_swaps.fetch_data --pool wbtc_weth_arb --start 2026-03-10T15:33Z --end 2026-03-11T17:10Z

Output: data/swaps_<pool>.csv with columns
    time, block, tx, amount0, amount1 (human units, signed from the pool's view: positive = token in),
    price (token1 per token0, human units, from sqrtPriceX96 AFTER the swap), liquidity (pool liquidity in range
    after the swap, raw), tick
No web3 dependency: eth_getLogs + eth_getBlockByNumber over urllib, decoded by hand.
Event: Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1,
            uint160 sqrtPriceX96, uint128 liquidity, int24 tick)   — identical on Uniswap v3 and Slipstream.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from common.data import DATA

SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
Q96 = 2 ** 96


@dataclass(frozen=True)
class Pool:
    key: str
    address: str
    rpc_env: str
    dec0: int
    dec1: int
    fee_tier: float
    label: str
    default_start: str
    default_end: str


POOLS = {
    "wbtc_weth_arb": Pool("wbtc_weth_arb", "0x2f5e87C9312fa29aed5c179E456625D79015299c", "ARBITRUM_RPC_URL", 8, 18, 0.0005,
                          "Uniswap v3 WBTC/WETH 0.05% · Arbitrum", "2026-03-10T15:33:00Z", "2026-03-11T17:10:00Z"),
    # Aerodrome Slipstream EURC/USDC on Base, tickSpacing 50 (CLFactory.getPool); fee() returned 76 pips = 0.0076% on 2026-10-23
    # (Slipstream fees are dynamic, so the Jan-2026 tier may have differed). dec0/dec1 are both 6 (EURC, USDC).
    "eurc_usdc_base": Pool("eurc_usdc_base", os.environ.get("EURC_USDC_POOL", "0xe846373c1a92b167b4e9cd5d8e4d6b1db9e90ec7"),
                           "BASE_RPC_URL", 6, 6, 0.000076, "Aerodrome Slipstream EURC/USDC · Base", "2026-01-05T00:00:00Z", "2026-01-05T06:00:00Z"),
}


# ----------------------------------------------------------------------------- rpc

def rpc(url: str, method: str, params: list, _id: int = 1):
    body = json.dumps({"jsonrpc": "2.0", "id": _id, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode())
            if "error" in out:
                raise RuntimeError(out["error"])
            return out["result"]
        except Exception as e:  # rate limits / transient
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
            last = e
    raise last


def rpc_batch(url: str, calls: list[tuple[str, list]]):
    body = json.dumps([{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in enumerate(calls)]).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    out = sorted(out, key=lambda o: o["id"])
    return [o.get("result") for o in out]


def block_at(url: str, ts: int, lo: int | None = None, hi: int | None = None) -> int:
    """First block with timestamp >= ts, by bisection on eth_getBlockByNumber."""
    hi = hi or int(rpc(url, "eth_blockNumber", []), 16)
    lo = lo or 1
    while lo < hi:
        mid = (lo + hi) // 2
        b = rpc(url, "eth_getBlockByNumber", [hex(mid), False])
        if int(b["timestamp"], 16) < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------- decode

def _int256(word: str) -> int:
    v = int(word, 16)
    return v - (1 << 256) if v >= (1 << 255) else v


def _int24(word: str) -> int:
    v = int(word, 16) & ((1 << 24) - 1)
    return v - (1 << 24) if v >= (1 << 23) else v


def decode_swap(log: dict, pool: Pool) -> dict:
    d = log["data"][2:]
    words = [d[i:i + 64] for i in range(0, len(d), 64)]
    amount0, amount1 = _int256(words[0]), _int256(words[1])
    sqrt_p = int(words[2], 16)
    liquidity = int(words[3], 16)
    tick = _int24(words[4])
    price = (sqrt_p / Q96) ** 2 * 10 ** (pool.dec0 - pool.dec1)
    return {"block": int(log["blockNumber"], 16), "tx": log["transactionHash"],
            "amount0": amount0 / 10 ** pool.dec0, "amount1": amount1 / 10 ** pool.dec1,
            "price": price, "liquidity": liquidity, "tick": tick, "log_index": int(log["logIndex"], 16)}


# ----------------------------------------------------------------------------- fetch

def pool_fee_pips(url: str, pool: Pool, block: int) -> int | None:
    """fee() (selector 0xddca3f43) at a historical block; Uniswap v3 returns uint24 pips, Slipstream likewise."""
    try:
        res = rpc(url, "eth_call", [{"to": pool.address, "data": "0xddca3f43"}, hex(block)])
        return int(res, 16)
    except Exception as e:  # non-archive node or unusual pool: fall back to the configured tier
        print("fee() read failed:", e)
        return None


def fetch_swaps(pool: Pool, url: str, start: datetime, end: datetime, chunk: int = 2000) -> pd.DataFrame:
    b0 = block_at(url, int(start.timestamp()))
    b1 = block_at(url, int(end.timestamp()), lo=b0)
    print(f"{pool.key}: blocks {b0:,} → {b1:,} ({b1 - b0:,} blocks)")
    fee_pips = pool_fee_pips(url, pool, b0)
    print(f"  pool fee() at block {b0:,}: {fee_pips} pips" if fee_pips is not None else f"  using configured tier {pool.fee_tier}")
    rows = []
    frm = b0
    while frm <= b1:
        to = min(frm + chunk - 1, b1)
        try:
            logs = rpc(url, "eth_getLogs", [{"fromBlock": hex(frm), "toBlock": hex(to), "address": pool.address, "topics": [SWAP_TOPIC]}])
        except RuntimeError as e:  # provider asks for a smaller range
            if chunk > 100:
                chunk //= 2
                continue
            raise e
        rows.extend(decode_swap(l, pool) for l in logs)
        frm = to + 1
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # block timestamps in batches of 100
    blocks = sorted(df["block"].unique())
    ts = {}
    for i in range(0, len(blocks), 100):
        bs = blocks[i:i + 100]
        res = rpc_batch(url, [("eth_getBlockByNumber", [hex(b), False]) for b in bs])
        for b, r in zip(bs, res):
            ts[b] = int(r["timestamp"], 16)
        time.sleep(0.05)
    df["time"] = pd.to_datetime(df["block"].map(ts), unit="s", utc=True)
    df["fee_pips"] = fee_pips if fee_pips is not None else round(pool.fee_tier * 1e6)
    df = df.sort_values(["block", "log_index"]).reset_index(drop=True)
    return df[["time", "block", "tx", "amount0", "amount1", "price", "liquidity", "tick", "fee_pips"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--rpc", help="override the RPC URL (else read from the pool's env var)")
    a = ap.parse_args()
    pool = POOLS[a.pool]
    url = a.rpc or os.environ.get(pool.rpc_env)
    if not url:
        raise SystemExit(f"set {pool.rpc_env} (or --rpc)")
    if int(pool.address, 16) == 0:
        raise SystemExit("set EURC_USDC_POOL to the Slipstream pool address first")
    start = datetime.fromisoformat((a.start or pool.default_start).replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat((a.end or pool.default_end).replace("Z", "+00:00")).astimezone(timezone.utc)
    df = fetch_swaps(pool, url, start, end)
    out = DATA / f"swaps_{pool.key}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} swaps to {out}")
    if len(df):
        vol0 = df["amount0"].abs().sum() / 2
        print(f"  {df['time'].iloc[0]} → {df['time'].iloc[-1]}  volume ≈ {vol0:,.2f} token0-eq  price {df['price'].iloc[0]:.6g} → {df['price'].iloc[-1]:.6g}")
