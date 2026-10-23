"""Delta-neutral funding-carry cluster — sizing, allocation and risk rules.

Clean-room reimplementation of the Hyperliquid spot-long / perp-short cluster I ran
at Lucidly (HyperEVM BoringVault, HyperCore spot + perps, SOL/BTC/HYPE/ETH), plus the
PT-discount variant (Pendle PT long + perp short held to maturity) whose backtest page
is referenced in the portfolio index.

Position model (one asset):
    spot_qty = perp_qty = q           (1:1 token hedge, no beta adjustment)
    spot notional S = q * px_spot,   perp margin M = q * px_perp / lev
    hourly PnL = funding_rate * q * px_perp  (short receives when funding > 0)
                 - taker fees on open/close  + basis convergence at close
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    coin: str
    leverage: float = 3.0
    sz_decimals: int = 2
    tier: int = 1


@dataclass
class Params:
    deployable_frac: float = 0.995       # seed capital considered deployable
    req_buffer: float = 1.005            # multiplier on spot + margin requirements
    min_alloc_usd: float = 10.0          # HL order-notional minimum
    order_slippage: float = 0.005        # limit-price tolerance on open/close (sizing)
    realized_slippage: float = 0.0005    # fill vs mid actually paid per leg (sim)
    basis_haircut: float = 0.0005        # spot-perp basis crossed per open and per close (sim)
    part_fill_prob: float = 0.08         # chance an open leaves one leg part-/un-filled for an hour (sim)
    orphan_usd: float = 5.0              # a leg below this with the other leg open = orphaned
    hedge_drift_threshold: float = 0.05  # |spot_value/perp_value - 1| that forces a close
    liq_guard: float = 0.80              # close when price >= liq_guard * liquidation px
    taker_fee: float = 0.00035           # per-leg taker fee (HL base tier)
    spot_fee: float = 0.0004
    funding_floor: float = 0.0           # open only when the funding signal > floor
    funding_window_h: int = 1            # hours averaged for the open/close signal (1 = production's single print)
    funding_close: float = 0.0           # close when latest funding <= this
    maint_margin: float = 0.0125         # 40x max leverage asset -> 1.25% maintenance
    tier_spread: tuple = (0.7, 0.3)


# ----------------------------------------------------------------------------- sizing

def seed_allocation(total_usdc: float, n_assets: int, leverage: float, p: Params) -> float:
    """Largest spot budget x such that spot + perp margin fits: x * (1 + 1/lev) <= C."""
    cap = total_usdc * p.deployable_frac
    x = cap / (1 + 1 / leverage)
    return x if x >= p.min_alloc_usd else 0.0


def allocate(spot_budget: float, funding: dict[str, float], assets: dict[str, Asset], p: Params) -> dict[str, float]:
    """Funding-rate-weighted split within a tier; assets with funding <= floor are dropped."""
    out = {}
    tiers = sorted({a.tier for a in assets.values()})
    for t in tiers:
        coins = [c for c, a in assets.items() if a.tier == t and funding.get(c, 0.0) > p.funding_floor]
        budget = spot_budget * (p.tier_spread[t - 1] if len(tiers) > 1 else 1.0)
        w = {c: max(funding[c], 0.0) for c in coins}          # weights never negative (a no-gate run can pass f <= 0)
        tot = sum(w.values())
        for c in coins:
            alloc = budget * w[c] / tot if tot > 0 else (budget / len(coins) if coins else 0.0)
            if alloc >= p.min_alloc_usd:
                out[c] = alloc
    return out


def size_legs(alloc_usd: float, px_spot: float, px_perp: float, asset: Asset, p: Params) -> dict:
    """Token quantity for both legs (rounded down to szDecimals) and the USDC each leg needs."""
    size_px = px_spot * (1 + p.order_slippage)          # conservative sizing price
    q = round_down(alloc_usd / size_px, asset.sz_decimals)
    return {
        "qty": q,
        "spot_usd": q * size_px * p.req_buffer,
        "margin_usd": q * px_perp * (1 - p.order_slippage) / asset.leverage * p.req_buffer,
        "entry_spot": px_spot * (1 + p.realized_slippage),
        "entry_perp": px_perp * (1 - p.realized_slippage),
    }


def round_down(x: float, decimals: int) -> float:
    f = 10 ** decimals
    return int(x * f) / f


# ----------------------------------------------------------------------------- risk

def short_liquidation_price(entry_px: float, leverage: float, maint: float) -> float:
    """Isolated short: equity = margin - q*(px - entry); liquidation when equity = maint * q * px.
    margin = q*entry/lev  ->  px_liq = entry * (1 + 1/lev) / (1 + maint)."""
    return entry_px * (1 + 1 / leverage) / (1 + maint)


def hedge_drift(spot_value: float, perp_value: float) -> float:
    return abs(spot_value / abs(perp_value) - 1) if perp_value else float("inf")


def orphaned_leg(spot_value: float, perp_value: float, p: Params) -> str | None:
    """Production repair rule: spot > $5 with no perp, or perp open with spot <= $5 -> cancel + close."""
    if spot_value > p.orphan_usd and perp_value <= p.orphan_usd:
        return "orphaned-leg"
    if perp_value > p.orphan_usd and spot_value <= p.orphan_usd:
        return "orphaned-leg"
    return None


def should_close(funding_rate: float, px: float, liq_px: float, spot_value: float, perp_value: float, p: Params) -> str | None:
    if funding_rate <= p.funding_close:
        return "funding<=0"
    if px >= p.liq_guard * liq_px:
        return "liq-guard"
    if hedge_drift(spot_value, perp_value) > p.hedge_drift_threshold:
        return "hedge-drift"
    return None


# ----------------------------------------------------------------------------- PT variant

def pt_discount_apy(pt_price: float, spot: float, days_to_maturity: float) -> float:
    """Implied fixed yield of a PT bought at pt_price redeeming 1 token at maturity."""
    if days_to_maturity <= 0 or pt_price <= 0:
        return 0.0
    return (spot / pt_price) ** (365 / days_to_maturity) - 1


def pt_delta(pt_price: float, spot: float) -> float:
    """Approximate spot delta of one PT: its market value in units of the underlying.
    Converges to 1.0 at maturity; below 1 while the PT trades at a discount."""
    return pt_price / spot
