"""dnENA — fixed-yield PT leg + token-for-token perp short, held to maturity.

Economics (from the launch design): buy PT-sENA on Pendle at a discount to sENA (the implied fixed yield),
short the same token quantity of ENA on HyperCore at 1x, keep a reserve in the HL account, top up margin
from the reserve before liquidation, and roll to the next maturity only when the expected gain beats the
round-trip fee. Return = PT discount accretion (locked at entry) + funding received (or paid) on the short,
minus Pendle and HL fees. Residual risks: a sENA/ENA depeg (the PT redeems sENA, the hedge is ENA) and a
funding squeeze that outlasts the reserve.

Units: capital in USD; px = ENA/USD; pt_px = PT price in ENA terms (< 1 before maturity, 1 at maturity);
rate = sENA per PT at redemption (≈ 1.0 for a liquid-staking wrapper without a rebase; set from data).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    reserve_frac: float = 0.15          # of capital, held in the HL account beside the short's margin
    leverage: float = 1.0               # short at 1x: margin = notional
    maint_margin: float = 0.0125        # HL maintenance for ENA-class assets (1/80)
    topup_health: float = 1.5           # top up margin from the reserve when equity / maintenance falls below this
    pendle_fee: float = 0.0008          # swap fee per PT leg (buy and sell), on notional
    hl_taker: float = 0.00035           # per perp leg
    hl_slippage: float = 0.0001         # per leg; taker + slippage = 4.5 bps, what the published run charged (HL fees $38.98 on 2 × $43.3k)
    rotate_rule: str = "gain_gt_fee"    # "gain_gt_fee" | "single_print" | "never"
    rotate_margin: float = 0.0          # extra gain required over the fee, as a fraction of notional
    chunk_usd: float = 20_000           # deployment chunk for HL depth
    chunk_gap_h: int = 1


def pt_discount_apy(pt_px: float, days_to_maturity: float) -> float:
    """Implied fixed yield of a PT bought at pt_px (ENA terms) redeeming 1 token at maturity."""
    if days_to_maturity <= 0 or pt_px <= 0:
        return 0.0
    return (1.0 / pt_px) ** (365 / days_to_maturity) - 1


def size_entry(capital: float, px: float, pt_px: float, p: Params) -> dict:
    """Split capital into PT cost, short margin and reserve so that pt_qty == short_qty.
    PT costs q*pt_px*px; the 1x short needs margin q*px/lev; reserve = reserve_frac * capital."""
    reserve = capital * p.reserve_frac
    deployable = capital - reserve
    q = deployable / (pt_px * px + px / p.leverage)
    return {"qty": q, "pt_cost": q * pt_px * px, "margin": q * px / p.leverage, "reserve": reserve}


def short_liquidation_px(entry_px: float, margin: float, qty: float, maint: float) -> float:
    """Isolated short: equity = margin - qty*(px - entry); liquidation when equity = maint*qty*px."""
    return (margin + qty * entry_px) / (qty * (1 + maint))


def margin_health(margin: float, entry_px: float, px: float, qty: float, funding_cum: float, maint: float) -> float:
    equity = margin + qty * (entry_px - px) + funding_cum
    return equity / (maint * qty * px) if qty > 0 else float("inf")


def topup_needed(margin: float, entry_px: float, px: float, qty: float, funding_cum: float, p: Params) -> float:
    """USD to move from reserve to margin to restore health to topup_health (0 if healthy)."""
    h = margin_health(margin, entry_px, px, qty, funding_cum, p.maint_margin)
    if h >= p.topup_health:
        return 0.0
    target_equity = p.topup_health * p.maint_margin * qty * px
    return max(0.0, target_equity - (margin + qty * (entry_px - px) + funding_cum))


def round_trip_fee(notional: float, p: Params) -> float:
    return notional * (2 * p.pendle_fee + 2 * (p.hl_taker + p.hl_slippage))


def should_rotate(cur_apy: float, cur_days_left: float, next_apy: float, next_days: float, notional: float, p: Params) -> bool:
    """Roll from the current PT to the next maturity's PT."""
    if p.rotate_rule == "never":
        return False
    if p.rotate_rule == "single_print":
        return next_apy > cur_apy                                   # the naive rule: any better print
    # gain_gt_fee: expected extra gain over the remaining horizon must beat the round-trip cost
    horizon = max(cur_days_left, 1.0) / 365
    gain = (next_apy - cur_apy) * horizon * notional
    return gain > round_trip_fee(notional, p) * (1 + p.rotate_margin)
