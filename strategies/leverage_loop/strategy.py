"""Leveraged stablecoin loop cluster — allocation and risk logic.

Clean-room reimplementation of the control layer I designed and ran at Lucidly
(Morpho Blue, Ethereum mainnet, June 2026). Everything here is pure functions
on numbers so it can be unit-tested and replayed against archive data.

Position model (one loop):
    equity e, leverage L  ->  collateral L*e, debt (L-1)*e, LTV (L-1)/L
    levered net APY       ->  net(L) = L*y - (L-1)*b = y + (L-1)*(y - b)

The gating metric is always net(L_target), never the raw spread y - b.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Market:
    key: str
    risk_score: float          # allocator divisor; lower = safer = more capital
    max_weight: float          # hard cap on equity share
    target_leverage: float = 8.0
    lltv: float = 0.915        # on-chain liquidation LTV
    entry_fee_bps: float = 0.0  # one-time wrap fee on GROSS (collateral) notional
    entry_cost_apy: float = 0.0  # annualised drag subtracted from the score


@dataclass
class Params:
    alloc_beta: float = 1.0
    alloc_max_weight: float = 0.60
    lever_margin_bps: float = 25
    withdrawal_buffer_bps: float = 200
    max_market_ownership_bps: float = 2500
    rebalance_band_bps: float = 1000
    min_rebalance_usd: float = 50
    close_after_samples: int = 6        # consecutive negative-net samples before de-lever
    reenter_margin_bps: float = 50
    reenter_samples: int = 3
    markdown_close_bps: float = 150
    markdown_sustain_ticks: int = 2
    max_ltv_headroom_frac: float = 0.375
    emergency_ltv_headroom_frac: float = 0.625


# ----------------------------------------------------------------------------- economics

def net_apy(leverage: float, collateral_yield: float, borrow_rate: float) -> float:
    return leverage * collateral_yield - (leverage - 1) * borrow_rate


def inversion_borrow_rate(leverage: float, collateral_yield: float) -> float:
    """Borrow rate at which net(L) = 0:  b* = L/(L-1) * y."""
    return leverage / (leverage - 1) * collateral_yield if leverage > 1 else float("inf")


def ltv_from_leverage(L: float) -> float:
    return (L - 1) / L


def leverage_from_ltv(ltv: float) -> float:
    return 1 / (1 - ltv)


def liquidation_markdown(L: float, lltv: float) -> float:
    """Same-block collateral exchange-rate markdown m* that lifts LTV to LLTV:
    LTV0/(1-m) = LLTV  ->  m* = 1 - LTV0/LLTV."""
    return 1 - ltv_from_leverage(L) / lltv


def ltv_bands(m: Market, p: Params) -> dict:
    t = ltv_from_leverage(m.target_leverage)
    return {
        "target_ltv": t,
        "max_ltv": t + (m.lltv - t) * p.max_ltv_headroom_frac,
        "emergency_ltv": t + (m.lltv - t) * p.emergency_ltv_headroom_frac,
    }


def equity_pnl_for_markdown(L: float, m: float, lltv: float) -> tuple[float, bool]:
    """Equity P&L fraction and whether the position is liquidated for a markdown m."""
    ltv_after = ltv_from_leverage(L) / (1 - m)
    if ltv_after >= lltv:
        return -1.0, True
    return -L * m, False


# ----------------------------------------------------------------------------- allocation

def risk_adjusted_score(m: Market, y: float, b: float, p: Params) -> float:
    n = net_apy(m.target_leverage, y, b) - m.entry_cost_apy
    if n <= p.lever_margin_bps / 1e4:
        return 0.0
    return n / (m.risk_score ** p.alloc_beta)


def target_weights(scores: dict[str, float], caps: dict[str, float], fallback_cap: float) -> dict[str, float]:
    """Proportional to score, clamped to per-market caps, freed mass renormalised
    onto uncapped markets ("spill-on-cap"). Weights sum to <= 1; the remainder is idle."""
    w = {k: 0.0 for k in scores}
    live = {k: s for k, s in scores.items() if s > 0}
    if not live:
        return w
    remaining = 1.0
    pool = dict(live)
    for _ in range(len(live) + 1):
        tot = sum(pool.values())
        if tot <= 0 or remaining <= 1e-12:
            break
        capped_now = {}
        for k, s in pool.items():
            share = remaining * s / tot
            cap = caps.get(k, fallback_cap)
            if share >= cap - w[k]:
                capped_now[k] = cap - w[k]
        if not capped_now:
            for k, s in pool.items():
                w[k] += remaining * s / tot
            remaining = 0
            break
        for k, add in capped_now.items():
            w[k] += add
            remaining -= add
            pool.pop(k)
    return w


# ----------------------------------------------------------------------------- state machine

ACTIVE, UNLEVERED, PARKED = "ACTIVE", "UNLEVERED", "PARKED"


@dataclass
class LoopState:
    state: str = ACTIVE
    neg_streak: int = 0
    pos_streak: int = 0
    hwm: float = 0.0
    markdown_streak: int = 0
    equity: float = 0.0


def next_net_state(st: LoopState, net: float, p: Params) -> str:
    """Spread trigger (patient): sustained negative net de-levers to 1x; sustained
    positive net above the re-enter margin re-arms. Does not touch PARKED->ACTIVE's
    markdown condition (handled in next_markdown_state)."""
    if net < 0:
        st.neg_streak += 1
        st.pos_streak = 0
    else:
        st.neg_streak = 0
        st.pos_streak = st.pos_streak + 1 if net > p.reenter_margin_bps / 1e4 else 0
    if st.state == ACTIVE and st.neg_streak >= p.close_after_samples:
        st.state = UNLEVERED
        st.neg_streak = 0
    elif st.state in (UNLEVERED, PARKED) and st.pos_streak >= p.reenter_samples and st.markdown_streak == 0:
        st.state = ACTIVE
        st.pos_streak = 0
    return st.state


def next_markdown_state(st: LoopState, price: float, p: Params) -> str:
    """Markdown trigger (reactive): exchange rate below HWM by markdown_close_bps for
    markdown_sustain_ticks consecutive ticks -> full exit to USDC."""
    st.hwm = max(st.hwm, price)
    if st.hwm > 0 and price < st.hwm * (1 - p.markdown_close_bps / 1e4):
        st.markdown_streak += 1
    else:
        st.markdown_streak = 0
    if st.markdown_streak >= p.markdown_sustain_ticks and st.state != PARKED:
        st.state = PARKED
    return st.state


# ----------------------------------------------------------------------------- sizing

def concentration_headroom(market_supply: float, our_debt: float, bps: float) -> float:
    return max(0.0, bps / 1e4 * market_supply - our_debt)


def size_lever(target_equity: float, L: float, available_liquidity: float, market_supply: float,
               our_debt: float, p: Params, per_tick_cap: float = 1_000_000) -> float:
    """Equity that can actually be levered this tick given borrow-side caps."""
    want_borrow = (L - 1) * target_equity
    cap = min(want_borrow, available_liquidity,
              concentration_headroom(market_supply, our_debt, p.max_market_ownership_bps), per_tick_cap)
    return cap / (L - 1) if L > 1 else target_equity


def should_rebalance(current: float, target: float, p: Params) -> bool:
    band = max(p.min_rebalance_usd, p.rebalance_band_bps / 1e4 * target)
    return abs(target - current) > band
