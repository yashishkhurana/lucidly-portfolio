"""USDC lending cluster — mean-variance allocator with risk haircuts, eligibility gates and planner.

Clean-room reimplementation of the Morpho Blue supply cluster I built at Lucidly (mainnet), with the
junior-tranche (Royco) variant's score rule, which prices the junior's depeg leverage before allocation.

    snapshot -> allocator -> planner -> Morpho supply/withdraw -> merkle manager payload -> guarded submit
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Market:
    key: str
    risk_score: float
    max_weight: float
    min_weight: float
    hurdle: float                 # annual APY the 6h-mean must clear to be allocated
    concentration_cap: float      # share of market borrows we may supply
    abs_cap: float                # absolute USDC cap
    min_deploy: float
    max_deploy: float
    utilization_cap: float = 0.95


@dataclass
class Params:
    risk_aversion: float = 1.0
    baseline_risk: float = 3.0
    risk_penalty_per_point: float = 0.005     # 50 bps per risk point above baseline
    hurdle_breach_ticks: int = 3              # consecutive ticks below hurdle before blocked
    exit_band: float = 0.005                  # existing position tolerated to hurdle - 50 bps
    exit_breach_ticks: int = 3
    min_total_rebalance_bps: float = 100      # drift gate
    min_rebalance_assets: float = 1_000
    max_deposit_util_drop_bps: float = 100
    withdraw_liquidity_fraction: float = 0.80
    withdraw_buffer_bps: float = 50


@dataclass
class MarketState:
    hurdle_breach: int = 0
    exit_breach: int = 0
    blocked: bool = False


def supply_apy(borrow_apr: float, utilization: float, fee: float = 0.0) -> float:
    return borrow_apr * utilization * (1 - fee)


def capacity(m: Market, total_borrow: float) -> float:
    return min(m.abs_cap, m.concentration_cap * total_borrow)


def max_deposit_for_util_drop(supply: float, borrow: float, util: float, drop_bps: float) -> float:
    """Deposit that lowers utilisation by at most drop_bps."""
    target = util - drop_bps / 1e4
    return max(0.0, borrow / target - supply) if target > 0 else 0.0


def update_gates(st: MarketState, m: Market, mean_apy: float, has_position: bool, p: Params) -> bool:
    """Returns True if the market is eligible for NEW allocation. Updates breach counters."""
    if mean_apy < m.hurdle:
        st.hurdle_breach += 1
    else:
        st.hurdle_breach = 0
    if has_position and mean_apy < m.hurdle - p.exit_band:
        st.exit_breach += 1
    else:
        st.exit_breach = 0
    st.blocked = st.hurdle_breach >= p.hurdle_breach_ticks
    return not st.blocked


def must_exit(st: MarketState, p: Params) -> bool:
    return st.exit_breach >= p.exit_breach_ticks


def solve_weights(mu: np.ndarray, cov: np.ndarray, risk: np.ndarray, upper: np.ndarray, p: Params) -> np.ndarray:
    """max mu_adj'w - (lambda/2) w'cov w   s.t. 0 <= w <= upper, sum(w) = min(1, sum(upper))."""
    n = len(mu)
    mu_adj = mu - np.maximum(0, risk - p.baseline_risk) * p.risk_penalty_per_point
    total = min(1.0, float(upper.sum()))
    if total <= 0:
        return np.zeros(n)
    # greedy warm start in order of adjusted return
    w0 = np.zeros(n)
    rem = total
    for i in np.argsort(-mu_adj):
        w0[i] = min(upper[i], rem)
        rem -= w0[i]
    obj = lambda w: -(mu_adj @ w - p.risk_aversion / 2 * w @ cov @ w)  # noqa: E731
    res = minimize(obj, w0, method="SLSQP", bounds=[(0, u) for u in upper],
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - total}])
    if not res.success:
        return w0
    return np.clip(res.x, 0, upper)


def plan(current: dict[str, float], target: dict[str, float], idle: float, liquidity: dict[str, float], p: Params) -> list[dict]:
    """Drift gate, then deposits sized by idle and withdrawals by visible liquidity."""
    port = idle + sum(current.values())
    drift = sum(abs(target.get(k, 0) - current.get(k, 0)) for k in set(current) | set(target))
    if port <= 0 or drift / port * 1e4 < p.min_total_rebalance_bps:
        return []
    actions = []
    for k in sorted(set(current) | set(target), key=lambda k: target.get(k, 0) - current.get(k, 0)):
        d = target.get(k, 0) - current.get(k, 0)
        if abs(d) < p.min_rebalance_assets:
            continue
        if d < 0:
            amt = min(-d, p.withdraw_liquidity_fraction * liquidity.get(k, 0), current[k])
            if amt > 0:
                actions.append({"market": k, "kind": "withdraw", "assets": amt})
                idle += amt
        else:
            amt = min(d, idle)
            if amt >= p.min_rebalance_assets:
                actions.append({"market": k, "kind": "supply", "assets": amt})
                idle -= amt
    return actions


# ----------------------------------------------------------------------------- junior-tranche variant

def tranche_score(apy: float, risk_score: float, beta: float = 1.0) -> float:
    return apy / risk_score ** beta


def junior_loss_from_depeg(depeg: float, junior_share: float) -> float:
    """A junior tranche absorbs the whole pool's loss first: loss_jr = min(1, depeg / junior_share).
    With a 25% junior tranche an 8.5% depeg is a 34% junior loss (~4x leverage)."""
    return min(1.0, depeg / junior_share) if junior_share > 0 else 1.0


def junior_apy(underlying_apy: float, junior_share: float, senior_rate: float) -> float:
    """Junior tranche yield when the senior takes a fixed rate: everything the underlying earns above the senior coupon
    accrues to the junior, scaled by its share of the pool. Negative when the underlying cannot cover the coupon."""
    if junior_share <= 0:
        return 0.0
    return (underlying_apy - senior_rate * (1 - junior_share)) / junior_share


def depeg_risk_penalty(px_vol_ann: float, junior_share: float, lam: float = 1.0) -> float:
    """Ex-ante price of the junior's depeg leverage: the underlying's annualised exchange-rate volatility multiplied by the
    first-loss leverage 1/junior_share, scaled by lam. Subtracted from the junior APY before scoring, so a tranche whose
    cushion is thin relative to how its underlying moves scores low or is excluded — the guardrail applied as a rule."""
    if junior_share <= 0:
        return float("inf")
    return lam * px_vol_ann / junior_share


def tranche_eligible(jr_apy: float, penalty: float, trailing_drawdown: float, junior_share: float, max_cushion_use: float = 0.25) -> bool:
    """Admit a junior tranche only when its risk-adjusted yield is positive and the underlying's trailing drawdown has
    used less than `max_cushion_use` of the junior cushion (loss_jr = drawdown / junior_share)."""
    return (jr_apy - penalty) > 0 and junior_loss_from_depeg(trailing_drawdown, junior_share) < max_cushion_use
