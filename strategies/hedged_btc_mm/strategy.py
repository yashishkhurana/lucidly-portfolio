"""Hedged WBTC/WETH concentrated-liquidity market making — LP leg + Aave hedge leg.

Clean-room reimplementation of the "LP and hedge" strategy I built at Lucidly
(Uniswap v3 WBTC/WETH 0.05% on Arbitrum, Aave v3 WBTC collateral / WETH debt).
Units: token0 = WBTC, token1 = WETH, price P = WETH per WBTC, equity measured in WBTC.

    LP leg (60%):  liquidity L in a pre-minted tick band; rotate band when price leaves it
    Hedge (40%):   supply WBTC, borrow WETH debt D so that the WETH inventory the LP
                   accumulates as P rises is offset; levered via flashloan, LTV <= 0.70
    net WETH delta = amt1_LP - D
"""
from __future__ import annotations

from dataclasses import dataclass

from common.clmm import amounts_for_liquidity, value_in_token0


@dataclass
class Params:
    mm_alloc: float = 0.60
    hedge_alloc: float = 0.40
    max_ltv: float = 0.70
    optimal_ltv: float = 0.60
    debt_drift_bps: float = 5000     # in-range hedge re-solve trigger (50%); explainer used 20%
    swap_haircut: float = 0.001      # min-out haircut on every swap
    borrow_apy: float = 0.0235       # WETH variable borrow
    fee_tier: float = 0.0005
    asym_penalty: float = 0.05


def ltv(debt_weth: float, collateral_wbtc: float, P: float) -> float:
    return (debt_weth / P) / collateral_wbtc if collateral_wbtc > 0 else 0.0


def max_debt_weth(hedge_equity_wbtc: float, P: float, max_ltv: float) -> float:
    """Flashloan-levered hedge: collateral = E + D/P, LTV = (D/P)/collateral <= max_ltv
    -> D/P <= E * max_ltv / (1 - max_ltv)."""
    return hedge_equity_wbtc * max_ltv / (1 - max_ltv) * P


def edge_loss(L: float, Pa: float, Pb: float, P0: float, E_h: float, D: float, entry_equity: float) -> tuple[float, float]:
    """Equity loss (in WBTC) if price moves to the lower / upper band edge with hedge debt D."""
    coll = E_h + D / P0
    out = []
    for Pe in (Pa, Pb):
        x, y = amounts_for_liquidity(L, Pe, Pa, Pb)
        eq = value_in_token0(x, y, Pe) + coll - D / Pe
        out.append(entry_equity - eq)
    return out[0], out[1]


def optimise_hedge(L: float, Pa: float, Pb: float, P0: float, E_h: float, p: Params, n: int = 400) -> float:
    """Debt D in [0, D_max] minimising max(loss_low, loss_up) + penalty * |loss_low - loss_up|
    (grid search; the production code used a bounded scalar optimiser)."""
    x0, y0 = amounts_for_liquidity(L, P0, Pa, Pb)
    entry = value_in_token0(x0, y0, P0) + E_h
    best, best_d = float("inf"), 0.0
    dmax = max_debt_weth(E_h, P0, p.max_ltv)
    for i in range(n + 1):
        d = dmax * i / n
        lo, up = edge_loss(L, Pa, Pb, P0, E_h, d, entry)
        obj = max(lo, up) + p.asym_penalty * abs(lo - up)
        if obj < best:
            best, best_d = obj, d
    return best_d



def should_rebalance_hedge(debt: float, target: float, coll: float, P: float, p: Params) -> bool:
    if ltv(debt, coll, P) > p.max_ltv:
        return True
    if target <= 0:
        return debt > 0
    return abs(debt - target) * 1e4 / target >= p.debt_drift_bps
