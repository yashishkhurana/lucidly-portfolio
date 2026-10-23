"""Uniswap v3 / Aerodrome Slipstream concentrated-liquidity math (token0/token1 units).

price P = token1 per token0.  tick t  <->  P = 1.0001**t
"""
from __future__ import annotations

import math

TICK_BASE = 1.0001


def tick_to_price(t: float) -> float:
    return TICK_BASE ** t


def price_to_tick(p: float) -> float:
    return math.log(p) / math.log(TICK_BASE)


def amounts_for_liquidity(L: float, P: float, Pa: float, Pb: float) -> tuple[float, float]:
    """Token amounts held by liquidity L in [Pa, Pb] at price P."""
    sp, sa, sb = math.sqrt(P), math.sqrt(Pa), math.sqrt(Pb)
    if sp <= sa:
        return L * (1 / sa - 1 / sb), 0.0
    if sp >= sb:
        return 0.0, L * (sb - sa)
    return L * (1 / sp - 1 / sb), L * (sp - sa)


def liquidity_for_amounts(x: float, y: float, P: float, Pa: float, Pb: float) -> float:
    sp, sa, sb = math.sqrt(P), math.sqrt(Pa), math.sqrt(Pb)
    if sp <= sa:
        return x * sa * sb / (sb - sa)
    if sp >= sb:
        return y / (sb - sa)
    return min(x * sp * sb / (sb - sp), y / (sp - sa))


def seed_from_token0_budget(budget0: float, P: float, Pa: float, Pb: float, swap_haircut: float = 0.001) -> tuple[float, float, float]:
    """Split a token0 budget into (x kept, y swapped) so liquidity is maximised, i.e. the
    position is balanced for the band at P. Returns (L, x, y) where x, y are the amounts the
    position ACTUALLY holds; see seed_with_leftover for the dust the haircut leaves undeposited."""
    L, x, y, _, _ = seed_with_leftover(budget0, P, Pa, Pb, swap_haircut)
    return L, x, y


def seed_with_leftover(budget0: float, P: float, Pa: float, Pb: float, swap_haircut: float = 0.001):
    """Returns (L, x_deposited, y_deposited, x_left, y_left). The swap haircut shortens the y leg,
    so liquidity is y-bound and part of the x leg stays undeposited (x_left); nothing vanishes:
    budget0 = x_dep + x_left + (y_dep + y_left)/P + haircut_cost."""
    sp, sa, sb = math.sqrt(P), math.sqrt(Pa), math.sqrt(Pb)
    if sp <= sa:
        L = liquidity_for_amounts(budget0, 0, P, Pa, Pb)
        return L, budget0, 0.0, 0.0, 0.0
    if sp >= sb:
        y = budget0 * P * (1 - swap_haircut)
        L = liquidity_for_amounts(0, y, P, Pa, Pb)
        return L, 0.0, y, 0.0, 0.0
    ratio = (sp - sa) * sp * sb / (sb - sp)   # y per x for a balanced position at P
    x_plan = budget0 / (1 + ratio / P)
    y = (budget0 - x_plan) * P * (1 - swap_haircut)
    L = liquidity_for_amounts(x_plan, y, P, Pa, Pb)
    x_dep, y_dep = amounts_for_liquidity(L, P, Pa, Pb)
    return L, x_dep, y_dep, max(0.0, x_plan - x_dep), max(0.0, y - y_dep)


def value_in_token0(x: float, y: float, P: float) -> float:
    return x + y / P


def band_around_tick(tick: float, width_ticks: int, spacing: int = 1) -> tuple[float, float]:
    lo = math.floor((tick - width_ticks / 2) / spacing) * spacing
    hi = lo + width_ticks
    return tick_to_price(lo), tick_to_price(hi)


def impermanent_loss_fraction(P0: float, P1: float, Pa: float, Pb: float) -> float:
    """IL of a position seeded balanced at P0 vs holding the seeded tokens, in token0 terms, as a fraction."""
    L, x0, y0 = seed_from_token0_budget(1.0, P0, Pa, Pb, 0.0)
    x1, y1 = amounts_for_liquidity(L, P1, Pa, Pb)
    hold = value_in_token0(x0, y0, P1)
    lp = value_in_token0(x1, y1, P1)
    return lp / hold - 1
