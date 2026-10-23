"""Shared concentrated-liquidity simulation engine (single active band, optional Aave-style hedge leg).

Used by strategies/hedged_btc_mm and strategies/eurc_usdc_mm. Units: everything in token0
(WBTC for WBTC/WETH, EURC for EURC/USDC); price P = token1 per token0.

Fees come in two modes:
  measured — when `path` carries per-swap columns (amount0, amount1, liquidity; from strategies/cl_swaps),
             every in-range swap pays `amount_in * fee_tier * L_strategy / (L_pool + L_strategy)`, with the
             fee taken in the token that came in and valued in token0 at the swap price. This is the pool's
             own arithmetic; nothing is assumed.
  modelled — otherwise, fee income accrues on in-range LP value at `fee_apr_ref * ref_width / width`
             (fee share scales with concentration). The bundled price-only paths use this; it is an
             assumption and every page says so.
Token amounts, IL, hedge debt, LTV and borrow interest are exact arithmetic in both modes.

Benchmarks reported:
  pnl           = equity_end - deposit   (token0 terms; includes revaluation of the token1 inventory)
  pnl_vs_hold   = equity_end - value at the end price of the tokens the first balancing swap would have
                  produced with NO haircut (the "hold" alternative to running the LP). Reported with its
                  components: fees, borrow, slippage (seeding + rotations), realised IL, in-band IL, and
                  `reval` = the remainder: revaluation at the end price of IL realised at earlier rotation
                  prices and of the haircut tokens. ~0 on a no-rotation path; on rotation paths it can be a
                  few percent of the total and is shown as its own term, never folded into IL.
Undeposited dust left by the haircut (the y leg is short, so liquidity is y-bound) is carried as idle
inventory in equity and folded into the next re-seed; it never disappears.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from common import metrics
from common.clmm import amounts_for_liquidity, band_around_tick, price_to_tick, seed_with_leftover, tick_to_price, value_in_token0

MINUTE = 1 / (365 * 24 * 60)
REF_WIDTH = 500


@dataclass
class CLParams:
    mm_alloc: float = 1.0
    hedge_alloc: float = 0.0
    max_ltv: float = 0.70
    optimal_ltv: float = 0.60
    debt_drift_bps: float = 5000
    swap_haircut: float = 0.001
    borrow_apy: float = 0.0
    asym_penalty: float = 0.05
    hedge_resolve_every: int = 60   # rows between in-range hedge re-solves
    fee_tier: float = 0.0005        # pool fee, used only in measured mode
    dec0: int = 8                   # token decimals, to bring the event's raw pool liquidity to human units
    dec1: int = 18


def simulate(path: pd.DataFrame, width: int, p: CLParams, deposit: float, fee_apr_ref: float,
             spacing: int = 1, hedge_fns=None, band_ticks: tuple[int, int] | None = None) -> dict:
    """hedge_fns = (optimise_hedge, should_rebalance_hedge, ltv) from strategies.hedged_btc_mm.strategy
    when a hedge leg is wanted; None runs the LP leg alone."""
    hedged = hedge_fns is not None and p.hedge_alloc > 0
    if hedged:
        optimise_hedge, should_rebalance_hedge, ltv_fn = hedge_fns
    P0 = float(path["price"].iloc[0])
    mm_budget, E_h = deposit * p.mm_alloc, deposit * p.hedge_alloc
    if band_ticks is not None:                              # a given first band (e.g. the one a log used)
        Pa, Pb = tick_to_price(band_ticks[0]), tick_to_price(band_ticks[1])
    else:
        Pa, Pb = band_around_tick(price_to_tick(P0), width, spacing)
    L, x, y, x_idle, y_idle = seed_with_leftover(mm_budget, P0, Pa, Pb, p.swap_haircut)
    _, xh, yh, xh_l, yh_l = seed_with_leftover(mm_budget, P0, Pa, Pb, 0.0)
    x_seed, y_seed = xh + xh_l, yh + yh_l                   # hold benchmark: the no-haircut split, never touched
    slippage = mm_budget - value_in_token0(x + x_idle, y + y_idle, P0)   # haircut on the seeding swap
    D = optimise_hedge(L, Pa, Pb, P0, E_h, p) if hedged else 0.0
    coll = E_h + D / P0
    slippage += (D / P0) * p.swap_haircut
    fees = il = borrow = 0.0
    x_e, y_e = x, y
    rebalances = 0
    rows = []
    fee_rate = fee_apr_ref * REF_WIDTH / width
    measured = {"amount0", "amount1", "liquidity"}.issubset(path.columns)
    liq_scale = 10 ** ((p.dec0 + p.dec1) / 2)   # L_raw = sqrt(x_raw * y_raw) -> L_human = L_raw / sqrt(10^dec0 * 10^dec1)
    has_fee = measured and "fee_pips" in path.columns          # per-row historical fee (dynamic-fee pools)
    cols = ["time", "price"] + (["amount0", "amount1", "liquidity"] if measured else []) + (["fee_pips"] if has_fee else [])
    prev_t = path["time"].iloc[0]
    for i, row in enumerate(path[cols].itertuples(index=False)):
        t, P = row[0], float(row[1])
        elapsed = (t - prev_t).total_seconds() / 60          # real elapsed minutes; 0 for same-second rows
        prev_t = t
        x, y = amounts_for_liquidity(L, P, Pa, Pb)
        in_range = Pa < P < Pb
        lp_val = value_in_token0(x, y, P)
        if in_range:
            if measured:
                a0, a1, l_pool = float(row[2]), float(row[3]), float(row[4]) / liq_scale
                tier = float(row[5]) / 1e6 if has_fee else p.fee_tier
                share = L / (l_pool + L) if (l_pool + L) > 0 else 0.0
                if a0 > 0:                     # token0 came in: fee in token0
                    fees += a0 * tier * share
                elif a1 > 0:                   # token1 came in: fee in token1, valued in token0
                    fees += a1 * tier * share / P
            else:
                fees += lp_val * fee_rate * MINUTE * elapsed
        borrow += (D / P) * p.borrow_apy * MINUTE * elapsed
        if not in_range:
            il += value_in_token0(x_e, y_e, P) - lp_val
            Pa, Pb = band_around_tick(price_to_tick(P), width, spacing)
            budget = lp_val + value_in_token0(x_idle, y_idle, P)       # idle dust folded back in
            L, x, y, x_idle, y_idle = seed_with_leftover(budget, P, Pa, Pb, p.swap_haircut)
            slippage += budget - value_in_token0(x + x_idle, y + y_idle, P)
            x_e, y_e = x, y
            if hedged:
                E_cur = coll - D / P
                newD = optimise_hedge(L, Pa, Pb, P, E_cur, p)
                slippage += abs(newD - D) / P * p.swap_haircut
                D = newD
                coll = E_cur + D / P
            rebalances += 1
        elif hedged and i % p.hedge_resolve_every == 0:
            E_cur = coll - D / P
            target = optimise_hedge(L, Pa, Pb, P, E_cur, p, n=80)
            if should_rebalance_hedge(D, target, coll, P, p):
                slippage += abs(target - D) / P * p.swap_haircut
                D = target
                coll = E_cur + D / P
                rebalances += 1
        hedge_eq = coll - D / P
        idle_val = value_in_token0(x_idle, y_idle, P)
        equity = lp_val + idle_val + hedge_eq + fees - borrow
        hold = value_in_token0(x_seed, y_seed, P) + E_h
        in_band_il = value_in_token0(x_e, y_e, P) - lp_val          # current band's entry tokens vs LP now
        rows.append({"time": t, "price": P, "equity": equity, "hold": hold, "lp_value": lp_val, "idle": idle_val,
                     "fees": fees, "il": il, "in_band_il": in_band_il, "slippage": slippage, "borrow": borrow,
                     "net_token1_delta": y - D,
                     "ltv": ltv_fn(D, coll, P) if hedged else 0.0, "lower": Pa, "upper": Pb})
    c = pd.DataFrame(rows).set_index("time")
    hours = (c.index[-1] - c.index[0]).total_seconds() / 3600
    pnl = float(c["equity"].iloc[-1] - deposit)
    pnl_vs_hold = float(c["equity"].iloc[-1] - c["hold"].iloc[-1])
    in_band_il = float(c["in_band_il"].iloc[-1])
    # remainder after the named terms: revaluation of earlier-realised IL and haircut tokens at the end price
    reval = pnl_vs_hold - (fees - borrow - slippage - il - in_band_il)
    return {"curve": c, "pnl": pnl, "pnl_vs_hold": pnl_vs_hold, "hours": hours, "days": hours / 24,
            "rebalances": rebalances, "fees": fees, "il": il, "in_band_il": in_band_il, "slippage": slippage,
            "borrow": borrow, "reval": reval, "dust_reval": reval, "fee_mode": "measured" if measured else "modelled",
            "max_dd": metrics.max_drawdown(c["equity"]), "max_abs_delta": float(c["net_token1_delta"].abs().max()),
            "max_ltv": float(c["ltv"].max()), "deposit": deposit}
