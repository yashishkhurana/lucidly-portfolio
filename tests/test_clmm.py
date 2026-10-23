import math

import numpy as np

import pytest

from common.clmm import (amounts_for_liquidity, band_around_tick, impermanent_loss_fraction, liquidity_for_amounts,
                         price_to_tick, seed_from_token0_budget, tick_to_price, value_in_token0)


def test_tick_price_round_trip():
    for t in (-50000, 0, 1234, 265640):
        assert abs(price_to_tick(tick_to_price(t)) - t) < 1e-6


def test_amounts_liquidity_round_trip_in_range():
    P, Pa, Pb = 34.35, 33.9, 34.8
    L = 2830.0
    x, y = amounts_for_liquidity(L, P, Pa, Pb)
    assert liquidity_for_amounts(x, y, P, Pa, Pb) == pytest.approx(L, rel=1e-9)


def test_amounts_at_edges_are_single_sided():
    L, Pa, Pb = 1000.0, 1.0, 1.21
    x_lo, y_lo = amounts_for_liquidity(L, 0.9, Pa, Pb)
    x_hi, y_hi = amounts_for_liquidity(L, 1.3, Pa, Pb)
    assert y_lo == 0 and x_lo > 0
    assert x_hi == 0 and y_hi > 0
    # below the range the position holds exactly L*(1/sqrt(Pa) - 1/sqrt(Pb)) of token0
    assert x_lo == pytest.approx(L * (1 / math.sqrt(Pa) - 1 / math.sqrt(Pb)))


def test_seed_from_budget_conserves_value_without_haircut():
    budget, P, Pa, Pb = 6.0, 34.35, 33.9, 34.8
    L, x, y = seed_from_token0_budget(budget, P, Pa, Pb, swap_haircut=0.0)
    assert value_in_token0(x, y, P) == pytest.approx(budget, rel=1e-9)
    x2, y2 = amounts_for_liquidity(L, P, Pa, Pb)
    assert x2 == pytest.approx(x, rel=1e-6) and y2 == pytest.approx(y, rel=1e-6)


def test_seed_haircut_reduces_value():
    budget, P, Pa, Pb = 6.0, 34.35, 33.9, 34.8
    _, x, y = seed_from_token0_budget(budget, P, Pa, Pb, swap_haircut=0.001)
    assert value_in_token0(x, y, P) < budget


def test_il_is_zero_at_entry_and_negative_elsewhere():
    P0, Pa, Pb = 1.0, 0.9, 1.1
    assert impermanent_loss_fraction(P0, P0, Pa, Pb) == pytest.approx(0.0, abs=1e-12)
    for P1 in (0.95, 1.05, 1.1, 0.9):
        assert impermanent_loss_fraction(P0, P1, Pa, Pb) < 0


def test_full_range_il_matches_closed_form():
    # full-range LP: IL(k) = 2*sqrt(k)/(1+k) - 1 for price ratio k
    P0, k = 1.0, 1.5
    il = impermanent_loss_fraction(P0, P0 * k, 1e-9, 1e9)
    assert il == pytest.approx(2 * math.sqrt(k) / (1 + k) - 1, abs=1e-6)


def test_narrow_band_il_exceeds_full_range_il():
    P0, k = 1.0, 1.02
    narrow = impermanent_loss_fraction(P0, P0 * k, 0.975, 1.025)
    wide = impermanent_loss_fraction(P0, P0 * k, 0.5, 2.0)
    assert narrow < wide < 0


def test_band_around_tick_respects_spacing():
    Pa, Pb = band_around_tick(265640, 250, spacing=50)
    lo, hi = price_to_tick(Pa), price_to_tick(Pb)
    assert round(lo) % 50 == 0 and round(hi - lo) == 250


def test_seed_with_haircut_conserves_tokens():
    from common.clmm import seed_with_leftover
    budget, P, Pa, Pb, h = 6.0, 34.35, 33.9, 34.8, 0.001
    L, x, y, x_left, y_left = seed_with_leftover(budget, P, Pa, Pb, h)
    # deposited amounts are exactly what liquidity L holds
    x2, y2 = amounts_for_liquidity(L, P, Pa, Pb)
    assert x2 == pytest.approx(x, rel=1e-9) and y2 == pytest.approx(y, rel=1e-9)
    # nothing vanishes: deposited + leftover + haircut cost == budget
    swapped_in = (budget - (x + x_left)) * P                 # token0 sold for token1
    haircut_cost0 = swapped_in * h / P
    assert value_in_token0(x + x_left, y + y_left, P) + haircut_cost0 == pytest.approx(budget, rel=1e-9)
    assert x_left >= 0 and y_left >= 0


def test_engine_decomposition_is_exact_without_rotations():
    import pandas as pd
    from common.cl_engine import CLParams, simulate
    t = pd.date_range("2026-01-01", periods=200, freq="min", tz="UTC")
    price = 1.0 + 0.001 * np.sin(np.linspace(0, 6, 200))    # stays inside a 50-tick band
    path = pd.DataFrame({"time": t, "price": price})
    r = simulate(path, 50, CLParams(swap_haircut=0.0005), 1000.0, 0.10, spacing=10)
    assert r["rebalances"] == 0 and r["il"] == 0.0
    # with no rotation the remainder term must be negligible: the named terms explain the result
    assert abs(r["reval"]) < 0.02 * abs(r["pnl_vs_hold"]) + 1e-6
    assert r["pnl_vs_hold"] == pytest.approx(r["fees"] - r["slippage"] - r["in_band_il"], rel=0.02)


def test_swap_decoder_and_measured_fee_mode():
    import pandas as pd
    from common.cl_engine import CLParams, simulate
    from strategies.cl_swaps.fetch_data import POOLS, Q96, decode_swap
    pool = POOLS["wbtc_weth_arb"]

    def word(v):
        return (v % (1 << 256)).to_bytes(32, "big").hex()
    sqrt_p = int((34.35 * 10 ** (pool.dec1 - pool.dec0)) ** 0.5 * Q96)
    log = {"data": "0x" + word(-10 ** 8) + word(int(34.3 * 10 ** 18)) + word(sqrt_p) + word(5 * 10 ** 18) + word(265640),
           "blockNumber": "0x10", "transactionHash": "0xab", "logIndex": "0x1"}
    d = decode_swap(log, pool)
    assert d["amount0"] == pytest.approx(-1.0) and d["amount1"] == pytest.approx(34.3)
    assert d["price"] == pytest.approx(34.35, rel=1e-9) and d["tick"] == 265640
    # one in-range swap with token1 coming in: fee = a1 * tier * L/(L_pool+L), valued in token0
    t = pd.date_range("2026-01-01", periods=2, freq="min", tz="UTC")
    path = pd.DataFrame({"time": t, "price": [34.35, 34.35], "amount0": [0.0, -1.0], "amount1": [0.0, 34.35],
                         "liquidity": [0.0, 0.0]})
    r = simulate(path, 250, CLParams(swap_haircut=0.0, fee_tier=0.0005, dec0=8, dec1=18), 10.0, 0.0)
    assert r["fee_mode"] == "measured"
    assert r["fees"] == pytest.approx(34.35 * 0.0005 / 34.35)   # L_pool = 0 -> our share is 100%
