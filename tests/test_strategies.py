import numpy as np
import pandas as pd
import pytest

from common import metrics
from strategies.coin_carry_vault.strategy import Ladder, decide, delever_route, price_drop_to_liquidation
from strategies.lending_cluster.strategy import junior_loss_from_depeg, max_deposit_for_util_drop, solve_weights
from strategies.lending_cluster.strategy import Params as LParams
from strategies.leverage_loop.strategy import (ACTIVE, PARKED, UNLEVERED, LoopState, Params, equity_pnl_for_markdown,
                                               inversion_borrow_rate, liquidation_markdown, ltv_bands, Market, net_apy,
                                               next_markdown_state, next_net_state, target_weights)


# ----------------------------------------------------------------------------- metrics

def test_metrics_on_flat_and_linear_curves():
    flat = pd.Series([100.0] * 50)
    assert metrics.max_drawdown(flat) == 0.0
    up = pd.Series(np.linspace(100, 110, 366))
    s = metrics.summary(up, metrics.DAILY)
    assert s["max_drawdown"] == 0.0 and s["total_return"] == pytest.approx(0.10)
    assert s["cagr"] == pytest.approx(0.10, rel=1e-6)


def test_max_drawdown_known_value():
    eq = pd.Series([100, 120, 90, 95, 130])
    assert metrics.max_drawdown(eq) == pytest.approx(-0.25)


# ----------------------------------------------------------------------------- leverage loop

def test_net_apy_can_be_positive_with_negative_raw_spread():
    assert net_apy(8, 0.058, 0.060) == pytest.approx(0.044, abs=1e-9)
    assert inversion_borrow_rate(8, 0.05) == pytest.approx(8 / 7 * 0.05)


def test_liquidation_markdown_matches_docs():
    assert liquidation_markdown(8, 0.915) == pytest.approx(0.04372, abs=1e-4)
    assert liquidation_markdown(4, 0.915) == pytest.approx(0.1803, abs=1e-3)
    pnl, liq = equity_pnl_for_markdown(8, 0.01, 0.915)
    assert pnl == pytest.approx(-0.08) and not liq
    _, liq = equity_pnl_for_markdown(8, 0.05, 0.915)
    assert liq


def test_ltv_bands_reproduce_production_values():
    b = ltv_bands(Market("x", 2, 1.0, 8.0, 0.915), Params())
    assert b["target_ltv"] == pytest.approx(0.875)
    assert b["max_ltv"] == pytest.approx(0.890)
    assert b["emergency_ltv"] == pytest.approx(0.900)


def test_target_weights_respect_caps_and_spill():
    w = target_weights({"a": 10, "b": 10, "c": 10}, {"a": 0.1, "b": 0.2, "c": 1.0}, 0.6)
    assert w["a"] == pytest.approx(0.1) and w["b"] == pytest.approx(0.2) and w["c"] == pytest.approx(0.7)
    assert sum(w.values()) <= 1 + 1e-9
    w = target_weights({"a": 10, "b": 0}, {"a": 0.5, "b": 0.5}, 0.6)
    assert w["b"] == 0 and w["a"] == pytest.approx(0.5)
    assert all(v == 0 for v in target_weights({"a": 0, "b": 0}, {}, 0.6).values())


def test_net_gate_hysteresis():
    p = Params(close_after_samples=3, reenter_samples=2, reenter_margin_bps=50)
    st = LoopState()
    for _ in range(2):
        assert next_net_state(st, -0.01, p) == ACTIVE
    assert next_net_state(st, -0.01, p) == UNLEVERED
    assert next_net_state(st, 0.001, p) == UNLEVERED          # below re-enter margin: stays
    assert next_net_state(st, 0.01, p) == UNLEVERED
    assert next_net_state(st, 0.01, p) == ACTIVE


def test_markdown_close_needs_sustain_and_parks():
    p = Params(markdown_close_bps=150, markdown_sustain_ticks=2)
    st = LoopState()
    next_markdown_state(st, 1.00, p)
    assert next_markdown_state(st, 0.99, p) == ACTIVE          # -1%: inside threshold
    assert next_markdown_state(st, 0.98, p) == ACTIVE          # first breach
    assert next_markdown_state(st, 0.98, p) == PARKED          # sustained


# ----------------------------------------------------------------------------- lending

def test_solver_respects_bounds_and_budget():
    mu = np.array([0.05, 0.04, 0.08])
    cov = np.eye(3) * 1e-5
    w = solve_weights(mu, cov, np.array([3, 3, 6]), np.array([0.4, 0.4, 0.1]), LParams())
    assert w.sum() == pytest.approx(0.9, abs=1e-6)
    assert (w <= np.array([0.4, 0.4, 0.1]) + 1e-9).all() and (w >= -1e-9).all()


def test_util_drop_guard():
    assert max_deposit_for_util_drop(1000, 900, 0.9, 100) == pytest.approx(900 / 0.89 - 1000)
    assert max_deposit_for_util_drop(1000, 5, 0.005, 100) == 0.0


def test_junior_loss_leverage():
    assert junior_loss_from_depeg(0.085, 0.25) == pytest.approx(0.34)
    assert junior_loss_from_depeg(0.5, 0.25) == 1.0




# ----------------------------------------------------------------------------- carry ladder

def test_ladder_decisions():
    L = Ladder()
    assert price_drop_to_liquidation(0.65, 0.86) == pytest.approx(1 - 0.65 / 0.86)
    assert decide(7500, 10000, 0.03, 0.098, 0, L)[0] == "delever_emergency"
    assert decide(7100, 10000, 0.03, 0.098, 0, L) == ("delever", 7100 - 0.65 * 10000)
    assert decide(5000, 10000, 0.03, 0.098, 0, L) == ("relever", 1500)
    assert decide(5000, 10000, 0.06, 0.098, 0, L)[0] == "hold"           # borrow above ceiling
    assert decide(5000, 10000, 0.03, 0.098, 10, L)[0] == "service_withdrawal"
    assert delever_route(1000, 500, 0, 0) == "flash_deleverage"
    assert delever_route(1000, 5000, 1000, 0) == "repay_with_idle"
    assert delever_route(1000, 5000, 100, 2000) == "two_stage_redeem"


# ----------------------------------------------------------------------------- dnENA

def test_dn_ena_sizing_and_margin():
    from strategies.dn_ena.strategy import (Params as EP, margin_health, pt_discount_apy, round_trip_fee,
                                            short_liquidation_px, should_rotate, size_entry, topup_needed)
    p = EP()
    e = size_entry(100_000, 0.14, 0.96, p)
    assert e["reserve"] == pytest.approx(15_000)
    assert e["pt_cost"] + e["margin"] + e["reserve"] == pytest.approx(100_000)        # capital conserved
    assert e["pt_cost"] == pytest.approx(e["qty"] * 0.96 * 0.14)                        # pt_qty == short_qty
    assert pt_discount_apy(0.96, 88) == pytest.approx((1 / 0.96) ** (365 / 88) - 1)
    liq = short_liquidation_px(0.14, e["margin"], e["qty"], p.maint_margin)
    assert margin_health(e["margin"], 0.14, liq, e["qty"], 0.0, p.maint_margin) == pytest.approx(1.0)   # health 1 at liq
    assert topup_needed(e["margin"], 0.14, 0.14, e["qty"], 0.0, p) == 0.0
    need = topup_needed(e["margin"], 0.14, liq * 0.999, e["qty"], 0.0, p)
    assert need > 0
    assert margin_health(e["margin"] + need, 0.14, liq * 0.999, e["qty"], 0.0, p.maint_margin) == pytest.approx(p.topup_health)
    fee = round_trip_fee(1_000, p)
    assert fee == pytest.approx(1_000 * (2 * 0.0008 + 2 * 0.00045))
    assert should_rotate(0.18, 44, 0.19, 88, 1_000, p) is False               # 1% × 44/365 × 1000 = 1.2 < 2.5 fee
    assert should_rotate(0.18, 44, 0.19, 88, 1_000, EP(rotate_rule="single_print")) is True
    assert should_rotate(0.18, 44, 0.30, 88, 1_000, EP(rotate_rule="never")) is False


def test_tranche_risk_rule():
    from strategies.lending_cluster.strategy import depeg_risk_penalty, junior_apy, tranche_eligible
    assert junior_apy(0.055, 0.33, 0.04) == pytest.approx((0.055 - 0.04 * 0.67) / 0.33)
    assert junior_apy(0.013, 0.33, 0.04) < 0                                    # underlying cannot cover the coupon
    assert depeg_risk_penalty(0.01, 0.25) == pytest.approx(0.04)                 # 4x leverage on 1% vol
    assert tranche_eligible(0.085, 0.04, 0.0, 0.33) is True
    assert tranche_eligible(0.085, 0.09, 0.0, 0.33) is False                     # penalty exceeds yield
    assert tranche_eligible(0.085, 0.01, 0.10, 0.33) is False                    # 10% drawdown uses 30% of the cushion
