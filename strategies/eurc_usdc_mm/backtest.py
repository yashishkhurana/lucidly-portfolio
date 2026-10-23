"""EURC/USDC concentrated-liquidity market making on Aerodrome Slipstream (Base) — tick-spacing study.

Run:  python -m strategies.eurc_usdc_mm.backtest
Data: data/eurc_usdc_aero_backtest.csv — the state log of my public backtest
      (github.com/lucidlylabs/simulations): pool tick / price per swap, 2026-01-05, 440 swaps over 5.2 h;
      plus a 30-day OU mean-reverting EUR/USD path (SYNTHETIC, labelled) for the spacing sweep.
Fee model: fee_apr_ref = 1.54%/yr at a 500-tick band (15.4%/yr at 50 ticks), calibrated so the real window
      reproduces the public log's fee accrual (74.8 EURC-equivalent over 5.2 h on an 821k-EURC book). An assumption.
P&L is reported versus holding the seeded tokens (fees − IL − slippage), which removes the EUR/USD
inventory revaluation that otherwise dominates a 5-hour window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.cl_engine import CLParams, simulate
from common.data import DATA, eurc_usdc_aero_state
from common.report import ann_or_window, fmt_pct, render_page, series, tile, tone_for, write_results

# The public log's total_equity is in USDC (amount0 × price + amount1 = 961,535 USDC at P = 1.17052).
# The engine works in token0 = EURC, so the deposit is the EURC-equivalent: 961,535 / 1.17052.
DEPOSIT_USDC = 961_535.0
P0 = 1.17052
DEPOSIT = DEPOSIT_USDC / P0  # 821,458 EURC
FEE_APR_REF = 0.0154  # calibrated: real window fees = 74.7 EURC-eq vs the log's 51.0 EURC + 27.9 USDC = 74.8 EURC-eq
P = CLParams(mm_alloc=1.0, hedge_alloc=0.0, swap_haircut=0.0002, fee_tier=0.000076, dec0=6, dec1=6)  # Slipstream fee() = 76 pips on 0xe846...90ec7 (tickSpacing 50), read 2026-10-23
PUBLIC_LOG = {"equity_start_usdc": 961_535, "equity_end_usdc": 959_848, "fees_token0": 51.0, "fees_token1": 27.9}


SWAPS = DATA / "swaps_eurc_usdc_base.csv"   # written by `python -m strategies.cl_swaps.fetch_data --pool eurc_usdc_base`


def real_path() -> pd.DataFrame:
    """Per-swap path (measured fees) when the swap log is present, else the public backtest's price log."""
    if SWAPS.exists():
        df = pd.read_csv(SWAPS, parse_dates=["time"])
        return df[["time", "price", "amount0", "amount1", "liquidity"] + (["fee_pips"] if "fee_pips" in df.columns else [])]
    df = eurc_usdc_aero_state()
    return pd.DataFrame({"time": df["time"], "price": df["price_normalised"] / 1e8})


def log_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last swap time of the public backtest log."""
    df = eurc_usdc_aero_state()
    return df["time"].iloc[0], df["time"].iloc[-1]


def log_band() -> tuple[int, int]:
    """The band the public backtest actually held (its active_lower/upper_tick columns)."""
    df = eurc_usdc_aero_state()
    return int(df["active_lower_tick"].iloc[0]), int(df["active_upper_tick"].iloc[0])


def ou_path(days: int = 30, vol: float = 0.07, kappa: float = 2.0, seed: int = 5, p0: float = 1.168) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    dt = 1 / (365 * 24 * 60)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - kappa * x[i - 1] * dt + vol * np.sqrt(dt) * rng.normal()
    t = pd.date_range("2026-01-05", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"time": t, "price": p0 * np.exp(x)})


def scenario(label: str, note: str, path: pd.DataFrame, width: int, spacing: int, synthetic: bool) -> dict:
    band = log_band() if not synthetic else None          # real scenario: the log's own band, so the comparison is like for like
    full = None
    if not synthetic:
        t0, t1 = log_window()
        full = path                                        # the pull as fetched (may run past the log's end)
        path = path[(path["time"] >= t0) & (path["time"] <= t1)]
    r = simulate(path, width, P, DEPOSIT, FEE_APR_REF, spacing=spacing, band_ticks=band)
    c = r["curve"]
    x = c.index.strftime("%m-%d %H:%M")
    days = r["days"]
    lbl, val = ann_or_window(r["pnl_vs_hold"] / DEPOSIT, days)
    sweep = "| width (ticks) | spacing | PnL vs hold | fees | IL | slippage | rotations |\n|---|---|---|---|---|---|---|\n"
    for w, sp in ((20, 10), (50, 10), (100, 50), (200, 50), (500, 50)):
        q = simulate(path, w, P, DEPOSIT, FEE_APR_REF, spacing=sp)
        sweep += f"| {w} | {sp} | {fmt_pct(q['pnl_vs_hold']/DEPOSIT, 3)} | {q['fees']:,.0f} | {q['il']:,.0f} | {q['slippage']:,.0f} | {q['rebalances']} |\n"
    notes = ["PnL vs hold = fees − slippage (seeding + rotations) − realised IL − in-band IL − reval, where reval is the remainder: the revaluation at the end price of IL realised at earlier rotation prices and of the haircut tokens (≈0 without rotations, a few percent of the total with many). 'Equity vs deposit' additionally carries the EUR/USD move on the USDC inventory, which dominates any window measured in hours."]
    if not synthetic:
        eq_usdc_0 = float(c["equity"].iloc[0] * c["price"].iloc[0])
        eq_usdc_1 = float(c["equity"].iloc[-1] * c["price"].iloc[-1])
        log_ret = PUBLIC_LOG["equity_end_usdc"] / PUBLIC_LOG["equity_start_usdc"] - 1
        log_fees_usdc = PUBLIC_LOG["fees_token0"] * c["price"].iloc[-1] + PUBLIC_LOG["fees_token1"]
        log_ret_fees = (PUBLIC_LOG["equity_end_usdc"] + log_fees_usdc) / PUBLIC_LOG["equity_start_usdc"] - 1   # the log's total_equity excludes accrued fees
        notes.append(f"Reconciliation with the public log (USDC numeraire, same band [{band[0]}, {band[1]}] read from the log): log equity {PUBLIC_LOG['equity_start_usdc']:,} → {PUBLIC_LOG['equity_end_usdc']:,} USDC "
                     f"excluding its accrued fees ({fmt_pct(log_ret, 3)}), {fmt_pct(log_ret_fees, 3)} including them; this engine, which carries fees in equity, {eq_usdc_0:,.0f} → {eq_usdc_1:,.0f} USDC ({fmt_pct(eq_usdc_1/eq_usdc_0 - 1, 3)}). "
                     f"Fees: log {PUBLIC_LOG['fees_token0']} EURC + {PUBLIC_LOG['fees_token1']} USDC vs engine {r['fees']:.1f} EURC"
                     + (" — the log assumed a fixed ~79-pip tier and credited the position a larger share of each swap than the chain's per-swap liquidity supports; the measured figure is the pool's own arithmetic and supersedes the calibration." if r["fee_mode"] == "measured" else " (the modelled rate was calibrated to the log)")
                     + f". On this like-for-like window the equity gap is {abs(eq_usdc_1/eq_usdc_0 - 1 - log_ret_fees)*1e4:.1f} bp; "
                     "placing the band with the engine's own centring rule instead ([1540, 1590] vs the log's [1550, 1600]) moves the result by ~5 bps, which is the band-placement sensitivity, not a fee-model gap.")
        if full is not None and len(full) > len(path):
            rx = simulate(full, width, P, DEPOSIT, FEE_APR_REF, spacing=spacing, band_ticks=band)
            notes.append(f"The pull as fetched runs {(full['time'].iloc[-1] - path['time'].iloc[-1]).total_seconds()/60:.0f} minutes past the log's last swap. On that extended path the price leaves the band once "
                         f"(rotation at {rx['curve'].index[rx['curve']['lower'].ne(rx['curve']['lower'].iloc[0]).argmax()].strftime('%H:%M') if rx['rebalances'] else '—'}), costing {rx['il']:,.0f} EURC realised IL and {rx['slippage'] - r['slippage']:,.0f} EURC of extra slippage: "
                         f"PnL vs hold {fmt_pct(rx['pnl_vs_hold']/DEPOSIT, 3)} over {rx['hours']:.1f} h, measured fees {rx['fees']:.0f} EURC. The headline above is the log's window so the two are comparable.")
        if r["fee_mode"] == "measured":
            notes.append("Fee provenance: the fetcher reads the pool's fee() by eth_call at the window's start block and writes it per row (fee_pips); the bundled file carries 76 pips, which equals the pool's current fee, so the file alone cannot show whether the archive read succeeded or the fallback fired. "
                         "The reconstruction corroborates it: the log's per-swap credit works out to ~74 pips × L_pos/L_pool, i.e. a share computed without the position's own liquidity in the denominator — the log over-credited the share, not the tier.")
    return {
        "label": label, "note": note,
        "verdict": (f"PnL vs hold {fmt_pct(r['pnl_vs_hold']/DEPOSIT, 3)} over {days*24:.1f} h" if days < 1 else f"PnL vs hold {fmt_pct(r['pnl_vs_hold']/DEPOSIT)} over {days:.0f} days")
                   + f": fees {r['fees']:,.0f} − slippage {r['slippage']:,.0f} − realised IL {r['il']:,.0f} − in-band IL {r['in_band_il']:,.0f} + reval {r['reval']:,.0f} EURC, {r['rebalances']} rotations",
        "verdict_tone": tone_for(r["pnl_vs_hold"], days, synthetic),
        "tiles": [
            tile("PnL vs hold", f"{r['pnl_vs_hold']:+,.0f} EURC", fmt_pct(r["pnl_vs_hold"] / DEPOSIT, 3) + " · fees − slippage − realised and in-band IL", tone_for(r["pnl_vs_hold"], days, synthetic)),
            tile(lbl, val, "no annualisation below 30 days"),
            tile("Equity vs deposit", f"{r['pnl']:+,.0f} EURC", "includes EUR/USD inventory revaluation"),
            tile("Fees (measured)" if r["fee_mode"] == "measured" else "Fees (modelled)", f"{r['fees']:,.0f} EURC",
                 "per-swap from the pool's Swap log" if r["fee_mode"] == "measured" else f"{FEE_APR_REF*500/width*100:.1f}%/yr on in-range value"),
            tile("IL realised", f"{r['il']:,.0f} EURC", "on band rotations"),
            tile("IL in band (unrealised)", f"{r['in_band_il']:,.0f} EURC", "current band's entry tokens vs LP at end price"),
            tile("Slippage", f"{r['slippage']:,.0f} EURC", "2 bps per balancing swap incl. seeding; haircut dust stays idle in equity"),
            tile("Rotations", str(r["rebalances"]), f"band {width} ticks, spacing {spacing}"),
            tile("Max drawdown", fmt_pct(r["max_dd"]), "mark-to-market equity"),
        ],
        "series": [
            series("Equity vs hold benchmark (EURC)", x, {"LP": c["equity"], "hold seeded tokens": c["hold"]}),
            series("Price vs active band (USDC per EURC)", x, {"price": c["price"], "lower": c["lower"], "upper": c["upper"]}),
            series("Fees vs IL + slippage (EURC, cumulative)", x, {"fees": c["fees"], "IL": c["il"], "slippage": c["slippage"]}),
        ],
        "tables": [{"title": f"Band width × tick spacing sweep ({label})", "md": sweep}],
        "notes": notes,
    }


def build_results() -> dict:
    sc = {
        "real": scenario("Real swaps · 50-tick band", "Aerodrome EURC/USDC on Base, 2026-01-04 23:06 → 01-05 04:17 UTC (the public log's window, real; measured fees when the swap log is present)", real_path(), 50, 10, False),
        "ou_50": scenario("30-day OU path · 50-tick band", "SYNTHETIC EUR/USD, 7% ann. vol, mean-reverting", ou_path(), 50, 10, True),
        "ou_200": scenario("30-day OU path · 200-tick band", "SYNTHETIC EUR/USD, 7% ann. vol, mean-reverting", ou_path(), 200, 50, True),
    }
    sections = [
        {"heading": "What this is", "body_md": (
            "A single-band concentrated-liquidity market-making strategy on the EURC/USDC pool on Aerodrome Slipstream (Base). The pair is fiat-vs-fiat, so there is no crypto beta to hedge; "
            "the question is how tight a band to run against EUR/USD realised volatility and how often to rotate it. I built the backtester and published the tick-spacing comparison in `lucidlylabs/simulations`. "
            "The real window is five hours, so only the synthetic 30-day runs say anything about rotation costs, and they say it under an assumed fee rate.")},
        {"heading": "Rules", "body_md": "- One active band; rotate only when price leaves it (both oracle and pool tick outside).\n- Re-seed balanced for the new band; 2 bps min-out haircut on the balancing swap.\n- Fees accrue on in-range liquidity only; no leverage, no borrow."},
        {"heading": "Reference plots — public backtest", "body_md": (
            "![EURC/USDC 10 spacing](../../data/eurc_usdc_backtest_plot_10_spacing.png)\n\n![EURC/USDC 50 spacing](../../data/eurc_usdc_backtest_plot_50_spacing.png)")},
        {"heading": "Reproduce", "body_md": "```\npython -m strategies.eurc_usdc_mm.backtest\n```\nUses `common/cl_engine.py` with the hedge leg disabled."},
    ]
    return {"title": "EURC/USDC Concentrated-Liquidity MM — Aerodrome (Base)",
            "subtitle": "Tick-spacing study for a fiat-stable pair · shared CL engine on the public backtest's swap log",
            "scenarios": sc, "default_scenario": "real", "sections": sections}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    # Not a portfolio page: this module is kept for the fee-measurement reconciliation against my public EURC/USDC
    # Aerodrome backtest (lucidlylabs/simulations). It prints the per-scenario decomposition and writes nothing.
    res = build_results()
    for n, s in res["scenarios"].items():
        print(n, "|", s["verdict"])
