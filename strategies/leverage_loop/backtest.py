"""Daily replay of the loop cluster allocator on archive Morpho Blue data (3 markets, 93 days).

Run:  python -m strategies.leverage_loop.backtest   (from the portfolio root)
Writes results.json and index.html next to this file.

Data: data/algo_history.csv — daily archive-RPC reads of collateral exchange rate (px),
market supply/borrow and borrow APR for siUSD/USDC, USD3/USDC, stcUSD/USDT, 2026-03-22..06-22.
Modelling choices (stated, not hidden):
  * collateral yield y = annualised trailing 3-day exchange-rate growth (cold start: no lever until known)
  * borrow APR b taken per daily sample; idle USDC earns 0
  * entry fee charged on gross notional on every stcUSD lever (Cap mint 10 bps); USDC loops free
  * net-gate sustain = 2 daily samples (proxy for the hourly 6-sample rule), re-enter after 1 sample > +50 bps
  * markdown close at 1.5% below HWM sustained 2 samples (never fires: no markdown in window)
  * concentration cap 25% of market supply; liquidity cap = supply - borrow
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import metrics
from common.data import morpho_algo_history
from common.report import fmt_pct, fmt_usd, render_page, series, tile, tone_for, write_results
from strategies.leverage_loop.strategy import (
    ACTIVE, PARKED, UNLEVERED, LoopState, Market, Params, inversion_borrow_rate,
    liquidation_markdown, equity_pnl_for_markdown, net_apy, next_markdown_state, next_net_state,
    risk_adjusted_score, should_rebalance, size_lever, target_weights,
)

MARKETS = {
    "stcusd": Market("stcusd", risk_score=2, max_weight=1.0, entry_fee_bps=10, entry_cost_apy=0.01),
    "siusd": Market("siusd", risk_score=3, max_weight=0.20),
    "usd3": Market("usd3", risk_score=5, max_weight=0.10),
}
LIVE_FROM = {"siusd": "2026-03-22", "stcusd": "2026-03-22", "usd3": "2026-06-03"}  # first day the exchange rate moves (market live)
START_CAPITAL = 1_000_000.0
YIELD_LOOKBACK = 3
GAS_PER_TX_USD = 25.0  # mainnet manage tx at ~50 gwei; charged on every lever, trim, de-lever, re-lever


def prepare() -> pd.DataFrame:
    df = morpho_algo_history()
    df["px"] = df["px"].astype(float) / 1e18
    # supply/borrow are in token units with 6 or 18 decimals; utilisation is scale-free
    df["util"] = (df["borrow"].astype(float) / df["supply"].astype(float)).clip(0, 1)
    # dollar depth from the collateral-market side: normalise per market by its decimals
    dec = {"siusd": 6, "usd3": 6, "stcusd": 6}
    df["supply_usd"] = [float(s) / 10 ** dec[m] for s, m in zip(df["supply"], df["market"])]
    df["borrow_usd"] = [float(s) / 10 ** dec[m] for s, m in zip(df["borrow"], df["market"])]
    df["live"] = [d >= pd.Timestamp(LIVE_FROM[m]) for d, m in zip(df["date"], df["market"])]
    px = df.pivot(index="date", columns="market", values="px").sort_index()
    y = (px / px.shift(YIELD_LOOKBACK)) ** (365 / YIELD_LOOKBACK) - 1
    df = df.merge(y.stack().rename("yield").reset_index(), on=["date", "market"], how="left")
    return df.sort_values(["date", "market"]).reset_index(drop=True)


def run(leverage: float, sustain: int = 2, params: Params | None = None, df: pd.DataFrame | None = None) -> dict:
    p = params or Params(close_after_samples=sustain, reenter_samples=1)
    df = prepare() if df is None else df
    mk = {k: Market(k, m.risk_score, m.max_weight, leverage, m.lltv, m.entry_fee_bps, m.entry_cost_apy)
          for k, m in MARKETS.items()}
    st = {k: LoopState() for k in mk}
    idle = START_CAPITAL
    dates = sorted(df["date"].unique())
    rows, actions = [], {"lever": 0, "trim": 0, "delever": 0, "relever": 0, "markdown_close": 0}
    fees_paid = 0.0
    prev_px = {}
    for d in dates:
        day = df[df["date"] == d].set_index("market")
        # 1. accrue: each loop's equity grows by net(L) daily; idle earns 0
        for k, s in st.items():
            if s.equity <= 0 or k not in day.index:
                continue
            r = day.loc[k]
            L = leverage if s.state == ACTIVE else 1.0
            y_real = (r["px"] / prev_px[k] - 1) if k in prev_px else 0.0
            b_daily = r["borrow_apr"] / 365
            s.equity *= 1 + L * y_real - (L - 1) * b_daily
        for k in day.index:
            prev_px[k] = day.loc[k, "px"]
        # 2. gates
        for k, s in st.items():
            if k not in day.index or not day.loc[k, "live"]:
                continue
            r = day.loc[k]
            if np.isnan(r["yield"]):
                continue
            n = net_apy(leverage, r["yield"], r["borrow_apr"])
            before = s.state
            next_markdown_state(s, r["px"], p)
            if s.state == PARKED and before != PARKED:
                actions["markdown_close"] += 1
                idle += s.equity
                s.equity = 0
                continue
            next_net_state(s, n, p)
            if before == ACTIVE and s.state == UNLEVERED:
                actions["delever"] += 1
                idle -= GAS_PER_TX_USD
            elif before != ACTIVE and s.state == ACTIVE and s.equity > 0:
                actions["relever"] += 1
                idle -= GAS_PER_TX_USD
                fees_paid += _entry_fee(mk[k], s.equity)
                s.equity -= _entry_fee(mk[k], s.equity)
        # 3. allocate
        scores, total_equity = {}, idle + sum(s.equity for s in st.values())
        for k, m in mk.items():
            if k not in day.index or not day.loc[k, "live"] or np.isnan(day.loc[k, "yield"]):
                scores[k] = 0.0
                continue
            r = day.loc[k]
            scores[k] = risk_adjusted_score(m, r["yield"], r["borrow_apr"], p) if st[k].state == ACTIVE else 0.0
            if st[k].state != ACTIVE and st[k].equity > 0:
                scores[k] = 0.0
        deployable = total_equity * (1 - p.withdrawal_buffer_bps / 1e4)
        w = target_weights(scores, {k: m.max_weight for k, m in mk.items()}, p.alloc_max_weight)
        for k, m in mk.items():
            s = st[k]
            tgt = deployable * w[k] if s.state == ACTIVE else s.equity  # hold non-active equity where it is
            if s.state != ACTIVE:
                continue
            if w[k] == 0 and s.equity > 0:
                # sub-margin blip: hold the position; only the net-gate (sustained) unwinds
                continue
            if not should_rebalance(s.equity, tgt, p):
                continue
            r = day.loc[k]
            if tgt > s.equity:
                add = min(tgt - s.equity, idle - total_equity * p.withdrawal_buffer_bps / 1e4)
                if leverage > 1:
                    add = min(add, size_lever(add, leverage, r["supply_usd"] - r["borrow_usd"],
                                              r["supply_usd"], (leverage - 1) * s.equity, p))
                if add <= p.min_rebalance_usd:
                    continue
                fee = _entry_fee(m, add)
                fees_paid += fee
                s.equity += add - fee
                idle -= add
                actions["lever"] += 1
                idle -= GAS_PER_TX_USD
            else:
                cut = s.equity - tgt
                s.equity -= cut
                idle += cut
                actions["trim"] += 1
                idle -= GAS_PER_TX_USD
        total = idle + sum(s.equity for s in st.values())
        rows.append({"date": d, "portfolio": total, "idle": idle,
                     **{f"eq_{k}": s.equity for k, s in st.items()},
                     **{f"state_{k}": s.state for k, s in st.items()},
                     **{f"net_{k}": net_apy(leverage, day.loc[k, "yield"], day.loc[k, "borrow_apr"])
                        if k in day.index and not np.isnan(day.loc[k, "yield"]) else np.nan for k in mk}})
    out = pd.DataFrame(rows).set_index("date")
    eq = out["portfolio"]
    m = metrics.summary(eq, metrics.DAILY)
    return {"curve": out, "metrics": m, "actions": actions, "fees": fees_paid, "final": float(eq.iloc[-1]),
            "deployed_pct": float(1 - out["idle"].iloc[-1] / eq.iloc[-1])}


def _entry_fee(m: Market, equity: float) -> float:
    return equity * m.target_leverage * m.entry_fee_bps / 1e4


def build_results() -> dict:
    df = prepare()
    runs = {L: run(L, df=df) for L in (8.0, 6.0, 3.0, 1.0)}
    hair = run(8.0, sustain=1, df=df)
    scenarios = {}
    runs_1x = runs[1.0]["metrics"]["total_return"]
    for L, r in runs.items():
        m, a = r["metrics"], r["actions"]
        c = r["curve"]
        name = f"{int(L)}x"
        tiles = [
            tile("Window return (92d)", fmt_pct(m["total_return"]), f"final {fmt_usd(r['final'])} · benchmark: " + (f"1x unlevered hold {fmt_pct(runs_1x)}" if L > 1 else "none stated (1x is the benchmark for the levered runs)"),
                 tone_for(m["total_return"], 92, benchmark=(runs_1x if L > 1 else None))),
            tile("Annualised (simple)", fmt_pct(m["total_return"] * 365 / 92), "92-day window"),
            tile("Max drawdown", fmt_pct(m["max_drawdown"]), "on portfolio value", "bad" if m["max_drawdown"] < -0.01 else "neutral"),
            tile("Entry fees paid", fmt_usd(r["fees"]), f"{r['fees']/START_CAPITAL*1e4:.0f} bps of capital · Cap mint 10 bps × {int(L)}x"),
            tile("Net-gate de-levers", str(a["delever"]), f"{a['relever']} re-levers · {a['markdown_close']} markdown closes"),
            tile("Levers / trims", f"{a['lever']} / {a['trim']}", "10% weight deadband"),
            tile("Deployed at end", f"{r['deployed_pct']*100:.0f}%", f"idle {fmt_usd(c['idle'].iloc[-1])} · 2% buffer"),
        ]
        last = c.iloc[-1]
        alloc_md = "| loop | equity | state | levered net (last) |\n|---|---|---|---|\n" + "\n".join(
            f"| {k} | {fmt_usd(last[f'eq_{k}'])} | {last[f'state_{k}']} | {fmt_pct(last[f'net_{k}'])} |" for k in MARKETS)
        scenarios[name] = {
            "label": name + (" (production)" if L == 8 else " hold" if L == 1 else ""),
            "note": f"2026-03-22 → 2026-06-22, $1.0M start, 3 Morpho Blue stablecoin loops, target leverage {int(L)}x",
            "verdict": (f"{'POSITIVE' if m['total_return'] > 0.0005 else 'NEGATIVE' if m['total_return'] < -0.0005 else 'FLAT'} carry after fees — "
                        f"{fmt_pct(m['total_return'])} over 92 days, max DD {fmt_pct(m['max_drawdown'])}, "
                        f"{a['delever']} forced de-levers"),
            "verdict_tone": tone_for(m["total_return"], 92, benchmark=(runs_1x if L > 1 else None)),
            "tiles": tiles,
            "series": [
                series("Portfolio value ($)", c.index.date, c["portfolio"], unit="$"),
                series("Equity by loop ($)", c.index.date, {k: c[f"eq_{k}"] for k in MARKETS}, unit="$"),
                series(f"Levered net APY at {int(L)}x (gate metric)", c.index.date,
                       {k: c[f"net_{k}"] for k in MARKETS}, unit="%"),
                series("Morpho borrow APR by market", c.index.date,
                       {k: df[df.market == k].set_index("date")["borrow_apr"].reindex(c.index) for k in MARKETS}, unit="%"),
                series("Idle USDC ($)", c.index.date, c["idle"], unit="$"),
            ],
            "tables": [{"title": "Final allocation", "md": alloc_md}],
            "notes": [
                "Idle USDC modelled at 0%; collateral yield = trailing 3-day exchange-rate growth; borrow APR per daily sample; $25 gas per manage tx (lever, trim, de-lever, re-lever); Cap mint fee on stcUSD levers only — USDC-loop wraps and all unwinds are fee-free, as on-chain.",
                "Net-gate sustain = 2 daily samples (proxy for 6 hourly samples); re-enter after one sample above +50 bps.",
                "No collateral markdown occurred in the window, so the PARKED path is exercised only in the parametric tables below.",
                "Accrual uses the realised daily exchange-rate move at L× (15 small down-days in stcUSD cost 8× each), not a smoothed yield; the internal Lucidly sim on smoothed yields showed +0.30% at 8x and +1.32% at 1x over the same window — same ordering, more flattering level.",
                "Market go-live dates respected: siUSD and stcUSD 2026-03-22, USD3 2026-06-03 (first day its exchange rate moves) — no pre-live rates are used.",
            ],
        }
    sweep = "| leverage | 92d return | annualised | max DD | entry fees | de-levers |\n|---|---|---|---|---|---|\n" + "\n".join(
        f"| {int(L)}x | {fmt_pct(r['metrics']['total_return'])} | {fmt_pct(r['metrics']['total_return'] * 365 / 92)} | "
        f"{fmt_pct(r['metrics']['max_drawdown'])} | {fmt_usd(r['fees'])} | {r['actions']['delever']} |"
        for L, r in runs.items())
    sweep += (f"\n\nNet-gate sensitivity at 8x: sustain 1 sample (hair-trigger) → {fmt_pct(hair['metrics']['total_return'])}, "
              f"{hair['actions']['delever']} de-levers, {fmt_usd(hair['fees'])} fees; sustain 2 → "
              f"{fmt_pct(runs[8.0]['metrics']['total_return'])}, {runs[8.0]['actions']['delever']} de-levers.")
    md_tbl = "| leverage | operating LTV | same-block markdown that liquidates (LLTV 0.915) |\n|---|---|---|\n" + "\n".join(
        f"| {L}x | {(L-1)/L:.4f} | {liquidation_markdown(L, 0.915)*100:.2f}% |" for L in (4, 6, 8, 10, 11))
    pnl_tbl = "| markdown | equity P&L at 8x | liquidated |\n|---|---|---|\n" + "\n".join(
        f"| {m*100:.2f}% | {'near-total loss' if liq else fmt_pct(pnl)} | {'yes' if liq else 'no'} |"
        for m, (pnl, liq) in ((m, equity_pnl_for_markdown(8, m, 0.915)) for m in (0.005, 0.01, 0.02, 0.0437, 0.05, 0.08)))
    inv_tbl = "| collateral yield | borrow APR that zeroes net at 8x |\n|---|---|\n" + "\n".join(
        f"| {y*100:.0f}% | {inversion_borrow_rate(8, y)*100:.2f}% |" for y in (0.03, 0.04, 0.05, 0.06, 0.07))
    sections = [
        {"heading": "What this is", "body_md": (
            "A USDC vault running a portfolio of leveraged stablecoin carry loops on Morpho Blue (Ethereum mainnet). "
            "Each loop wraps USDC into a yield-bearing stable (stcUSD, siUSD, USD3), supplies it as collateral, borrows the loan token "
            "(USDT for stcUSD via an oracle-floored DEX leg inside the flashloan, USDC otherwise) and re-levers to a fixed per-market "
            "target in one atomic Morpho flashloan. I owned the control design: allocation, leverage policy, the de-risking state machine, "
            "the sizing caps and the backtests. Live from June 2026 on mainnet with three markets.\n\n"
            "```\nnet(L) = L*y - (L-1)*b = y + (L-1)*(y - b)      LTV = (L-1)/L\n```\n"
            "The gate metric is the levered net at target leverage, never the raw spread: a loop is still net-positive with y < b while L*y > (L-1)*b.")},
        {"heading": "Allocation and sizing", "body_md": (
            "- `score_i = net_i(L_target) / risk_i^beta` minus an entry-cost drag (1%/yr for stcUSD); a loop below the 25 bps lever margin scores 0.\n"
            "- Weights proportional to score, clamped to hard caps (stcUSD 1.0 / siUSD 0.20 / USD3 0.10), freed mass renormalised onto the others (spill-on-cap).\n"
            "- `cap_borrow = min(target_borrow, available_liquidity, 25% of market supply − our debt, $1M per tick)`; a capped leg shrinks and the idle waits (wait-not-spill).\n"
            "- 2% idle withdrawal buffer; an equity move fires only beyond max($50, 10% of target); leverage adjusts only beyond 5% drift of target.\n"
            "- Fixed per-market leverage, not an optimiser: a grid search over L returns the max corner in any positive-spread regime at ~29 RPC reads per loop per tick; the cases a dial would handle are covered by SAFETY, the net-gate and the markdown close.")},
        {"heading": "Risk framework", "body_md": (
            "Three states, two decoupled triggers at different speeds:\n\n"
            "| transition | trigger | threshold | cadence |\n|---|---|---|---|\n"
            "| ACTIVE → UNLEVERED (1x, hold collateral) | levered net < 0, sustained | 6 consecutive hourly samples | hourly |\n"
            "| UNLEVERED → ACTIVE | net > +50 bps for 3 samples | — | hourly |\n"
            "| any → PARKED (exit to USDC) | exchange rate < HWM − 1.5% | 2 consecutive 5-min ticks | 5 min |\n"
            "| SAFETY de-lever | LTV > max_ltv (0.890 at 8x) | back to 0.875; above 0.900 bypass gas budget | 5 min |\n\n"
            "A rate problem (negative spread) means de-lever and keep the yield asset; a principal problem (markdown) means exit regardless of carry.\n\n"
            "**Same-block markdown is the real liquidation control.** Morpho marks ERC4626 collateral at `convertToAssets`; a markdown m lifts LTV to LTV0/(1−m), "
            "and no off-chain reaction beats a same-block move: `m* = 1 − LTV0/LLTV`.\n\n" + md_tbl + "\n\n" + pnl_tbl +
            "\n\nUnprotected single-step band at the 1.5% close threshold: ~1.5%..4.37%. \"High leverage is only safe on collateral that cannot gap\" — the junior cushion under each senior-tranche collateral, not reaction speed, bounds this.\n\n"
            "**Borrow-rate inversion.** Net zeroes when `b > L/(L−1) · y` (1.143·y at 8x). siUSD/USDC borrow printed 21.65% at 100% utilisation on 2026-06-20 — the live inversion risk, and why siUSD is hard-capped at 0.20.\n\n" + inv_tbl)},
        {"heading": "Leverage sweep and net-gate sensitivity", "body_md": sweep + (
            "\n\nIn this window the in-sample spread (y − b) was only ~+0.3%: each turn of leverage added ~0.3% of gross carry but induced more than that in "
            "net-gate churn (~70 bps per stcUSD de-lever/re-lever round-trip, Cap mint fee × leverage) plus tail risk, so realised return fell monotonically "
            "with leverage while drawdown rose. The conclusion I took into production: size leverage to the live spread, and make the net-gate cost-aware "
            "(longer sustain for entry-fee markets).")},
        {"heading": "Adversarial review — top findings", "body_md": (
            "- **Same-block markdown → liquidation (critical, accepted residual at 8x):** mitigated by the reactive 1.5% close and senior-tranche collateral; tightening options were lower per-market leverage (~4.5x), per-loop equity caps, or an oracle-push listener.\n"
            "- **Net-gate close was dead:** `spot_beats_idle = yield > 0` was always true for a yield-bearing stable, so the close branch could never fire. Rewritten as a sustained-negative-levered-net state machine.\n"
            "- **Allocator churn re-paying the mint fee:** a $50 flat floor on a multi-100k book trimmed and re-levered every 15 min; fixed with a 10% weight deadband and an entry-cost gate (defer above 150 bps, never auto-enter above 400 bps).\n"
            "- **Strategist EOA out of gas → trading freeze (2026-06-22):** ~0.00021 ETH left, every rebalance failed for >1h; a SAFETY de-lever would also have failed. Added low-gas paging to the health job and preflight.\n"
            "- **Decoder missing a selector:** a leaf resolving in the merkle validator does not prove the deployed decoder supports it (`0xde42fa1c`); the stcUSD Cap path hit this. Rule: always run the anvil mainnet-fork E2E before trusting a new leaf. Three root regenerations came from this class of wiring bug.\n"
            "- **Depeg-stuck exit on the USDT bridge:** beyond the 30 bps Chainlink floor the swap leg defers, so the largest book can hold but not adjust; accepted, residue swept and counted in NAV.\n"
            "- **Oracle staleness unchecked** on markdown/swap-floor reads — flagged, listed to tighten.")},
        {"heading": "Operations", "body_md": (
            "| job | cadence | does |\n|---|---|---|\n"
            "| snapshot | 5 min | per-loop collateral/debt/LTV, IRM borrow rate, oracle; persist samples; SAFETY; markdown close |\n"
            "| rebalance | 15 min | allocate + size + lever/unwind/adjust; dust sweep |\n"
            "| net_gate | 60 min | sustained-negative de-lever / re-arm |\n"
            "| withdraw | 5 min | free idle for pending redemptions, worst loop first (indexer-driven, fail-closed) |\n"
            "| health | 15 min | deposit-halt on share-price drop vs HWM, cron liveness watchdog, LTV page, low-gas warning, Discord |\n"
            "| loan_token_sweep | hourly | swap USDT residue > $50 back to USDC via oracle-floored DEX leg |\n\n"
            "Read-only preflight validator (66 checks: every leaf against the live root, adapter binding, decoder constant, oracles, leverage feasibility vs LLTV, strategist role and gas), dry-run CLI, emergency exit, anvil fork E2E (lever/adjust/unwind, rebalance, SAFETY, net-gate, withdraw, health, concentration cap), Postgres state (snapshot, net history/state, actions, health, heartbeats), 26-panel Grafana dashboard, all-or-nothing env boot.")},
        {"heading": "Reproduce", "body_md": (
            "```\npython -m strategies.leverage_loop.backtest\n```\n"
            "`strategy.py` holds the pure control logic (net APY, allocator with spill-on-cap, three-state gate, LTV bands, markdown and inversion maths, sizing caps). "
            "`backtest.py` replays it day by day on `data/algo_history.csv` (archive-RPC reads) and writes `results.json`, which this page renders.")},
    ]
    return {
        "title": "Leveraged Stablecoin Loop Cluster — Morpho Blue",
        "subtitle": "Clean-room reimplementation and replay of the control layer · Ethereum mainnet · live June 2026",
        "scenarios": scenarios,
        "default_scenario": "8x",
        "sections": sections,
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    res = build_results()
    write_results(here, res)
    render_page(here, res)
    for n, s in res["scenarios"].items():
        print(n, s["verdict"])
