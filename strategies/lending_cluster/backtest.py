"""Daily replay of the lending-cluster allocator on archive Morpho Blue data, plus the
junior-tranche depeg stress from the June 2026 postmortem.

Run:  python -m strategies.lending_cluster.backtest
Data: data/algo_history.csv (93 days, 3 Morpho markets): supply APY = borrow APR × utilisation.
      The three markets stand in for the production lineup (AA_FalconX/USDC, mF-ONE/USDC,
      srRoyUSDC/USDC, fxSAVE/USDC) whose indexer history is not public; caps, hurdles and
      risk scores below mirror the production config scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import metrics
from common.data import morpho_algo_history
from common.report import fmt_pct, fmt_usd, render_page, series, tile, tone_for, write_results
from strategies.lending_cluster.strategy import (
    Market, MarketState, Params, capacity, junior_loss_from_depeg, max_deposit_for_util_drop, must_exit,
    plan, solve_weights, supply_apy, update_gates,
)

MARKETS = {
    "siusd": Market("siusd", risk_score=3, max_weight=0.40, min_weight=0.05, hurdle=0.04, concentration_cap=0.20, abs_cap=10e6, min_deploy=10e3, max_deploy=100e3),
    "usd3": Market("usd3", risk_score=4, max_weight=0.40, min_weight=0.05, hurdle=0.03, concentration_cap=0.20, abs_cap=5e6, min_deploy=10e3, max_deploy=100e3),
    "stcusd": Market("stcusd", risk_score=6, max_weight=0.40, min_weight=0.01, hurdle=0.04, concentration_cap=0.15, abs_cap=0.5e6, min_deploy=1e3, max_deploy=50e3),
}
CAPITAL = 1_000_000.0
LOOKBACK = 3  # days (prod: 6h window at 6h cadence)


def prepare() -> pd.DataFrame:
    df = morpho_algo_history()
    dec = 6
    df["supply_usd"] = df["supply"].astype(float) / 10 ** dec
    df["borrow_usd"] = df["borrow"].astype(float) / 10 ** dec
    df["util"] = (df["borrow_usd"] / df["supply_usd"]).clip(0, 1)
    df["apy"] = [supply_apy(b, u) for b, u in zip(df["borrow_apr"], df["util"])]
    return df


def run(p: Params, idle_yield: float = 0.0, max_deploy_scale: float = 1.0) -> dict:
    df = prepare()
    apy = df.pivot(index="date", columns="market", values="apy").sort_index()
    util = df.pivot(index="date", columns="market", values="util").sort_index()
    sup = df.pivot(index="date", columns="market", values="supply_usd").sort_index()
    bor = df.pivot(index="date", columns="market", values="borrow_usd").sort_index()
    keys = list(MARKETS)
    pos = {k: 0.0 for k in keys}
    st = {k: MarketState() for k in keys}
    idle = CAPITAL
    rows, n_actions = [], 0
    for i, d in enumerate(apy.index):
        # accrue yesterday's APY
        for k in keys:
            pos[k] *= 1 + apy[k].iloc[i] / 365
        idle *= 1 + idle_yield / 365
        port = idle + sum(pos.values())
        if i < LOOKBACK:
            rows.append({"date": d, "portfolio": port, "idle": idle, **{f"pos_{k}": pos[k] for k in keys}, **{f"apy_{k}": apy[k].iloc[i] for k in keys}})
            continue
        win = apy.iloc[i - LOOKBACK:i + 1]
        mu, cov = win.mean().values, win.cov().values
        upper = np.zeros(len(keys))
        for j, k in enumerate(keys):
            m = MARKETS[k]
            eligible = update_gates(st[k], m, float(mu[j]), pos[k] > 0, p)
            if util[k].iloc[i] > m.utilization_cap or not eligible:
                upper[j] = 0.0
                continue
            cap = capacity(m, bor[k].iloc[i])
            upper[j] = min(m.max_weight, cap / port)
        w = solve_weights(mu, cov, np.array([MARKETS[k].risk_score for k in keys]), upper, p)
        target = {}
        for j, k in enumerate(keys):
            m = MARKETS[k]
            t = min(w[j] * port, capacity(m, bor[k].iloc[i]))
            if w[j] < m.min_weight:
                t = 0.0
            if must_exit(st[k], p):
                t = 0.0
            delta = t - pos[k]
            if delta > 0:
                delta = min(delta, max_deposit_for_util_drop(sup[k].iloc[i], bor[k].iloc[i], util[k].iloc[i], p.max_deposit_util_drop_bps),
                            m.max_deploy * max_deploy_scale)
                if delta < m.min_deploy:
                    delta = 0.0
            target[k] = pos[k] + delta
        acts = plan(pos, target, idle, {k: sup[k].iloc[i] - bor[k].iloc[i] for k in keys}, p)
        for a in acts:
            n_actions += 1
            if a["kind"] == "supply":
                pos[a["market"]] += a["assets"]
                idle -= a["assets"]
            else:
                pos[a["market"]] -= a["assets"]
                idle += a["assets"]
        rows.append({"date": d, "portfolio": idle + sum(pos.values()), "idle": idle,
                     **{f"pos_{k}": pos[k] for k in keys}, **{f"apy_{k}": apy[k].iloc[i] for k in keys}})
    c = pd.DataFrame(rows).set_index("date")
    return {"curve": c, "metrics": metrics.summary(c["portfolio"], metrics.DAILY), "actions": n_actions, "apy": apy}


def build_results() -> dict:
    base = run(Params())
    fast = run(Params(), max_deploy_scale=10.0)
    scenarios = {}
    for name, r, label, note in (
        ("prod", base, "Production caps", "per-tick deploy caps $100k/$100k/$50k, 3-day APY window, 1% drift gate"),
        ("fast", fast, "10× deploy caps", "same rules, per-tick deploy caps ×10 — the caps are not what limits deployment"),
    ):
        c, m = r["curve"], r["metrics"]
        last = c.iloc[-1]
        keys = list(MARKETS)
        idle_bench = None   # no offline benchmark series (Aave v3 / Morpho vault USDC supply) is bundled, so never 'good'
        avg_dep = float(1 - (c["idle"] / c["portfolio"]).mean())
        aw = sum(c[f"pos_{k}"].iloc[-1] * c[f"apy_{k}"].iloc[-1] for k in keys) / max(1, c["portfolio"].iloc[-1] - c["idle"].iloc[-1])
        scenarios[name] = {
            "label": label, "note": note,
            "verdict": f"{fmt_pct(m['total_return'])} over 93 days ({fmt_pct(m['total_return'] * 365 / 93)} annualised, simple), max DD {fmt_pct(m['max_drawdown'])}, {r['actions']} supply/withdraw actions",
            "verdict_tone": tone_for(m["total_return"], 93, benchmark=idle_bench),
            "tiles": [
                tile("Window return", fmt_pct(m["total_return"]), f"final {fmt_usd(c['portfolio'].iloc[-1])} · no benchmark series bundled, so not marked good", tone_for(m["total_return"], 93, benchmark=idle_bench)),
                tile("Annualised (simple)", fmt_pct(m["total_return"] * 365 / 93), "93-day window"),
                tile("Max drawdown", fmt_pct(m["max_drawdown"]), "supply-only: the loss path (bad debt, oracle) did not occur in-sample"),
                tile("Average deployed", f"{avg_dep*100:.0f}%", "of capital over the window — the real finding"),
                tile("Deployed at end", f"{(1 - last['idle']/last['portfolio'])*100:.0f}%", f"idle {fmt_usd(last['idle'])} (0% yield)"),
                tile("Weighted APY at end", fmt_pct(aw), "on deployed capital"),
                tile("Actions", str(r["actions"]), "supply + withdraw over the window"),
                tile("Markets", "3", "risk 3 / 4 / 6 · caps 0.40 · hurdles 4% / 3% / 4%"),
            ],
            "series": [
                series("Portfolio value ($)", c.index.date, c["portfolio"], unit="$"),
                series("Position by market ($)", c.index.date, {k: c[f"pos_{k}"] for k in keys}, unit="$"),
                series("Supply APY by market (borrow APR × utilisation)", c.index.date, {k: c[f"apy_{k}"] for k in keys}, unit="%"),
                series("Idle USDC ($)", c.index.date, c["idle"], unit="$"),
            ],
            "notes": [
                "Supply-side strategy: the loss path is bad debt / oracle failure, not price; the gates (utilisation cap, oracle staleness, hurdle hysteresis, concentration) are the whole risk model.",
                "Roughly half the book sat idle at 0% for the whole window: with three markets whose window-mean APY hovers around the 3–4% hurdles and hard 0.40 weight caps, the hurdle hysteresis and caps bind, not the per-tick deploy caps (the 10× scenario moves deployment by ~1 point). In production the fallback market (cbBTC/USDC) absorbed the remainder; here there is none, so the idle drag is shown rather than hidden.",
                "The quadratic term is inert at this window length: with a 4-sample APY window the covariance is ~1e-5 and λ = 1, so the penalty is ~1e-6 against a linear term of ~0.05 and SLSQP returns the greedy fill under caps. The same was true of the 6 h production window. The risk control is the haircut, the caps and the gates — the solver is there for the day a longer window is used.",
            ],
        }
    # junior-tranche stress
    jr_rows = "| apxUSD depeg | junior share 25% | junior share 33% | junior share 50% |\n|---|---|---|---|\n" + "\n".join(
        f"| {d*100:.1f}% | {fmt_pct(-junior_loss_from_depeg(d, 0.25))} | {fmt_pct(-junior_loss_from_depeg(d, 0.33))} | {fmt_pct(-junior_loss_from_depeg(d, 0.50))} |"
        for d in (0.02, 0.05, 0.085, 0.16, 0.25))
    scenarios["junior"] = {
        "label": "Junior-tranche variant · depeg stress",
        "note": "Royco Dawn junior tranches (syrupUSDC, apyUSD, sNUSD, stcUSD) — score = APY / risk, 24h async redemptions",
        "verdict": "June 2026 postmortem: ~25.6k USDC in the apyUSD junior tranche lost 8,733 USDC (−34.1%) when apxUSD depegged to $0.84 — ~4× leverage on an 8.5% depeg, ~$3.3k of it timing",
        "verdict_tone": "bad",
        "tiles": [
            tile("Position", "$25,596", "18,600 apyUSD → 25,078 JT shares, 2026-05-26/30"),
            tile("Realised loss", "−$8,733", "−34.1% in USDC terms", "bad"),
            tile("Underlying depeg", "−16%", "apxUSD $1.00 → $0.84 trough on 2026-06-18"),
            tile("Junior leverage", "~4×", "junior absorbs the senior's share of the drawdown"),
            tile("Timing cost", "~$3.3k", "redeemed during FIXED_TERM at the trough; 4 days later −$5.4k instead of −$10.1k"),
            tile("What held", "cap ≤25%", "concentration cap bounded exposure; coverage gate cancelled the over-ask no-penalty"),
        ],
        "tables": [{"title": "Junior-tranche loss vs underlying depeg (first-loss arithmetic)", "md": jr_rows + "\n\n`loss_jr = min(1, depeg / junior_share)` — the tranche's protection of the senior is leverage on the junior."}],
        "notes": [
            "Guardrails I wrote up from this: price the junior tranche's depeg leverage into the risk score (not just the APY), never redeem into a FIXED_TERM trough when the term clears the coverage loss, persist request intent before broadcast (the orphaned-request incidents of 06-07/08), and cross-check TVL adapters against `convertToAssets` (a +37% double-count was live for days).",
        ],
    }
    sections = [
        {"heading": "What this is", "body_md": (
            "Two supply-side USDC clusters I designed and operated on mainnet: (1) a Morpho Blue lending cluster that spreads a vault's idle USDC across curated markets with a "
            "mean-variance allocator, hard eligibility gates and a planner that turns weight deltas into supply/withdraw calls through the merkle-verified vault manager; "
            "(2) a junior-tranche cluster on Royco Dawn markets with 24-hour asynchronous redemptions, a wrap → finalize → execute deposit pipeline and a repair pipeline for orphaned on-chain requests.\n\n"
            "```\nsupply_apy = borrow_apr × utilisation × (1 − fee)\nmax  mu_adj'w − (λ/2) w'Σw     mu_adj = mu − 50 bps × max(0, risk − 3)\ns.t. 0 ≤ w ≤ min(max_weight, capacity/portfolio),  Σw = min(1, Σ upper)\ncapacity = min(abs_cap, concentration_cap × market_borrows)\n```")},
        {"heading": "Eligibility and exits", "body_md": (
            "| gate | rule |\n|---|---|\n"
            "| oracle health | ≥ 2 observations in window, heartbeat-aware staleness per feed (BTC/USD 2 h, mF-ONE 5 d), else zero capacity |\n"
            "| utilisation | > 0.95 → excluded; health job trims supplied markets above 0.95 while liquidity exists |\n"
            "| hurdle hysteresis | 6 h-mean APY below hurdle for 3 consecutive ticks (18 h) → no new allocation |\n"
            "| exit band | existing position tolerated to hurdle − 50 bps; below for 3 ticks → exit |\n"
            "| deposit depth | a deposit may lower utilisation by ≤ 100 bps: `max_dep = borrow/(util − 0.01) − supply` |\n"
            "| drift gate | no rebalance unless total drift ≥ 1% of portfolio; per-leg floor 1,000 USDC |\n"
            "| withdrawals | reserve = pending queue demand + 50 bps, 1 h floor, subgraph ≤ 50 blocks behind; withdraw ≤ 80% of visible liquidity per market, worst market first |\n\n"
            "Hurdle checks run on the window mean, not the instantaneous print — the same hysteresis idea as the loop cluster's net-gate.")},
        {"heading": "Production review and incidents", "body_md": (
            "- Lending cluster readiness review: 13 findings (P0–P2) before capital — receipt-status handling, 300 s receipt timeout, nonce lock across build + send, equality constraint when the upper bounds don't sum to 1, fail-safe when the market-data source is fully down. Suite 88 → 100+ tests; `validate_leaves` and `validate_runtime` preflights.\n"
            "- Junior cluster first deployment (2026-05-26, fixed in-session): reverted txs treated as success; aggregator fills short of quote reverting whole bundles (fixed by sizing at `amountOutMin`, then at realised balance); a 7,751% APY spike from annualising delta rows over short intervals; a dead APY signal that funnelled every deposit to one market; TVL adapters over-valuing tranches by +37% / +4.8%.\n"
            "- Orphaned on-chain requests (06-07/08): a stale-credentials DB incident orphaned requests whose tx succeeded with no DB row, and a cron re-requesting every 20 min against a shrinking balance produced a geometric dust loop (9 cancel txs). Shipped detection (`audit_gateway_requests`, `reconcile_redemptions`) and a runbook; root fix = persist intent before broadcast.\n"
            "- apyUSD junior-tranche loss (06-18): see the stress scenario.")},
        {"heading": "Operations", "body_md": (
            "Lending: 6 h reconcile cron (rebalance) + 20 min health cron (withdrawal reserve, utilisation exits, EOA gas ≥ 0.005 ETH, heartbeat staleness); dry-run by default, `--confirm` to submit; emergency exit by market or all; Grafana (health, heartbeats, actions, allocator state); Discord on every submit and every non-OK health tick.\n\n"
            "Junior: executor 5 min / allocator 20 min / withdraw-prep 20 min / health 15 min on a single APScheduler process; per-market 8–14 merkle leaves; health checks for paused entry points, tranche NAV, share-price HWM deposit-halt, drain staleness, subgraph lag; unit tests with ~65% line coverage (private).")},
        {"heading": "Reproduce", "body_md": "```\npython -m strategies.lending_cluster.backtest\n```\n`strategy.py` = allocator (SLSQP), gates, planner, tranche arithmetic; `backtest.py` = daily replay on `data/algo_history.csv` and the depeg stress table."},
    ]
    return {"title": "USDC Lending Cluster — Morpho Blue supply + junior tranches",
            "subtitle": "Mean-variance allocator with risk haircuts, eligibility gates, planner and withdrawal reserve · mainnet · plus the apyUSD postmortem",
            "scenarios": scenarios, "default_scenario": "prod", "sections": sections}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    res = build_results()
    write_results(here, res)
    render_page(here, res)
    for n, s in res["scenarios"].items():
        print(n, "|", s["verdict"])
