"""Daily simulation of the BTC carry vault on the LTV ladder.

Run:  python -m strategies.coin_carry_vault.backtest
Data: Morpho USDC borrow APR from data/algo_history.csv (siUSD/USDC market, 93 days, real) as the
      borrow-cost series; BTC price path SYNTHETIC (GBM, 45% vol, labelled) with a stress path
      containing a -35% drawdown; syUSD yield 9.8% (the public vault lineup figure, Jun 2026).
Everything is measured in BTC: the depositor's benchmark is holding the coin.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import metrics
from common.data import morpho_algo_history
from common.report import fmt_pct, render_page, series, tile, tone_for, write_results
from strategies.coin_carry_vault.strategy import Ladder, decide, ltv, price_drop_to_liquidation

DEPOSIT_BTC = 10.0
SYUSD_APY = 0.098
SLIPPAGE = 0.001   # on flash-deleverage collateral sales


def paths(days: int = 93, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.45 / np.sqrt(365), days)
    base = 82_000 * np.exp(np.cumsum(r))
    crash = base.copy()
    k = days // 2
    crash[k:] *= 0.65          # single-day -35% gap: the ladder gets no daily steps to de-lever into
    return {"gbm": base, "crash": crash}


def run(px: np.ndarray, borrow: np.ndarray, L: Ladder) -> dict:
    coll_btc, debt_usd, credit_usd = DEPOSIT_BTC, 0.0, 0.0
    rows, acts = [], {"relever": 0, "delever": 0, "delever_emergency": 0, "liquidated": False, "liq_cost_btc": 0.0}
    for i, (p, b) in enumerate(zip(px, borrow)):
        credit_usd *= 1 + SYUSD_APY / 365
        debt_usd *= 1 + b / 365
        coll_usd = coll_btc * p
        cur = ltv(debt_usd, coll_usd)
        ltv_pre = cur                                   # LTV as the day opened, before any liquidation
        if cur >= L.liquidation_ltv:
            # Morpho liquidation: the liquidator repays debt and seizes debt × (1 + LIF) of collateral,
            # capped at the collateral held; any residual debt is bad debt socialised to the market, not
            # charged to this position. LIF 5% (Morpho's incentive at this LLTV is ~4.3%; 5% is conservative).
            seize = min(coll_btc, debt_usd * 1.05 / p)
            repaid = min(debt_usd, seize * p / 1.05)
            coll_btc -= seize
            debt_usd = 0.0 if coll_btc <= 1e-12 else debt_usd - repaid   # residual after full seizure is bad debt, socialised
            acts["liquidated"] = True
            acts["liq_cost_btc"] = acts.get("liq_cost_btc", 0.0) + (seize - repaid / p)   # the incentive paid
            coll_usd = coll_btc * p                                                  # decide() sees the post-seizure book
        action, amt = decide(debt_usd, coll_usd, b, SYUSD_APY, 0.0, L)
        if action.startswith("delever"):
            use = min(amt, credit_usd)
            credit_usd -= use
            debt_usd -= use
            short = amt - use
            if short > 0:  # flash deleverage: sell collateral
                coll_btc -= short * (1 + SLIPPAGE) / p
                debt_usd -= short
            acts[action] += 1
        elif action == "relever":
            debt_usd += amt
            credit_usd += amt
            acts["relever"] += 1
        eq_btc = coll_btc + (credit_usd - debt_usd) / p
        rows.append({"day": i, "price": p, "equity_btc": eq_btc, "ltv": ltv_pre, "ltv_close": ltv(debt_usd, coll_btc * p),
                     "debt": debt_usd, "credit": credit_usd, "borrow_apr": b, "carry": SYUSD_APY - b,
                     "liq_drop": price_drop_to_liquidation(ltv_pre, L.liquidation_ltv)})
    c = pd.DataFrame(rows).set_index("day")
    return {"curve": c, "actions": acts, "metrics": metrics.summary(c["equity_btc"], metrics.DAILY)}


def build_results() -> dict:
    hist = morpho_algo_history()
    borrow = hist[hist.market == "siusd"].sort_values("date")["borrow_apr"].values
    dates = hist[hist.market == "siusd"].sort_values("date")["date"].dt.date.values
    L = Ladder()
    scenarios = {}
    for name, label, px in (("gbm", "BTC GBM path (45% vol)", paths()["gbm"]), ("crash", "BTC −35% one-day gap", paths()["crash"])):
        r = run(px, borrow, L)
        c, m, a = r["curve"], r["metrics"], r["actions"]
        excess = c["equity_btc"].iloc[-1] / DEPOSIT_BTC - 1
        scenarios[name] = {
            "label": label,
            "note": "cbBTC collateral → Morpho USDC borrow at target LTV 0.65 → syUSD at 9.8%; borrow APR = real Morpho series (93 days); price path SYNTHETIC (labelled)",
            "verdict": f"{excess:+.3%} in BTC terms over 93 days vs holding ({fmt_pct(excess*365/93)} ann.), {a['delever']+a['delever_emergency']} de-levers, {a['relever']} re-levers"
                       + (", LIQUIDATED" if a["liquidated"] else ", never liquidated"),
            "verdict_tone": "bad" if a["liquidated"] or excess < 0 else "neutral",
            "tiles": [
                tile("Excess vs hold (BTC)", f"{excess:+.3%}", f"{c['equity_btc'].iloc[-1]:.4f} BTC on 10 BTC · synthetic price path", "bad" if excess < 0 else "neutral"),
                tile("Annualised (simple)", fmt_pct(excess * 365 / 93), "in BTC units — price exposure is the depositor's"),
                tile("Avg net carry", fmt_pct(c["carry"].mean()), "syUSD 9.8% − Morpho USDC borrow APR"),
                tile("Max borrow APR", fmt_pct(c["borrow_apr"].max()), "real Morpho series; re-lever blocked above 5%"),
                tile("Max LTV (day open)", f"{c['ltv'].max():.3f}", "before any action that day · ladder 0.60 / 0.65 / 0.70 / 0.74 / liq 0.86", "bad" if c["ltv"].max() >= 0.86 else "neutral"),
                tile("Min drop to liquidation", fmt_pct(c["liq_drop"].min()), "worst day open: `1 − ltv/0.86`; negative = already past LLTV", "bad" if c["liq_drop"].min() < 0.15 else "neutral"),
                tile("De-levers / re-levers", f"{a['delever']+a['delever_emergency']} / {a['relever']}", f"{a['delever_emergency']} emergency (LTV > 0.74)"),
                tile("Liquidated", "yes" if a["liquidated"] else "no", (f"liquidation incentive paid {a['liq_cost_btc']:.3f} BTC; collateral seized, USDC credit kept — measured in BTC the book is near flat because the credit buys more BTC after the gap" if a["liquidated"] else "Morpho LLTV 0.86"), "bad" if a["liquidated"] else "neutral"),
            ],
            "series": [
                series("Equity in BTC (vs 10 BTC hold)", dates, c["equity_btc"]),
                series("LTV vs ladder (day open)", dates, {"ltv": c["ltv"], "max 0.70": [0.70] * len(c), "liq 0.86": [0.86] * len(c)}),
                series("BTC price (path)", dates, c["price"], unit="$"),
                series("Net carry: syUSD − borrow APR", dates, c["carry"], unit="%"),
            ],
            "notes": [
                "After a liquidation the book is all USDC credit, so its BTC-denominated equity then moves inversely with BTC for the rest of the window; the liquidation itself cost the 5% incentive on the seized collateral.",
                "SYNTHETIC BTC path; real Morpho USDC borrow APR (siUSD/USDC market) as the borrow cost; syUSD yield held at 9.8%. This page is ladder arithmetic and the delever route, not a return claim.",
                "Carry on the ETH variant adds wstETH staking yield on the collateral; the BTC variant has only the stable-vault spread, so it inverts whenever USDC borrow > ~9.8%.",
                "De-levering first drains the yield-vault credit; only the shortfall is sold via flash-deleverage (collateral sold inside a Morpho flashloan at 10 bps slippage).",
            ],
        }
    sections = [
            {"heading": "Ladder arithmetic", "body_md": (
            "| LTV now | drop to liquidation (LLTV 0.86) | LTV after a −35% gap |\n|---|---|---|\n" + "\n".join(
                f"| {l:.2f} | {(1 - l/0.86)*100:.1f}% | {l/0.65:.2f} |" for l in (0.50, 0.60, 0.65, 0.70, 0.74)) +
            "\n\nThe ladder is tuned for drift, not gaps: it holds LTV in a 0.60–0.70 band and de-levers daily, which is enough for a −20% move spread over days and not enough for a −35% move in one block. "
            "Position sizing against the one-day tail (the vol-cone tool's 95% 30-day low) is the control that would matter.")},
        {"heading": "What this is", "body_md": (
            "Asset-denominated carry vaults (cbBTC and wstETH collateral) on Morpho Blue: borrow USDC to a target LTV and deposit it in a stablecoin yield vault; the depositor keeps the coin's price exposure and earns the spread on top. "
            "I owned the LTV ladder, the carry guard, the delever decision tree, the two-stage redemption and the repair pipeline that made the exits safe when the yield vault was illiquid.\n\n"
            "```\nnet carry on borrowed USDC = yield_vault_apy (+ staking yield on ETH) − borrow_apr\nLTV = debt_usd / collateral_usd ;  drop to liquidation = 1 − LTV / 0.86   (0.65 → 24.4%)\n```")},
        {"heading": "LTV ladder and decision tree", "body_md": (
            "| level | value | action |\n|---|---|---|\n| MIN | 0.60 | re-lever to target if borrow APR ≤ 5% and carry > 0, and no pending withdrawals |\n| TARGET | 0.65 | open / rebalance aim |\n| MAX | 0.70 | de-lever to target: `repay = debt − 0.65 × collateral` |\n| EMERGENCY | 0.74 | same, bypass the 50 gwei gas budget |\n| LIQUIDATION | 0.86 | Morpho market LLTV (12-point buffer) |\n\n"
            "Freeing USDC to repay follows a fixed route: yield-vault liquidity ≥ needed? no → flash-deleverage (sell collateral inside a flashloan, with a quoted-out ≥ needed guard); "
            "residual after idle USDC ≤ dust → repay with idle (1 tx); vault holds enough shares → two-stage redeem (request, ~60 s maturity, self-solve close in one atomic tx); else flash-deleverage. "
            "Safety preempts economics; withdrawals preempt re-levering; state is re-read between the two branches.")},
        {"heading": "Repair pipeline and hardening", "body_md": (
            "- Two-stage redemption writes a durable `close` ledger row from the receipt before the wait, and releases the snapshot lock during the wait (an earlier version held it 60 s).\n"
            "- Repair pipeline: flash-deleverage → repair request → Postgres job row → worker with live-priced floor, deadline guard and attempt cap; jobs claimed with `FOR UPDATE SKIP LOCKED`.\n"
            "- Known race documented with fix candidates: a repair request orphaned between phases 1b and 1c; pre-launch LTV at ~76.6% traced to ~186 syUSD escrowed in expired requests.\n"
            "- Event-driven snapshots from Chainlink `AnswerUpdated` on the price aggregators (debounced ~20 s) on top of the cron; preflight validator checks vault addresses, LLTV == 0.86, feed freshness, EOA gas and borrow APR vs the re-lever ceiling.\n"
            "- Unit and mainnet-fork test suites (private).")},
        {"heading": "Reproduce", "body_md": "```\npython -m strategies.coin_carry_vault.backtest\n```"},
    ]
    return {"title": "BTC / ETH Carry Vaults — Morpho Blue LTV ladder",
            "subtitle": "Coin-denominated leveraged carry with a hardened delever / repair pipeline · Ethereum mainnet",
            "scenarios": scenarios, "default_scenario": "gbm", "sections": sections}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    res = build_results()
    write_results(here, res)
    render_page(here, res)
    for n, s in res["scenarios"].items():
        print(n, "|", s["verdict"])
