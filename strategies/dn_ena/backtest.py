"""dnENA — PT-sENA fixed yield + token-for-token ENA short, four scenario families on one engine.

Run:  python -m strategies.dn_ena.backtest
Data (fixed layout, see fetch_data.py): data/pendle_pt_sena.csv, data/hl_funding_ENA.csv, data/binance_ENAUSDT_1h.csv.
Until those exist the engine runs on a SYNTHETIC path that reproduces the launch-page window's shape
(ENA −25% over 88 days, 18% implied PT yield, small funding) so the page builds and says so.

Scenarios
  1. actual     the Feb–Apr 2026 window the published backtest page used, reproduced from data
  2. rolling    every 88-day window across the data, 1-day steps: a distribution, not one number
  3. stress     ENA +100% in a week (margin top-up and idle-capital cost); funding −20%/yr for a month;
                a 10% sENA/ENA depeg at redemption (the tail the hedge does not cover)
  4. rotation   the "rotate only when gain > fee" rule vs a single-print rule vs never, same data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import metrics
from common.data import DATA
from common.report import ann_or_window, fmt_pct, fmt_usd, render_page, series, tile, tone_for, write_results
from strategies.dn_ena.strategy import (Params, margin_health, pt_discount_apy, round_trip_fee, short_liquidation_px,
                                        should_rotate, size_entry, topup_needed)

CAPITAL = 100_000.0
HOLD_DAYS = 88


# ----------------------------------------------------------------------------- data

def load() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Hourly frame: px (ENA/USD), funding (hourly rate); PT frame: time, market, maturity, pt_price_ena, implied_apy."""
    f, k, p = DATA / "hl_funding_ENA.csv", DATA / "binance_ENAUSDT_1h.csv", DATA / "pendle_pt_sena.csv"
    if f.exists() and k.exists() and p.exists():
        fund = pd.read_csv(f, parse_dates=["time"]).set_index("time")["fundingRate"]
        px = pd.read_csv(k, parse_dates=["time"]).set_index("time")["open"].iloc[:-1]
        fund.index, px.index = fund.index.floor("h"), px.index.floor("h")
        fund, px = fund[~fund.index.duplicated(keep="last")], px[~px.index.duplicated(keep="last")]
        hourly = pd.concat([px.rename("px"), fund.rename("funding")], axis=1, sort=True).dropna()
        pt = pd.read_csv(p, parse_dates=["time", "maturity"])
        pt["time"] = pt["time"].dt.floor("h")
        return hourly, pt, "Pendle PT-sENA history + Hyperliquid ENA funding + Binance ENA spot (fetched)"
    return synthetic(), None, "SYNTHETIC — reproduces the launch-page window's shape; run fetch_data.py to replace"


def synthetic(days: int = 400, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24
    t = pd.date_range("2025-09-01", periods=n, freq="h", tz="UTC")
    sig = 1.1 / np.sqrt(8760)
    r = rng.normal(-sig * sig / 2, sig, n)             # zero-drift GBM: no free upward trend in the placeholder
    px = 0.45 * np.exp(np.cumsum(r))
    f = np.zeros(n)
    for i in range(1, n):
        f[i] = 0.98 * f[i - 1] + rng.normal(0, 3e-6)
    return pd.DataFrame({"px": px, "funding": f}, index=t)


def pt_curve(hourly_index: pd.DatetimeIndex, pt: pd.DataFrame | None, start: pd.Timestamp, implied_apy: float = 0.18) -> tuple[pd.Series, pd.Timestamp, pd.Series]:
    """PT price in ENA terms and the maturity for a position opened at `start`.
    With data: the market that is already trading at `start` whose maturity is the nearest at or beyond start + HOLD_DAYS;
    if none reaches that far, the longest-dated market trading at `start` (a shorter hold). Never synthetic when data exists.
    Without data: a constant-yield accretion (placeholder, labelled on the page)."""
    if pt is not None:
        live = pt[(pt["time"] <= start) & (pt["maturity"] > start + pd.Timedelta(days=7))]
        if len(live):
            mats = live.groupby("market")["maturity"].first()
            far = mats[mats >= start + pd.Timedelta(days=HOLD_DAYS)]
            m = far.idxmin() if len(far) else mats.idxmax()
            s = pt[pt["market"] == m].set_index("time").sort_index()
            s = s[~s.index.duplicated(keep="last")]
            mat = s["maturity"].iloc[0]
            price = s["pt_price_ena"].reindex(hourly_index).ffill()      # NaN before listing; a window never starts there
            apy = s["implied_apy"].reindex(hourly_index).ffill()
            price[hourly_index >= mat] = 1.0
            return price, mat, apy
        raise ValueError(f"no PT-sENA market trading at {start}")
    mat = start + pd.Timedelta(days=HOLD_DAYS)
    implied_apy = 0.18 + 0.04 * np.sin(2 * np.pi * start.dayofyear / 60)   # placeholder term structure
    ttm_days = np.clip((mat - hourly_index).total_seconds() / 86400, 0, None)
    price = pd.Series(1.0 / (1 + implied_apy) ** (ttm_days / 365), index=hourly_index)
    return price, mat, pd.Series(implied_apy, index=hourly_index)


# ----------------------------------------------------------------------------- engine

def run(hourly: pd.DataFrame, pt_px: pd.Series, maturity: pd.Timestamp, start: pd.Timestamp, p: Params,
        capital: float = CAPITAL, depeg_at_maturity: float = 0.0, funding_override: pd.Series | None = None,
        px_override: pd.Series | None = None, exit_at: pd.Timestamp | None = None) -> dict:
    """Hold one PT position from `start` to maturity (or data end, or an early `exit_at`), with the short,
    reserve and top-ups. At maturity the PT redeems at par (less any depeg); an early exit sells it at pt_px."""
    end = min(maturity, hourly.index[-1]) if exit_at is None else min(exit_at, maturity, hourly.index[-1])
    w = hourly.loc[start:end]
    px = (px_override if px_override is not None else w["px"]).loc[w.index]
    fund = (funding_override if funding_override is not None else w["funding"]).loc[w.index]
    ptp = pt_px.loc[w.index]
    px0, pt0 = float(px.iloc[0]), float(ptp.iloc[0])
    e = size_entry(capital, px0, pt0, p)
    q, margin, reserve = e["qty"], e["margin"], e["reserve"]
    fees = e["pt_cost"] * p.pendle_fee + q * px0 * (p.hl_taker + p.hl_slippage)
    liq_px = short_liquidation_px(px0, margin, q, p.maint_margin)
    fund_cum = topups = 0.0
    hedge_on, hedge_lost_at = True, None
    rows = []
    for t, (x, fr, pp) in enumerate(zip(px, fund, ptp)):
        if hedge_on:
            fund_cum += fr * q * x                                   # short receives when funding > 0
            need = topup_needed(margin, px0, x, q, fund_cum, p)
            if need > 0:
                moved = min(need, reserve)
                reserve -= moved
                margin += moved
                topups += moved
                if margin_health(margin, px0, x, q, fund_cum, p.maint_margin) < 1.0:   # reserve exhausted: liquidated
                    hedge_on, hedge_lost_at = False, w.index[t]
                    fees += q * x * (p.hl_taker + p.hl_slippage)
                    short_pnl_locked = q * (px0 - x) + fund_cum
        short_pnl = (q * (px0 - x) + fund_cum) if hedge_on else short_pnl_locked
        hl_equity = margin + short_pnl + reserve if hedge_on else max(0.0, margin + short_pnl_locked) + reserve
        pt_val = q * pp * x
        equity = pt_val + hl_equity - fees
        rows.append({"time": w.index[t], "equity": equity, "px": x, "pt_px": pp, "hl_equity": hl_equity, "funding_cum": fund_cum,
                     "health": margin_health(margin, px0, x, q, fund_cum, p.maint_margin) if hedge_on else 0.0,
                     "delta_tokens": q * pp - (q if hedge_on else 0.0)})
    c = pd.DataFrame(rows).set_index("time")
    # redemption: PT -> sENA -> ENA at (1 - depeg); close the short
    x_end = float(px.iloc[-1])
    redeem = q * (1 - depeg_at_maturity) * x_end if w.index[-1] >= maturity else q * float(ptp.iloc[-1]) * x_end
    fees += q * x_end * (p.hl_taker + p.hl_slippage) + (0.0 if w.index[-1] >= maturity else q * ptp.iloc[-1] * x_end * p.pendle_fee)
    final = redeem + (margin + (q * (px0 - x_end) + fund_cum if hedge_on else short_pnl_locked) + reserve) - fees
    days = (w.index[-1] - w.index[0]).total_seconds() / 86400
    return {"curve": c, "final": final, "pnl": final - capital, "days": days, "qty": q, "pt_cost": e["pt_cost"], "margin0": e["margin"],
            "reserve0": e["reserve"], "funding": fund_cum, "fees": fees, "topups": topups, "liq_px": liq_px,
            "worst_health": float(c["health"][c["health"] > 0].min()) if (c["health"] > 0).any() else 0.0,
            "hedge_lost_at": hedge_lost_at, "max_dd": metrics.max_drawdown(c["equity"]), "implied_apy": pt_discount_apy(pt0, max(1.0, (maturity - w.index[0]).total_seconds() / 86400)),
            "pt_pnl": redeem - e["pt_cost"], "short_pnl": (q * (px0 - x_end) if hedge_on else short_pnl_locked - fund_cum)}


# ----------------------------------------------------------------------------- scenario families

def actual_window(hourly, pt, p: Params) -> tuple[dict, pd.Timestamp]:
    start = pd.Timestamp("2026-02-01", tz="UTC")
    if start not in hourly.index:
        start = hourly.index[max(0, len(hourly) // 3)]
    ptp, mat, _ = pt_curve(hourly.index, pt, start)
    return run(hourly, ptp, mat, start, p), start


def rolling(hourly, pt, p: Params, step_days: int = 1) -> pd.DataFrame:
    out, skipped = [], 0
    starts = hourly.index[::24 * step_days]
    for s in starts:
        if s + pd.Timedelta(days=HOLD_DAYS) > hourly.index[-1]:
            break
        try:
            ptp, mat, _ = pt_curve(hourly.index, pt, s)
        except ValueError:
            skipped += 1                              # no PT-sENA market trading yet: counted, not modelled
            continue
        r = run(hourly, ptp, mat, s, p)
        out.append({"start": s, "ret": r["pnl"] / CAPITAL, "days": r["days"], "funding": r["funding"], "fees": r["fees"], "topups": r["topups"],
                    "worst_health": r["worst_health"], "hedge_lost": r["hedge_lost_at"] is not None, "pt_pnl": r["pt_pnl"], "short_pnl": r["short_pnl"]})
    df = pd.DataFrame(out)
    df.attrs["skipped"] = skipped
    return df


def stresses(hourly, pt, p: Params, start) -> dict:
    ptp, mat, _ = pt_curve(hourly.index, pt, start)
    base = run(hourly, ptp, mat, start, p)
    # a) +100% from entry over the first week (linear), then flat at 2x: isolates the top-up rule and the idle-capital cost
    px = hourly["px"].copy()
    w = px.loc[start:]
    n1 = min(24 * 7, len(w))
    shocked = np.empty(len(w))
    shocked[:n1] = w.iloc[0] * np.linspace(1, 2, n1)
    if len(w) > n1:
        shocked[n1:] = 2 * w.iloc[0]
    px.loc[start:] = shocked
    rally = run(hourly, ptp, mat, start, p, px_override=px)
    # where is the cliff: scale the week-1 rally until the hedge is lost
    cliff = None
    for k in np.arange(2.0, 4.01, 0.05):
        px_k = hourly["px"].copy()
        sk = np.empty(len(w))
        sk[:n1] = w.iloc[0] * np.linspace(1, k, n1)
        if len(w) > n1:
            sk[n1:] = k * w.iloc[0]
        px_k.loc[start:] = sk
        if run(hourly, ptp, mat, start, p, px_override=px_k)["hedge_lost_at"] is not None:
            cliff = float(k)
            break
    # b) funding at -20%/yr for the first 30 days
    fu = hourly["funding"].copy()
    fu.loc[start:start + pd.Timedelta(days=30)] = -0.20 / 8760
    squeeze = run(hourly, ptp, mat, start, p, funding_override=fu)
    # c) 10% sENA/ENA depeg realised at redemption
    return {"base": base, "rally": rally, "squeeze": squeeze, "cliff": cliff}


def depeg_stress(hourly, pt, p: Params, start, depeg: float = 0.10) -> dict:
    """The uncovered tail: PT redeems sENA, the hedge is ENA. Not a page scenario; run `python -m strategies.dn_ena.backtest --depeg`."""
    ptp, mat, _ = pt_curve(hourly.index, pt, start)
    return run(hourly, ptp, mat, start, p, depeg_at_maturity=depeg)


def rotation_study(hourly, pt, start, check_every_h: int = 24) -> pd.DataFrame:
    """Chain of positions over the data from `start`. Each day from one week after entry the rule is asked whether to sell
    the current PT at market and buy the longest-dated PT trading that day (a different maturity); otherwise hold to maturity
    and roll into the next market. Rotation pays the round-trip fee on the notional."""
    out = []
    for rule in ("gain_gt_fee", "single_print", "never"):
        p = Params(rotate_rule=rule)
        t, cap, rolls, cost, log = start, CAPITAL, 0, 0.0, []
        while t + pd.Timedelta(days=14) < hourly.index[-1]:
            try:
                ptp, mat, apy = pt_curve(hourly.index, pt, t)
            except ValueError:
                break
            r = run(hourly, ptp, mat, t, p, capital=cap)
            exit_at = None
            if rule != "never":
                for ts in hourly.loc[t + pd.Timedelta(days=7):min(mat, hourly.index[-1]) - pd.Timedelta(days=7)].index[::check_every_h]:
                    try:
                        ptp2, mat2, apy2 = pt_curve(hourly.index, pt, ts)
                    except ValueError:
                        continue
                    if mat2 == mat:
                        continue
                    notional = r["qty"] * float(hourly["px"].loc[ts])
                    if should_rotate(float(apy.loc[ts]), (mat - ts).days, float(apy2.loc[ts]), (mat2 - ts).days, notional, p):
                        exit_at = ts
                        break
            if exit_at is not None:
                r = run(hourly, ptp, mat, t, p, capital=cap, exit_at=exit_at)   # sell the PT at market
                fee = round_trip_fee(r["qty"] * float(hourly["px"].loc[exit_at]), p)   # reported; the legs are already charged in run()
                log.append((t, exit_at, mat, float(apy.loc[exit_at]), float(apy2.loc[exit_at]), mat2))
                cap, cost, rolls, t = r["final"], cost + fee, rolls + 1, exit_at
                continue
            cap, t = r["final"], min(mat, hourly.index[-1]) + pd.Timedelta(hours=1)
            if t not in hourly.index:
                break
        out.append({"rule": rule, "final": cap, "ret": cap / CAPITAL - 1, "rotations": rolls, "rotation_cost": cost, "log": log})
    return pd.DataFrame(out)


def overlap_table(pt: pd.DataFrame | None, p: Params) -> str:
    """For every period in which two PT-sENA maturities traded at once: the running PT's implied APY, the new listing's,
    and the gap the gain-over-fee rule needs to rotate (round-trip fee / remaining horizon)."""
    if pt is None:
        return "no Pendle data"
    mats = pt.groupby("market")["maturity"].first().sort_values()
    rows = ["| running PT | new listing | overlap (days) | running APY (median) | new APY (median) | gap needed to rotate | fired |", "|---|---|---|---|---|---|---|"]
    ms = list(mats.index)
    for a, b_ in zip(ms[:-1], ms[1:]):
        sa, sb = pt[pt["market"] == a].set_index("time")["implied_apy"], pt[pt["market"] == b_].set_index("time")["implied_apy"]
        both = pd.concat([sa.rename("a"), sb.rename("b")], axis=1).dropna()
        if both.empty:
            continue
        days_left = max(1.0, (mats[a] - both.index[0]).days)
        need = round_trip_fee(1.0, p) / (days_left / 365)
        fired = bool(((both["b"] - both["a"]) * (((mats[a] - both.index).days.values.clip(1)) / 365) > round_trip_fee(1.0, p)).any())
        rows.append(f"| {mats[a]:%d %b %Y} | {mats[b_]:%d %b %Y} | {(both.index[-1]-both.index[0]).days} | {both['a'].median():.1%} | {both['b'].median():.1%} | {need:+.1%} (at first overlap) | {'yes' if fired else 'no'} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- page

def build_results() -> dict:
    hourly, pt, src = load()
    synthetic_data = src.startswith("SYNTHETIC")
    p = Params()
    act, start = actual_window(hourly, pt, p)
    roll = rolling(hourly, pt, p)
    sweep = {rf: rolling(hourly, pt, Params(reserve_frac=rf), step_days=2) for rf in (0.15, 0.30, 0.50)}
    st = stresses(hourly, pt, p, start)
    rot = rotation_study(hourly, pt, start)
    c = act["curve"]
    x = c.index.strftime("%m-%d")
    scen = {}
    scen["actual"] = {
        "label": "Actual window (Feb–Apr 2026)" + (" — SYNTHETIC placeholder" if synthetic_data else ""),
        "note": f"PT-sENA + 1x ENA short on HyperCore, $100k, 15% reserve, held to maturity · data: {src}",
        "verdict": f"{fmt_pct(act['pnl']/CAPITAL)} over {act['days']:.0f} days: PT leg {fmt_usd(act['pt_pnl'])}, short {fmt_usd(act['short_pnl'])}, funding {fmt_usd(act['funding'])}, fees {fmt_usd(-act['fees'])}"
                   + (f"; published backtest page on real data: +$1,650 (+1.65%, 6.83% ann.)" if synthetic_data else ""),
        "verdict_tone": tone_for(act["pnl"], act["days"], synthetic_data),
        "tiles": [
            tile("Net PnL", fmt_usd(act["pnl"]), fmt_pct(act["pnl"] / CAPITAL) + " on capital" + (" · synthetic path" if synthetic_data else ""), tone_for(act["pnl"], act["days"], synthetic_data)),
            tile(*ann_or_window(act["pnl"] / CAPITAL, act["days"]), f"{act['days']:.0f}-day hold"),
            tile("PT implied APY at entry", fmt_pct(act["implied_apy"]), "fixed yield locked at the discount"),
            tile("Funding received", fmt_usd(act["funding"]), "on the 1x short; sign follows ENA funding"),
            tile("Fees", fmt_usd(-act["fees"]), "Pendle 8 bps on the PT buy (redemption is free) + HL 4.5 bps per perp leg (taker 3.5 + 1 slippage, what the published run charged)"),
            tile("PT deployed / short notional", f"{fmt_usd(act['pt_cost'])} / {fmt_usd(act['margin0'])}", f"{act['qty']:,.0f} PT = {act['qty']:,.0f} ENA short"),
            tile("Liquidation price (entry)", f"${act['liq_px']:.4f}", f"{act['liq_px']/float(c['px'].iloc[0])-1:+.0%} above entry on the margin alone; the reserve extends it to about {(act['margin0']+act['reserve0'])/(act['qty']*float(c['px'].iloc[0]))*100:+.0f}% (stress scenario)"),
            tile("Worst margin health", f"{act['worst_health']*100:,.0f}%", "100% = liquidation; the top-up rule holds health at 150%; top-ups from reserve " + (f"{fmt_usd(act['topups'])}" if act["topups"] else "none needed")),
            tile("Max drawdown", fmt_pct(act["max_dd"]), "mark-to-market incl. PT discount"),
            tile("Published page, same window (typed in, not computed)", "+$1,650", "testingdnena: +1.65%, 6.83% ann., funding $84.86, fees −$72.35, DD −$885 — the figures the scenario above is checked against", "neutral"),
        ],
        "series": [
            series("Portfolio equity ($)", x, c["equity"], unit="$"),
            series("ENA spot vs PT mark ($)", x, {"ENA": c["px"], "PT (ENA × pt_price)": c["px"] * c["pt_px"]}),
            series("Net delta drift (PT delta − hedge, tokens)", x, c["delta_tokens"]),
            series("HL account equity ($)", x, c["hl_equity"], unit="$"),
        ],
        "notes": [
            "The PT has less than one token of delta before maturity (it trades at a discount) while the hedge is fixed at 1.0 per PT, so the book is net short early and converges to flat at maturity; this is the design, and the drift series shows it.",
            "Deployment is chunked ($20k per hour) for HL book depth in the live design; the engine opens at once, which flatters nothing in a down-market window and slightly understates slippage in an up-market one.",
        ],
    }
    if len(roll):
        q = roll["ret"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        scen["rolling"] = {
            "label": f"Rolling 88-day windows ({len(roll)})" + (" — SYNTHETIC placeholder" if synthetic_data else ""),
            "note": f"every start date, 1-day steps, same rules; {roll.attrs.get('skipped', 0)} start dates before the first PT-sENA listing skipped, none modelled · data: {src}",
            "verdict": f"{(roll['ret'] > 0).mean()*100:.0f}% of {len(roll)} windows positive, median {fmt_pct(q[0.5] * 365 / HOLD_DAYS)} annualised ({fmt_pct(q[0.5])} per window), worst {fmt_pct(roll['ret'].min())}; hedge lost in {int(roll['hedge_lost'].sum())}",
            "verdict_tone": "neutral" if synthetic_data else ("good" if q[0.05] > 0 else "neutral"),
            "tiles": [
                tile("Windows positive", f"{(roll['ret'] > 0).mean()*100:.0f}%", f"{len(roll)} windows of {HOLD_DAYS} days, every start date"),
                tile("Median window return", fmt_pct(q[0.5]), f"{fmt_pct(q[0.5] * 365 / HOLD_DAYS)} annualised (simple)"),
                tile("Worst window", fmt_pct(roll['ret'].min()), f"start {roll.loc[roll['ret'].idxmin(), 'start']:%Y-%m-%d}"),
                tile("5th / 95th percentile", f"{fmt_pct(q[0.05])} / {fmt_pct(q[0.95])}", "distribution, not a number"),
                tile("Hedge lost (reserve exhausted)", str(int(roll["hedge_lost"].sum())), "windows where top-ups ran out and the short was liquidated"),
                tile("Median funding", fmt_usd(roll["funding"].median()), "per window"),
                tile("Median top-ups", fmt_usd(roll["topups"].median()), "reserve moved to margin"),
            ],
            "tables": [{"title": "Reserve sweep (same windows, 2-day steps)", "md":
                        "| reserve | windows positive | median annualised | worst window | hedge lost |\n|---|---|---|---|---|\n" + "\n".join(
                            f"| {rf:.0%} | {(d['ret'] > 0).mean()*100:.0f}% | {fmt_pct(d['ret'].median() * 365 / HOLD_DAYS)} | {fmt_pct(d['ret'].min())} | {int(d['hedge_lost'].sum())} of {len(d)} |"
                            for rf, d in sweep.items())
                        + "\n\nAt 1x the short is liquidated when ENA rises by roughly (1 + reserve/notional) above entry; the reserve is the only parameter that moves that line. A larger reserve earns nothing and lowers the yield on capital — the trade shown here."}],
            "series": [series("Window return by start date", roll["start"].dt.strftime("%Y-%m-%d"), roll["ret"], unit="%"),
                       series("Worst margin health by start date (100% = liquidation)", roll["start"].dt.strftime("%Y-%m-%d"), roll["worst_health"] * 100)],
            "notes": ["A fixed yield locked at entry makes the base case positive by construction; what varies is funding, the rally-driven top-ups, and whether the reserve survives. Windows where it does not are the result to look at, and the reserve sweep is the design response.",
                      "A window holds the PT to the nearest maturity at or beyond 88 days from its start, or to the longest-dated market trading at the time when none reaches that far, so window lengths vary with Pendle's listing calendar."],
        }
    b, ra, sq = st["base"], st["rally"], st["squeeze"]
    scen["stress"] = {
        "label": "Stresses",
        "note": "same window, one shock at a time, stated as stresses; the rally is held flat at 2x after the week so only the top-up rule is being tested",
        "verdict": f"ENA +100% in a week: {fmt_pct(ra['pnl']/CAPITAL)} ({'hedge held' if ra['hedge_lost_at'] is None else 'HEDGE LOST'}, top-ups {fmt_usd(ra['topups'])} of the {fmt_usd(ra['reserve0'])} reserve, worst health {ra['worst_health']*100:,.0f}%); funding −20%/yr for 30 d: {fmt_pct(sq['pnl']/CAPITAL)} (funding {fmt_usd(sq['funding'])} against a fixed-yield leg of {fmt_usd(b['pt_pnl']+b['short_pnl'])})",
        "verdict_tone": "neutral",
        "tiles": [
            tile("Base (this window)", fmt_pct(b["pnl"] / CAPITAL), "no shock"),
            tile("ENA +100% in week 1, then flat", fmt_pct(ra["pnl"] / CAPITAL), f"top-ups {fmt_usd(ra['topups'])} of {fmt_usd(ra['reserve0'])} reserve · worst health {ra['worst_health']*100:,.0f}% · " + ("hedge held" if ra["hedge_lost_at"] is None else f"hedge lost {ra['hedge_lost_at']:%Y-%m-%d}"), "bad" if ra["hedge_lost_at"] else "neutral"),
            tile("Funding −20%/yr for 30 d", fmt_pct(sq["pnl"] / CAPITAL), f"funding {fmt_usd(sq['funding'])}: eats {abs(sq['funding'])/max(1,(b['pt_pnl']+b['short_pnl'])):.0%} of the fixed-yield leg", "bad" if sq["pnl"] < 0 else "neutral"),
            tile("Idle capital in the rally", fmt_usd(ra["topups"]), "reserve moved to margin to hold the hedge; earns nothing while there"),
            tile("Where the hedge is lost", f"×{st['cliff']:.2f}" if st["cliff"] else "> ×4", "week-1 rally multiple at which margin + reserve are exhausted (same path, scaled)"),
        ],
        "series": [series("Equity under each stress ($)", ra["curve"].index.strftime("%m-%d"),
                          {"base": b["curve"]["equity"], "rally": ra["curve"]["equity"], "funding squeeze": sq["curve"]["equity"]}, unit="$")],
        "notes": ["If a rally outruns margin + reserve the short is liquidated and the book is left long PT: the result then follows ENA's price, whatever its sign — the reserve is sized against this case, and the tiles show how much of it the +100% week used.",
                  "The one risk the hedge does not cover is the sENA/ENA exchange rate: the PT redeems sENA, the short is ENA. It is a property of the underlying rather than a parameter of the book, and it is run as a script in the repo rather than a page scenario (`--depeg`)."],
    }
    scen["rotation"] = {
        "label": "Rotation rule",
        "note": "chain of positions from the same start; the rule is checked once per position, at its midpoint, against the PT that would be bought then (the live design checked on every new listing)",
        "verdict": (f"no rule fired: {fmt_pct(rot['ret'].iloc[0])} for gain_gt_fee, single_print and never alike — no new listing printed above the running PT in any overlap (table)"
                    if rot["rotations"].sum() == 0 else " · ".join(f"{r.rule}: {fmt_pct(r.ret)} with {r.rotations} rotations ({fmt_usd(r.rotation_cost)} cost)" for r in rot.itertuples())),
        "verdict_tone": "neutral",
        "tiles": ([tile("All three rules", fmt_pct(rot["ret"].iloc[0]), "identical chains: no overlap offered a gap that clears the fee hurdle — see the overlap table", "neutral")]
                  if rot["rotations"].sum() == 0 else
                  [tile(r.rule, fmt_pct(r.ret), f"{r.rotations} early rotations · {fmt_usd(r.rotation_cost)} of round-trip fees (inside the fees line)") for r in rot.itertuples()]),
        "tables": [{"title": "Every overlap of two PT-sENA maturities in the data", "md": overlap_table(pt, Params()) +
                    f"\n\nRound-trip fee {round_trip_fee(1.0, Params())*1e4:.0f} bps of notional (2 × Pendle 8 bps + 2 × HL 4.5 bps); the rule rotates only when (new APY − running APY) × remaining horizon beats it."}],
        "notes": ["`gain_gt_fee` is the rule in the design: exit early only if (next_apy − cur_apy) × remaining horizon × notional beats the round-trip fee (2 × Pendle 8 bps + 2 × HL 4.5 bps). "
                  "`single_print` rotates on any better print and pays the round-trip every time. `never` holds every PT to maturity.",
                  "Where the three rules agree it is because no new listing printed above the running PT during an overlap (the table): sENA maturities on Pendle list a few weeks before the previous one expires, and the expiring PT's implied APY runs up into maturity. The rule is live in the chain and checked daily; on this data it has nothing to fire on, which is itself the answer to 'why not roll early'."],
    }
    sections = [
        {"heading": "What this is", "body_md": (
            "dnENA: buy PT-sENA on Pendle at a discount, short the same token quantity of ENA on HyperCore at 1x, hold a 15% reserve in the HL account, top up margin from the reserve before liquidation, and hold to maturity — rolling only when the expected gain on a new maturity beats the round-trip fee. "
            "Return = the fixed yield locked at the PT discount + funding on the short − fees: the yield is known at entry, funding is the variable, delta is designed out token-for-token, and the margin is run defensively from the reserve.\n\n"
            "```\nq = (capital − reserve) / (pt_px·px + px/lev)        pt_qty = short_qty = q\nimplied APY = (1/pt_px)^(365/days) − 1              liq_px = (margin + q·px0) / (q·(1 + maint))\ntop up from reserve when equity/maintenance < 1.5     rotate iff (apy_next − apy_cur)·T·notional > round-trip fee\n```")},
        {"heading": "How to read the four families", "body_md": (
            "1. **Actual window** — the published backtest page's Feb–Apr 2026 window reproduced from Pendle, HL and Binance data (its real-data result is kept as a tile; if the rebuilt number differs the page shows the rebuilt one).\n"
            "2. **Rolling windows** — the same rules on every 88-day start date; the distribution is the result and the negative windows are the finding.\n"
            "3. **Stresses** — one shock at a time, labelled as such: a +100% rally in a week (does the top-up rule hold, what idle capital does it cost), −20%/yr funding for a month (how much of the fixed yield it eats).\n"
            "4. **Rotation** — the gain-over-fee rule against a single-print rule and against never rotating, on the same data: the design argument shown with numbers.")},
        {"heading": "Reproduce", "body_md": "```\npython -m strategies.dn_ena.fetch_data --from 2025-06-01    # Pendle PT-sENA history, HL ENA funding, Binance ENA spot\npython -m strategies.dn_ena.backtest\npython -m strategies.dn_ena.backtest --depeg             # the sENA/ENA depeg script (prints, no page)\n```\n`strategy.py` = sizing, margin, top-up, rotation rule; `backtest.py` = the engine and the four scenario families."},
    ]
    return {"title": "dnENA — PT-sENA fixed yield + token-for-token ENA short",
            "subtitle": "Pendle PT + HyperCore 1x short with a 15% reserve, held to maturity · actual window, rolling windows, stresses, rotation rule",
            "scenarios": scen, "default_scenario": "actual", "sections": sections}


if __name__ == "__main__":
    import sys
    here = Path(__file__).resolve().parent
    if "--depeg" in sys.argv:
        hourly, pt, src = load()
        p = Params()
        act, start = actual_window(hourly, pt, p)
        for dp in (0.02, 0.05, 0.10):
            r = depeg_stress(hourly, pt, p, start, dp)
            print(f"sENA/ENA depeg {dp:.0%} at redemption: {fmt_pct(r['pnl']/CAPITAL)} on $100k (base {fmt_pct(act['pnl']/CAPITAL)}) · data: {src}")
        sys.exit(0)
    res = build_results()
    write_results(here, res)
    render_page(here, res)
    for n, s in res["scenarios"].items():
        print(n, "|", s["verdict"][:200])
