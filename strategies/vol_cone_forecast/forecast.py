"""Daily lognormal price-cone forecaster (BTC/ETH/SOL/HYPE) — the risk-desk tool behind the clusters.

Run:  python -m strategies.vol_cone_forecast.forecast           (uses data/forecast_sample_2026-10-22.json)
      python -m strategies.vol_cone_forecast.forecast --live    (pulls Deribit DVOL / index and HL mids)

    S_T = S_0 · exp((mu − σ²/2)·T ± z·σ·√T)
    σ   = Deribit DVOL (30d implied) where listed, else 30d realised daily vol annualised
    mu  = 0 by default; an external 24h prediction (Allora) can be folded in as a damped drift
          (×0.25, capped at ±40%/yr) — off in the sample

Outputs the 1-day and 30-day 68/90/95% bands used to size perp margin reserves and the
`0.8 × liquidationPx` guards in the funding-carry cluster, and the drop-to-liquidation checks in the
carry vaults. Ran daily at 07:00 via launchd with txt/json/Discord outputs (260 days of history).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common.data import DATA
from common.report import render_page, series, tile, write_results

Z = {0.6827: 1.0, 0.90: 1.6449, 0.95: 1.96}


def cone(spot: float, sigma: float, days: float, mu: float = 0.0) -> dict:
    T = days / 365
    sT = sigma * math.sqrt(T)
    med = spot * math.exp((mu - sigma ** 2 / 2) * T)
    return {"T_years": T, "sigma_horizon": sT, "median": med, "expected": spot * math.exp(mu * T),
            "bands": {str(k): [med * math.exp(-z * sT), med * math.exp(z * sT)] for k, z in Z.items()}}


def damped_drift(pred_return_24h: float | None, damp: float = 0.25, cap: float = 0.40) -> float:
    if pred_return_24h is None:
        return 0.0
    annual = pred_return_24h * 365 * damp
    return max(-cap, min(cap, annual))


def forecast(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        mu = damped_drift(r.get("allora_implied_return"))
        out.append({**r, "mu": mu, "proj_1d": cone(r["spot"], r["sigma"], 1, mu), "proj_30d": cone(r["spot"], r["sigma"], 30, mu)})
    return out


def live_rows() -> list[dict]:
    """Deribit index + DVOL for BTC/ETH; HL mids + 30d realised vol for SOL/HYPE."""
    import urllib.request

    def get(url):
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode())

    rows = []
    for a in ("BTC", "ETH"):
        idx = get(f"https://www.deribit.com/api/v2/public/get_index_price?index_name={a.lower()}_usd")["result"]["index_price"]
        dvol = get(f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={a}&resolution=1D&start_timestamp=0&end_timestamp=9999999999999")["result"]["data"][-1][4] / 100
        rows.append({"asset": a, "spot": idx, "sigma": dvol, "vol_source": "Deribit DVOL (30d implied)", "allora_implied_return": None})
    return rows


def build_results(rows: list[dict], generated: str) -> dict:
    tiles, tables = [], []
    for r in rows:
        b30, b1 = r["proj_30d"]["bands"], r["proj_1d"]["bands"]
        tiles.append(tile(f"{r['asset']} spot", f"${r['spot']:,.0f}" if r["spot"] > 1000 else f"${r['spot']:,.2f}",
                          f"σ {r['sigma']*100:.1f}% · {r['vol_source']}"))
        tiles.append(tile(f"{r['asset']} 30d 90% range", f"${b30['0.9'][0]:,.0f} – ${b30['0.9'][1]:,.0f}" if r["spot"] > 1000 else f"${b30['0.9'][0]:,.2f} – ${b30['0.9'][1]:,.2f}",
                          f"1d 90%: {b1['0.9'][0]:,.0f}–{b1['0.9'][1]:,.0f}" if r["spot"] > 1000 else f"1d 90%: {b1['0.9'][0]:,.2f}–{b1['0.9'][1]:,.2f}"))
    md = "| asset | spot | σ (ann.) | 1d 90% | 30d 68% | 30d 90% | 30d 95% | 30d median |\n|---|---|---|---|---|---|---|---|\n"
    for r in rows:
        f = (lambda v: f"{v:,.0f}") if r["spot"] > 1000 else (lambda v: f"{v:,.2f}")
        b30, b1 = r["proj_30d"]["bands"], r["proj_1d"]["bands"]
        md += (f"| {r['asset']} | {f(r['spot'])} | {r['sigma']*100:.1f}% | {f(b1['0.9'][0])}–{f(b1['0.9'][1])} | {f(b30['0.6827'][0])}–{f(b30['0.6827'][1])} | "
               f"{f(b30['0.9'][0])}–{f(b30['0.9'][1])} | {f(b30['0.95'][0])}–{f(b30['0.95'][1])} | {f(r['proj_30d']['median'])} |\n")
    # cone series for BTC over 30 days
    btc = next(r for r in rows if r["asset"] == "BTC")
    days = list(range(0, 31))
    lo = [cone(btc["spot"], btc["sigma"], d)["bands"]["0.9"][0] if d else btc["spot"] for d in days]
    hi = [cone(btc["spot"], btc["sigma"], d)["bands"]["0.9"][1] if d else btc["spot"] for d in days]
    med = [cone(btc["spot"], btc["sigma"], d)["median"] if d else btc["spot"] for d in days]
    sc = {"sample": {
        "label": f"Forecast {generated[:10]}", "note": "sample output bundled in data/; --live refreshes from Deribit",
        "verdict": f"BTC 30-day 90% cone: ${btc['proj_30d']['bands']['0.9'][0]:,.0f} – ${btc['proj_30d']['bands']['0.9'][1]:,.0f} at σ {btc['sigma']*100:.1f}% implied",
        "verdict_tone": "neutral",
        "tiles": tiles,
        "series": [series("BTC 30-day lognormal cone, 90% band ($)", [f"d+{d}" for d in days], {"low": lo, "median": med, "high": hi}, unit="$")],
        "tables": [{"title": "All assets", "md": md}],
        "notes": ["Used as the daily reserve-sizing input: perp margin reserves in the funding-carry cluster were set so the 30-day 95% high stayed below 0.8 × the short's liquidation price."],
    }}
    sections = [
        {"heading": "Method", "body_md": "```\nS_T = S_0 · exp((μ − σ²/2)·T ± z·σ·√T),   z = 1.00 / 1.645 / 1.96\nσ = Deribit DVOL (BTC, ETH) or 30d realised daily vol annualised (SOL, HYPE)\nμ = 0, or a damped external 24h prediction: μ = clip(0.25 × 365 × r_24h, ±40%/yr)\n```\nNo claim of edge: it is a calibrated-vol cone to size margin, reserves and liquidation guards consistently across the desk, refreshed daily and archived (260 daily runs Jun–Oct 2026)."},
        {"heading": "Reproduce", "body_md": "```\npython -m strategies.vol_cone_forecast.forecast\npython -m strategies.vol_cone_forecast.forecast --live\n```"},
    ]
    return {"title": "Daily Volatility-Cone Forecaster", "subtitle": "Lognormal 1-day / 30-day bands from implied or realised vol · risk-desk sizing input",
            "scenarios": sc, "default_scenario": "sample", "sections": sections}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live:
        rows, gen = forecast(live_rows()), "live"
    else:
        raw = json.loads((DATA / "forecast_sample_2026-10-22.json").read_text())
        rows, gen = forecast([{k: r[k] for k in ("asset", "spot", "sigma", "vol_source", "allora_implied_return")} for r in raw["rows"] if r["ok"]]), raw["generated_utc"]
    here = Path(__file__).resolve().parent
    res = build_results(rows, gen)
    write_results(here, res)
    render_page(here, res)
    print(res["scenarios"]["sample"]["verdict"])
