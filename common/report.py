"""Write a strategy's results.json and render its HTML page from the shared template.

results.json shape:
{
  "title", "subtitle", "generated",
  "scenarios": {name: {"label", "note", "verdict", "verdict_tone", "tiles": [...], "series": [...], "tables": [...], "notes": [...]}},
  "default_scenario", "sections": [{"heading", "body_md"}]
}
tile   = {"label", "value", "sub", "tone": "good|bad|neutral|warn"}
series = {"label", "x": [...], "y": [...] | {name: [...]}, "kind": "line|bar", "unit"}

Tone rules (applied through `tone_for`): "good" only when a P&L beats its stated benchmark over a
window of at least MIN_GOOD_DAYS; shorter windows and synthetic data are never "good". Annualised
figures are only produced for windows of at least MIN_ANNUALISE_DAYS (see `ann_or_window`).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "pages" / "template.html"
MIN_GOOD_DAYS = 30
MIN_ANNUALISE_DAYS = 30
STRIDE_TARGET = 400   # max points per series written to results.json
SIG_FIGS = 10         # floats are rounded to this many significant figures when written


def fmt_usd(x: float, digits: int = 1) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 999_950:
        return f"{sign}${x/1e6:.{digits}f}M"
    if x >= 1e3:
        return f"{sign}${x/1e3:.{digits}f}k"
    return f"{sign}${x:,.2f}"


def fmt_pct(x: float, digits: int = 2) -> str:
    if abs(x) < 10 ** -(digits + 2):
        return f"{0:.{digits}f}%"
    return f"{x*100:+.{digits}f}%"


def tone_for(pnl: float, days: float, synthetic: bool = False, benchmark: float | None = None) -> str:
    """'good' needs a real-data window >= MIN_GOOD_DAYS and pnl above a STATED benchmark; with no benchmark the
    best a positive result gets is 'neutral'. 'bad' is any loss (vs the benchmark when one is given)."""
    if benchmark is None:
        return "bad" if pnl < 0 else "neutral"
    if pnl < benchmark:
        return "bad"
    if synthetic or days < MIN_GOOD_DAYS - 0.05:
        return "neutral"
    return "good"


def ann_or_window(ret: float, days: float) -> tuple[str, str]:
    """(label, value) — annualised only for windows >= MIN_ANNUALISE_DAYS, else the plain window return."""
    if days >= MIN_ANNUALISE_DAYS - 0.05:
        return "Annualised (simple)", fmt_pct(ret * 365 / days)
    if days >= 1:
        return f"Window return ({days:.1f} d)", fmt_pct(ret)
    return f"Window return ({days*24:.1f} h)", fmt_pct(ret)


def tile(label: str, value: str, sub: str = "", tone: str = "neutral") -> dict:
    return {"label": label, "value": value, "sub": sub, "tone": tone}


def series(label: str, x, y, kind: str = "line", unit: str = "") -> dict:
    """Series are strided down to ~STRIDE_TARGET points so pages stay small."""
    x = list(x)
    n = len(x)
    step = max(1, n // STRIDE_TARGET)

    def clean(v):
        v = list(map(float, v))[::step]
        return [None if f != f else f for f in v]
    if isinstance(y, dict):
        y = {k: clean(v) for k, v in y.items()}
    else:
        y = clean(y)
    return {"label": label, "x": [str(v) for v in x[::step]], "y": y, "kind": kind, "unit": unit}


def round_floats(obj, sig: int = SIG_FIGS):
    """Round every float to `sig` significant figures so last-bit noise across numpy/scipy builds
    doesn't change the written files."""
    if isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, sig) for v in obj]
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    try:
        return float(f"{float(obj):.{sig}g}")
    except (TypeError, ValueError):
        return obj


def write_results(strategy_dir: Path, results: dict) -> Path:
    results.setdefault("generated", date.today().isoformat())
    out = strategy_dir / "results.json"
    out.write_text(json.dumps(round_floats(results), indent=1, default=float))
    return out


def render_page(strategy_dir: Path, results: dict | None = None) -> Path:
    """Inline results.json into the template so the page works from file:// with no server."""
    if results is None:
        results = json.loads((strategy_dir / "results.json").read_text())
    html = TEMPLATE.read_text()
    payload = json.dumps(round_floats(results), default=float).replace("</", "<\\/")
    html = html.replace("/*__RESULTS__*/null", payload)
    html = html.replace("__TITLE__", results.get("title", "Strategy"))
    out = strategy_dir / "index.html"
    out.write_text(html)
    return out
