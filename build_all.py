"""Rebuild every strategy's results.json and page.  python build_all.py"""
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = [
    "strategies.leverage_loop.backtest",
    "strategies.lending_cluster.backtest",
    "strategies.dn_cluster.backtest",
    "strategies.hedged_btc_mm.backtest",
    "strategies.eurc_usdc_mm.backtest",
    "strategies.coin_carry_vault.backtest",
    "strategies.vol_cone_forecast.forecast",
]

if __name__ == "__main__":
    for m in MODULES:
        print("==", m)
        subprocess.run([sys.executable, "-m", m], cwd=ROOT, check=True)
