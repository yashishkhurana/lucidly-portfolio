"""Performance and risk metrics shared by every strategy backtest.

All functions take plain numpy arrays / pandas Series of per-period returns or equity
and return floats. Periods-per-year is always explicit so daily, hourly and
8-hour (funding window) series are handled the same way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAILY = 365
HOURLY = 365 * 24
FUNDING_8H = 365 * 3


def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = (len(equity) - 1) / periods_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def annualised_vol(rets: pd.Series, periods_per_year: int) -> float:
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(rets: pd.Series, periods_per_year: int, rf_annual: float = 0.0) -> float:
    if len(rets) < 2:
        return float("nan")
    excess = rets - rf_annual / periods_per_year
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sortino(rets: pd.Series, periods_per_year: int) -> float:
    downside = rets[rets < 0]
    if len(downside) < 2:
        return float("nan")
    dd = np.sqrt((downside**2).mean())
    return float(rets.mean() / dd * np.sqrt(periods_per_year)) if dd > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summary(equity: pd.Series, periods_per_year: int) -> dict:
    rets = returns_from_equity(equity)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "ann_vol": annualised_vol(rets, periods_per_year),
        "sharpe": sharpe(rets, periods_per_year),
        "sortino": sortino(rets, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "periods": int(len(equity)),
        "pct_periods_positive": float((rets > 0).mean()) if len(rets) else float("nan"),
    }
