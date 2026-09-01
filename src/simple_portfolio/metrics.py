"""Metrics for portfolio backtesting."""

import numpy as np
import pandas as pd

from simple_portfolio.simulator import BacktestResults


def consistency(navs: pd.Series) -> float:
    """Consistency metric kappa from the cumulative log-return trajectory."""
    L = np.log(navs.to_numpy() / navs.to_numpy()[0])
    t = np.arange(len(L), dtype=float)
    b = L[-1] / t[-1]  # mean per-period log return
    return (252 * b * len(L)) / np.abs(L - b * t).sum()


def mdcr(navs: pd.Series) -> float:
    """Mean deviation from constant return (MDCR) from the cumulative log-return trajectory."""
    L = np.log(navs.to_numpy() / navs.to_numpy()[0])
    rho = L[-1] / (len(L))  # mean per-period log return
    t = np.arange(len(L), dtype=float)
    exp_nav = np.exp(rho * t)
    dcrs = np.abs(navs / exp_nav - 1)
    return dcrs.mean()


def compute_metrics(results: BacktestResults, ffrs: pd.Series, cpi: pd.Series) -> pd.Series:
    """Compute metrics for a backtest results."""
    navs = results.navs
    returns = navs.pct_change().dropna()
    cumu_rets = navs.iloc[-1] / navs.iloc[0]
    cumu_ffr = (1 + ffrs.loc[returns.index]).prod()
    cumu_cpi = (1 + cpi.loc[returns.index]).prod()

    mean_returns = cumu_rets ** (252 / len(returns)) - 1
    mean_ffr_returns = (cumu_rets / cumu_ffr) ** (252 / len(returns)) - 1
    mean_cpi_returns = (cumu_rets / cumu_cpi) ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)

    sharpe = mean_returns / vol
    ffr_sharpe = mean_ffr_returns / vol
    cpi_sharpe = mean_cpi_returns / vol

    cummax = navs.cummax()
    drawdown = (cummax - navs) / cummax
    max_drawdown = drawdown.max()
    avg_drawdown = drawdown.mean()
    turnover = results.turnover.mean() * 252

    stats = pd.Series(
        {
            "Return": mean_returns,
            "Return - FFR": mean_ffr_returns,
            "Return - CPI": mean_cpi_returns,
            "Volatility": vol,
            "Sharpe Ratio": sharpe,
            "Sharpe Ratio (FFR)": ffr_sharpe,
            "Sharpe Ratio (CPI)": cpi_sharpe,
            "Max Drawdown": max_drawdown,
            "Avg. Drawdown": avg_drawdown,
            "Turnover": turnover,
            "Consistency": consistency(navs),
            "MDCR": mdcr(navs),
        }
    )
    return stats
