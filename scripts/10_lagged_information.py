"""Backtest Markowitz with alpha-only and joint alpha/covariance information lags."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from simple_portfolio.metrics import compute_metrics
from simple_portfolio.optimizer import AnchoredVolControlPortfolioConstructor
from simple_portfolio.simulator import BacktestData, BacktestResults, run_backtest

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2026-01-01")
EVAL_START = pd.Timestamp("2006-01-01")
VOL_TARGET = 0.07 / np.sqrt(252)
ASSETS = ["SPY", "AGG", "GLD"]
LAGS = [0, 1, 5, 21]
METRICS = [
    "Return",
    "Volatility",
    "Sharpe Ratio (FFR)",
    "Max Drawdown",
    "Avg. Drawdown",
    "Turnover",
]


def lag_risk_model(
    risk_model: dict[str, dict[pd.Timestamp, np.ndarray]], lag: int
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    """Associate each risk-model date with the estimate from ``lag`` trading days earlier."""
    dates = sorted(risk_model["Fs"])
    return {
        component: {date: values[dates[max(i - lag, 0)]] for i, date in enumerate(dates)}
        for component, values in risk_model.items()
    }


def evaluate(result: BacktestResults, ffr: pd.Series, cpi: pd.Series) -> pd.Series:
    """Evaluate over the paper's common 2006--2026 window."""
    sliced = BacktestResults(
        navs=result.navs.loc[EVAL_START:END_DATE],
        composition=result.composition.loc[EVAL_START:END_DATE],
        turnover=result.turnover.loc[EVAL_START:END_DATE],
        metadata=result.metadata,
    )
    return compute_metrics(sliced, ffr, cpi)[METRICS]


def main() -> None:
    """Run Markowitz with alpha-only and joint information lags."""
    closes = pd.read_parquet("data/raw/closes.parquet")
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0
    alpha = pd.read_parquet("data/processed/alpha_ridge_sbg.parquet")
    with Path("data/processed/risk_model/sbg.pkl").open("rb") as file:
        risk_model = pickle.load(file)  # noqa: S301

    data = BacktestData.from_pandas(
        start_date=START_DATE,
        end_date=END_DATE,
        closes=closes,
        ffrs=ffr,
        rebal_freq="M",
    )
    lookup = {ts: i for i, ts in enumerate(data.timeline)}
    anchor = np.zeros(len(data.assets))
    universe = np.zeros(len(data.assets), dtype=bool)
    for asset, weight in zip(ASSETS, [0.5, 0.3, 0.2], strict=True):
        idx = int(np.where(data.assets == asset)[0][0])
        anchor[idx] = weight
        universe[idx] = True

    rows = {}
    for lag in LAGS:
        lagged_alpha = alpha.shift(lag).fillna(0.0) if lag else alpha
        risk_models = {
            "Alpha and covariance lagged": lag_risk_model(risk_model, lag),
            "Alpha only lagged": risk_model,
        }
        for test, test_risk_model in risk_models.items():
            strategy = AnchoredVolControlPortfolioConstructor(
                ts_lookup=lookup,
                alphas=lagged_alpha.loc[data.timeline].to_numpy(),
                risk_model=test_risk_model,
                vol_target=VOL_TARGET,
                anchor=anchor,
                universe=universe,
                leverage=1.0,
            )
            result = run_backtest(f"{test}, {lag}-day lag", data, strategy)
            rows[(test, lag)] = evaluate(result, ffr, cpi)

    table = pd.DataFrame(rows).T
    table.index.names = ["Information lag", "Lag (trading days)"]
    Path("output/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("output/tables/results_lagged_information.csv")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))


main()
