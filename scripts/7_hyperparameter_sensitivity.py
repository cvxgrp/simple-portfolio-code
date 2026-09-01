"""Small one-at-a-time sweep around the production ridge-alpha parameters."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from simple_portfolio.alpha.geo_alpha import get_ridge_alpha_over_time
from simple_portfolio.metrics import compute_metrics
from simple_portfolio.optimizer import AnchoredVolControlPortfolioConstructor
from simple_portfolio.simulator import BacktestData, BacktestResults, run_backtest

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2026-01-01")
EVAL_START = pd.Timestamp("2006-01-01")
VOL_TARGET = 0.07 / np.sqrt(252)
ASSETS = ["SPY", "AGG", "GLD"]
BASE = {"h": 100, "min_history": 512, "halflife": 252, "ridge": 10.0}
SPECS = [
    ("baseline", BASE),
    ("h=63", BASE | {"h": 63}),
    ("h=84", BASE | {"h": 84}),
    ("h=110", BASE | {"h": 110}),
    ("h=126", BASE | {"h": 126}),
    ("window=252", BASE | {"min_history": 252}),
    ("window=420", BASE | {"min_history": 420}),
    ("window=630", BASE | {"min_history": 630}),
    ("window=756", BASE | {"min_history": 756}),
    ("halflife=126", BASE | {"halflife": 126}),
    ("halflife=200", BASE | {"halflife": 200}),
    ("halflife=315", BASE | {"halflife": 315}),
    ("halflife=504", BASE | {"halflife": 504}),
    ("ridge=3", BASE | {"ridge": 3.0}),
    ("ridge=7", BASE | {"ridge": 7.0}),
    ("ridge=15", BASE | {"ridge": 15.0}),
    ("ridge=30", BASE | {"ridge": 30.0}),
]
WINDOWS = {
    "full": (pd.Timestamp("2006-01-01"), pd.Timestamp("2026-01-01")),
    "first_half": (pd.Timestamp("2006-01-01"), pd.Timestamp("2016-01-01")),
    "second_half": (pd.Timestamp("2016-01-01"), pd.Timestamp("2026-01-01")),
}
METRICS = ["Return", "Volatility", "Sharpe Ratio (FFR)", "Max Drawdown", "Turnover"]


def sliced(result: BacktestResults, start: pd.Timestamp, end: pd.Timestamp) -> BacktestResults:
    """Slice a result to an evaluation window."""
    return BacktestResults(
        navs=result.navs.loc[start:end],
        composition=result.composition.loc[start:end],
        turnover=result.turnover.loc[start:end],
        metadata=result.metadata,
    )


def main() -> None:
    """Fit each alpha variant, backtest it, and save full and split-sample metrics."""
    closes = pd.read_parquet("data/raw/closes.parquet")
    ext_features = pd.read_parquet("data/processed/ext_features.parquet")
    etf_features = pd.read_parquet("data/processed/etf_features.parquet")
    etf_features = etf_features[
        [col for col in etf_features if any(f"_{asset}" in col for asset in ASSETS)]
    ]
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0
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
    for label, params in SPECS:
        print(f"Fitting {label}: {params}", flush=True)
        alpha = get_ridge_alpha_over_time(
            closes,
            ext_features,
            etf_features,
            ASSETS,
            min_window=63,
            verbose=False,
            **params,
        )
        alpha = alpha.reindex(index=closes.index, columns=closes.columns, fill_value=0.0).fillna(
            0.0
        )
        strategy = AnchoredVolControlPortfolioConstructor(
            ts_lookup=lookup,
            alphas=alpha.loc[data.timeline].to_numpy(),
            risk_model=risk_model,
            vol_target=VOL_TARGET,
            anchor=anchor,
            universe=universe,
            leverage=1.0,
        )
        result = run_backtest(label, data, strategy)
        for window, (start, end) in WINDOWS.items():
            rows[(label, window)] = compute_metrics(sliced(result, start, end), ffr, cpi)[METRICS]

    table = pd.DataFrame(rows).T
    table.index.names = ["Specification", "Window"]
    Path("output/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("output/tables/ridge_hyperparameter_sweep.csv")
    table.xs("full", level="Window").sort_values("Sharpe Ratio (FFR)", ascending=False).to_csv(
        "output/tables/ridge_hyperparameter_sweep_full.csv"
    )
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))


main()
