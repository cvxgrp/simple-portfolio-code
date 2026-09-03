"""End-date, one-opt, and walk-forward stability checks for ridge Markowitz."""

import pickle
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simple_portfolio.alpha.geo_alpha import get_ridge_alpha_over_time  # noqa: E402
from simple_portfolio.metrics import compute_metrics  # noqa: E402
from simple_portfolio.optimizer import AnchoredVolControlPortfolioConstructor  # noqa: E402
from simple_portfolio.simulator import (  # noqa: E402
    BacktestData,
    BacktestResults,
    run_backtest,
)

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2026-01-01")
EVAL_START = pd.Timestamp("2006-01-01")
VOL_TARGET = 0.07 / np.sqrt(252)
ASSETS = ["SPY", "AGG", "GLD"]
BASE_PARAMS = {
    "cov_window": 11,
    "horizon": 100,
    "alpha_halflife": 252,
    "ridge": 10.0,
}
SPECS = {
    "baseline": BASE_PARAMS,
    "cov_window=5": BASE_PARAMS | {"cov_window": 5},
    "cov_window=22": BASE_PARAMS | {"cov_window": 22},
    "horizon=84": BASE_PARAMS | {"horizon": 84},
    "horizon=110": BASE_PARAMS | {"horizon": 110},
    "alpha_halflife=200": BASE_PARAMS | {"alpha_halflife": 200},
    "alpha_halflife=315": BASE_PARAMS | {"alpha_halflife": 315},
    "ridge=7": BASE_PARAMS | {"ridge": 7.0},
    "ridge=15": BASE_PARAMS | {"ridge": 15.0},
}
COORDINATE_GROUPS = {
    "cov_window": ["cov_window=5", "baseline", "cov_window=22"],
    "horizon": ["horizon=84", "baseline", "horizon=110"],
    "alpha_halflife": ["alpha_halflife=200", "baseline", "alpha_halflife=315"],
    "ridge": ["ridge=7", "baseline", "ridge=15"],
}
METRICS = [
    "Return",
    "Volatility",
    "Sharpe Ratio (FFR)",
    "Max Drawdown",
    "Avg. Drawdown",
    "Turnover",
]


class SwitchingConstructor:
    """Delegate each rebalance to the candidate selected for that calendar year."""

    def __init__(self, constructors: dict, selections: dict[int, str]) -> None:
        self.constructors = constructors
        self.selections = selections

    def __call__(self, ts, curr_weights, universe, ffr):
        """Use the selected constructor, falling back to baseline before selection starts."""
        label = self.selections.get(pd.Timestamp(ts).year, "baseline")
        return self.constructors[label](ts, curr_weights, universe, ffr)


def covariance_risk_model(
    returns: pd.DataFrame, lookback: int
) -> dict[str, dict[pd.Timestamp, np.ndarray]]:
    """Reproduce the production SBG risk model with a chosen trailing window."""
    asset_index = returns.columns
    factors = list(range(len(ASSETS)))
    fs: dict[pd.Timestamp, pd.DataFrame] = {}
    d_halves: dict[pd.Timestamp, pd.Series] = {}
    for i in range(lookback, len(returns)):
        date = returns.index[i]
        window = returns.loc[:date, ASSETS].tail(lookback).dropna()
        if len(window) < lookback:
            continue
        vols = np.sqrt(np.square(window).mean())
        corr = window.div(vols, axis=1).cov().to_numpy()
        factor = np.linalg.cholesky(corr + 1e-14 * np.eye(len(ASSETS)))
        factor = pd.DataFrame(
            np.diag(vols.to_numpy()) @ factor,
            index=ASSETS,
            columns=factors,
        )
        residual = pd.Series(1e-7, index=ASSETS, dtype=float)
        fs[date] = factor.reindex(index=asset_index, columns=factors, fill_value=0.0)
        d_halves[date] = residual.reindex(index=asset_index, fill_value=0.0)
    return {"Fs": fs, "D_halves": d_halves}


def sliced(result: BacktestResults, start: pd.Timestamp, end: pd.Timestamp) -> BacktestResults:
    """Slice a result to an evaluation interval."""
    return BacktestResults(
        navs=result.navs.loc[start:end],
        composition=result.composition.loc[start:end],
        turnover=result.turnover.loc[start:end],
        metadata=result.metadata,
    )


def score(
    result: BacktestResults,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ffr: pd.Series,
    cpi: pd.Series,
) -> float:
    """Compute the paper's FFR Sharpe over one parameter-selection window."""
    return float(compute_metrics(sliced(result, start, end), ffr, cpi)["Sharpe Ratio (FFR)"])


def main() -> None:
    """Run candidate backtests, rolling selections, plots, and summary tables."""
    print("Started walk-forward sensitivity analysis...", flush=True)
    out_dir = Path("output/tables/ridge_stability")
    out_dir.mkdir(parents=True, exist_ok=True)
    closes = pd.read_parquet("data/raw/closes.parquet")
    returns = closes.pct_change().dropna(how="all")
    ext_features = pd.read_parquet("data/processed/ext_features.parquet")
    etf_features = pd.read_parquet("data/processed/etf_features.parquet")
    etf_features = etf_features[
        [col for col in etf_features if any(f"_{asset}" in col for asset in ASSETS)]
    ]
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0
    with Path("data/processed/risk_model/sbg.pkl").open("rb") as file:
        production_risk = pickle.load(file)  # noqa: S301

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

    risk_models = {
        11: production_risk,
        5: covariance_risk_model(returns, 5),
        22: covariance_risk_model(returns, 22),
    }
    alpha_keys = {
        (params["horizon"], params["alpha_halflife"], params["ridge"]) for params in SPECS.values()
    }
    alphas = {}
    print("Started fitting alpha models...", flush=True)
    for horizon, halflife, ridge in sorted(alpha_keys):
        print(f"Fitting alpha h={horizon}, halflife={halflife}, ridge={ridge}", flush=True)
        alpha = get_ridge_alpha_over_time(
            closes,
            ext_features,
            etf_features,
            ASSETS,
            h=horizon,
            min_history=512,
            halflife=halflife,
            ridge=ridge,
            min_window=63,
            verbose=False,
        )
        alphas[(horizon, halflife, ridge)] = alpha.reindex(
            index=closes.index, columns=closes.columns, fill_value=0.0
        ).fillna(0.0)
    print("Done.", flush=True)

    constructors = {}
    results = {}
    print("Started running sensitivity backtests...", flush=True)
    for label, params in SPECS.items():
        alpha_key = (params["horizon"], params["alpha_halflife"], params["ridge"])
        constructors[label] = AnchoredVolControlPortfolioConstructor(
            ts_lookup=lookup,
            alphas=alphas[alpha_key].loc[data.timeline].to_numpy() * (21 / 252),
            risk_model=risk_models[params["cov_window"]],
            vol_target=VOL_TARGET,
            anchor=anchor,
            universe=universe,
            leverage=1.0,
            bid_ask_spread=5e-4,
            cash_rate_horizon_days=21,
        )
        results[label] = run_backtest(label, data, constructors[label])
    print("Done.", flush=True)

    # Fixed-start, varying-end-date Sharpe and its one-opt hindsight envelope.
    print("Started evaluating walk-forward stability...", flush=True)
    end_rows = []
    for end_year in range(2011, 2026):
        end = pd.Timestamp(f"{end_year}-12-31")
        sharpes = {
            label: score(result, EVAL_START, end, ffr, cpi) for label, result in results.items()
        }
        best = max(sharpes, key=sharpes.get)
        end_rows.append(
            {
                "End year": end_year,
                "Current Sharpe": sharpes["baseline"],
                "Best 1-opt Sharpe": sharpes[best],
                "Best 1-opt specification": best,
            }
        )
    end_table = pd.DataFrame(end_rows).set_index("End year")
    end_table.to_csv(out_dir / "end_time_sharpe.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(end_table.index, end_table["Current Sharpe"], marker="o", label="Current parameters")
    ax.plot(
        end_table.index,
        end_table["Best 1-opt Sharpe"],
        linestyle="--",
        color="#777777",
        label="Best 1-opt at each end date",
    )
    ax.set_xlabel("Backtest end year")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "end_time_sharpe.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "end_time_sharpe.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    selection_rows = []
    coordinate_rows = []
    walkforward_rows = []
    for lookback in (5, 10):
        selections = {}
        first_year = 2006 + lookback
        for deploy_year in range(first_year, 2026):
            train_start = pd.Timestamp(f"{deploy_year - lookback}-01-01")
            train_end = pd.Timestamp(f"{deploy_year - 1}-12-31")
            sharpes = {
                label: score(result, train_start, train_end, ffr, cpi)
                for label, result in results.items()
            }
            best = max(sharpes, key=sharpes.get)
            selections[deploy_year] = best
            selection_rows.append(
                {
                    "Lookback years": lookback,
                    "Deployment year": deploy_year,
                    "Selected specification": best,
                    "Training Sharpe": sharpes[best],
                    "Baseline training Sharpe": sharpes["baseline"],
                }
            )
            for coordinate, labels in COORDINATE_GROUPS.items():
                coordinate_best = max(labels, key=lambda label: sharpes[label])
                coordinate_rows.append(
                    {
                        "Lookback years": lookback,
                        "Deployment year": deploy_year,
                        "Coordinate": coordinate,
                        "Selected specification": coordinate_best,
                        "Selected value": SPECS[coordinate_best][coordinate],
                        "Current value": BASE_PARAMS[coordinate],
                        "Training Sharpe": sharpes[coordinate_best],
                    }
                )

        switching = SwitchingConstructor(constructors, selections)
        walkforward = run_backtest(f"walk-forward {lookback}y", data, switching)
        evaluation_start = pd.Timestamp(f"{first_year}-01-01")
        for label, result in {
            "Current parameters": results["baseline"],
            "Walk-forward 1-opt": walkforward,
        }.items():
            metrics = compute_metrics(sliced(result, evaluation_start, END_DATE), ffr, cpi)[METRICS]
            walkforward_rows.append(
                {"Lookback years": lookback, "Strategy": label} | metrics.to_dict()
            )

    selections_df = pd.DataFrame(selection_rows)
    selections_df.to_csv(out_dir / "walkforward_selections.csv", index=False)
    coordinates_df = pd.DataFrame(coordinate_rows)
    coordinates_df.to_csv(out_dir / "coordinate_choices.csv", index=False)
    walkforward_df = pd.DataFrame(walkforward_rows)
    walkforward_df.to_csv(out_dir / "walkforward_performance.csv", index=False)

    frequencies = (
        coordinates_df.assign(
            Current=coordinates_df["Selected value"] == coordinates_df["Current value"]
        )
        .groupby(["Lookback years", "Coordinate"])
        .agg(
            Evaluations=("Current", "size"),
            Current_selected=("Current", "sum"),
            Current_fraction=("Current", "mean"),
            Median_selected_value=("Selected value", "median"),
        )
    )
    frequencies.to_csv(out_dir / "coordinate_stability_summary.csv")

    print("\nEnd-time Sharpe")
    print(end_table.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nCoordinate stability")
    print(frequencies.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nWalk-forward performance")
    print(walkforward_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Done.", flush=True)


main()
