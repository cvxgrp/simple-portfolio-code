"""Risk-based and Black--Litterman benchmarks for the SPY/AGG/GLD portfolio.

This is a scratch experiment only. Each risky sleeve is re-estimated monthly
from the production covariance matrix. We report both fully invested and 7%
volatility-controlled (no leverage) variants with production transaction costs.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from simple_portfolio.metrics import compute_metrics
from simple_portfolio.optimizer import AnchoredVolControlPortfolioConstructor
from simple_portfolio.simulator import BacktestData, BacktestResults, run_backtest

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2026-01-01")
EVAL_START = pd.Timestamp("2006-01-01")
ASSETS = ["SPY", "AGG", "GLD"]
ANCHOR = np.array([0.5, 0.3, 0.2])
EQUAL_BUDGET = np.full(3, 1.0 / 3.0)
VOL_TARGET = 0.07 / np.sqrt(252)
BL_RISK_AVERSION = 2.5
BL_TAU = 0.05
METRICS = [
    "Return",
    "Volatility",
    "Sharpe Ratio (FFR)",
    "Max Drawdown",
    "Avg. Drawdown",
    "Turnover",
]


class PrecomputedWeights:
    """Return a precomputed target vector without renormalizing away cash."""

    def __init__(self, lookup: dict[pd.Timestamp, int], weights: np.ndarray) -> None:
        self.lookup = lookup
        self.weights = weights

    def __call__(self, ts, curr_weights, universe, ffr):
        return self.weights[self.lookup[ts]] * universe


def covariance_at(risk_model: dict, date: pd.Timestamp, indices: np.ndarray) -> np.ndarray:
    """Recover the risky-asset covariance matrix from the factor representation."""
    factor = np.asarray(risk_model["Fs"][date], dtype=float)[indices]
    residual_half = np.asarray(risk_model["D_halves"][date], dtype=float)[indices]
    covariance = factor @ factor.T + np.diag(residual_half**2)
    return 0.5 * (covariance + covariance.T) + 1e-14 * np.eye(len(indices))


def solve_simplex(objective, start: np.ndarray) -> np.ndarray:
    """Solve a smooth long-only, fully invested allocation problem."""
    for guess in (start, EQUAL_BUDGET):
        result = minimize(
            objective,
            guess,
            method="SLSQP",
            bounds=[(1e-8, 1.0)] * len(guess),
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success:
            weights = np.maximum(result.x, 0.0)
            return weights / weights.sum()
    weights = np.maximum(np.asarray(start, dtype=float), 0.0)
    return weights / weights.sum()


def risk_budget_weights(
    covariance: np.ndarray, budget: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """Match fractional volatility contributions to the requested risk budget."""

    def loss(weights: np.ndarray) -> float:
        marginal_variance = covariance @ weights
        variance = float(weights @ marginal_variance)
        fractions = weights * marginal_variance / variance
        return float(np.square(fractions - budget).sum())

    return solve_simplex(loss, start)


def black_litterman_weights(
    covariance_daily: np.ndarray, alpha: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """Combine a 50/30/20 equilibrium prior with ridge-alpha absolute views."""
    covariance = 21.0 * covariance_daily
    prior = BL_RISK_AVERSION * covariance @ ANCHOR
    prior_covariance = BL_TAU * covariance
    view_covariance = BL_TAU * np.diag(np.diag(covariance))
    prior_precision = np.linalg.inv(prior_covariance)
    view_precision = np.linalg.inv(view_covariance)
    posterior_covariance = np.linalg.inv(prior_precision + view_precision)
    posterior_mean = posterior_covariance @ (prior_precision @ prior + view_precision @ alpha)

    def negative_utility(weights: np.ndarray) -> float:
        return float(
            -posterior_mean @ weights + 0.5 * BL_RISK_AVERSION * weights @ covariance @ weights
        )

    return solve_simplex(negative_utility, start)


def sliced(result: BacktestResults) -> BacktestResults:
    return BacktestResults(
        navs=result.navs.loc[EVAL_START:END_DATE],
        composition=result.composition.loc[EVAL_START:END_DATE],
        turnover=result.turnover.loc[EVAL_START:END_DATE],
        metadata=result.metadata,
    )


def main() -> None:
    print("Started risk-based benchmark analysis...", flush=True)
    closes = pd.read_parquet("data/raw/closes.parquet")
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0
    alpha = pd.read_parquet("data/processed/alpha_ridge_sbg.parquet") * (21 / 252)
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
    indices = np.array([int(np.where(data.assets == asset)[0][0]) for asset in ASSETS])
    universe = np.zeros(len(data.assets), dtype=bool)
    universe[indices] = True
    anchor_full = np.zeros(len(data.assets))
    anchor_full[indices] = ANCHOR

    labels = [
        "Equal weight",
        "Inverse volatility",
        "Equal risk contribution",
        "50/30/20 risk budget",
        "Global minimum variance",
        "Maximum diversification",
        "Black-Litterman (ridge views)",
    ]
    native = {label: np.zeros((len(data.timeline), len(data.assets))) for label in labels}
    controlled = {label: np.zeros_like(native[label]) for label in labels}
    previous = {label: ANCHOR.copy() for label in labels}

    print("Started computing benchmark weights...", flush=True)
    for row, date_value in enumerate(data.timeline):
        date = pd.Timestamp(date_value)
        covariance = covariance_at(risk_model, date, indices)
        volatilities = np.sqrt(np.diag(covariance))
        alpha_today = alpha.loc[date, ASSETS].to_numpy(dtype=float)

        sleeves = {
            "Equal weight": EQUAL_BUDGET,
            "Inverse volatility": (1.0 / volatilities) / (1.0 / volatilities).sum(),
            "Equal risk contribution": risk_budget_weights(
                covariance, EQUAL_BUDGET, previous["Equal risk contribution"]
            ),
            "50/30/20 risk budget": risk_budget_weights(
                covariance, ANCHOR, previous["50/30/20 risk budget"]
            ),
            "Global minimum variance": solve_simplex(
                lambda weights, covariance=covariance: float(weights @ covariance @ weights),
                previous["Global minimum variance"],
            ),
            "Maximum diversification": solve_simplex(
                lambda weights, covariance=covariance, volatilities=volatilities: (
                    -float(weights @ volatilities / np.sqrt(weights @ covariance @ weights))
                ),
                previous["Maximum diversification"],
            ),
            "Black-Litterman (ridge views)": black_litterman_weights(
                covariance, alpha_today, previous["Black-Litterman (ridge views)"]
            ),
        }
        for label, sleeve in sleeves.items():
            previous[label] = sleeve
            estimated_volatility = np.sqrt(sleeve @ covariance @ sleeve)
            scale = min(VOL_TARGET / estimated_volatility, 1.0)
            native[label][row, indices] = sleeve
            controlled[label][row, indices] = scale * sleeve
    print("Done.", flush=True)

    constructors = {}
    for label in labels:
        constructors[f"{label} (native)"] = PrecomputedWeights(lookup, native[label])
        constructors[f"{label} (VC)"] = PrecomputedWeights(lookup, controlled[label])

    # Production Markowitz reference: same ridge alpha, covariance, target,
    # long-only/no-leverage constraints, and 50/30/20 l1 trust region.
    constructors["Production Markowitz"] = AnchoredVolControlPortfolioConstructor(
        ts_lookup=lookup,
        alphas=alpha.loc[data.timeline].to_numpy(),
        risk_model=risk_model,
        vol_target=VOL_TARGET,
        anchor=anchor_full,
        universe=universe,
        leverage=1.0,
        bid_ask_spread=5e-4,
        cash_rate_horizon_days=21,
    )

    rows = {}
    weight_rows = {}
    out_dir = Path("output/tables/risk_based_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Started running benchmark backtests...", flush=True)
    for name, constructor in constructors.items():
        print(f"Backtesting {name}", flush=True)
        result = run_backtest(name, data, constructor)
        rows[name] = compute_metrics(sliced(result), ffr, cpi)[METRICS]
        result.save(out_dir / f"{name.lower().replace(' ', '_').replace('/', '_')}.pkl")
        weights = result.composition.loc[EVAL_START:END_DATE, ASSETS]
        weight_rows[name] = weights.mean().rename(name)
    print("Done.", flush=True)

    table = pd.DataFrame(rows).T
    table.to_csv(out_dir / "performance.csv")
    pd.DataFrame(weight_rows).T.to_csv(out_dir / "average_risky_weights.csv")
    print("\nPerformance")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nAverage risky weights")
    print(pd.DataFrame(weight_rows).T.to_string(float_format=lambda value: f"{value:.4f}"))
    print("Done.", flush=True)


main()
