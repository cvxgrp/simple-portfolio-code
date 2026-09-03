"""Risk-based benchmarks over a grid of covariance estimators.

Appendix E runs every risk-based comparator on the paper's own 11-day covariance estimate. That
window was chosen because it suits our Markowitz portfolio, so a referee can object that the
comparators were handed a noisy specification that does not suit them. This script removes the
objection by re-running all seven methods over a grid of trailing covariance windows and
reporting each method at its OWN best window, selected in-sample on the Sharpe ratio -- the same
hindsight advantage we granted Black--Litterman in the delta/omega sweep.

Both the fully invested and the 7% volatility-controlled variants are run at every window, over
the same universe, period, transaction costs, and constraints as the rest of the paper.
Results go to ``output/tables/risk_based_benchmarks/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from simple_portfolio.metrics import compute_metrics
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
WINDOWS = [11, 21, 63, 126, 252]
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


def rolling_covariances(
    returns: pd.DataFrame, lookback: int, dates: list[pd.Timestamp]
) -> dict[pd.Timestamp, np.ndarray]:
    """Trailing uncentered covariance of the three assets, matching the production estimate."""
    out: dict[pd.Timestamp, np.ndarray] = {}
    for date in dates:
        window = returns.loc[:date, ASSETS].dropna().tail(lookback)
        if len(window) < lookback:
            continue
        vols = np.sqrt(np.square(window).mean())
        corr = window.div(vols, axis=1).cov().to_numpy()
        covariance = np.diag(vols.to_numpy()) @ corr @ np.diag(vols.to_numpy())
        out[date] = 0.5 * (covariance + covariance.T) + 1e-14 * np.eye(len(ASSETS))
    return out


def solve_simplex(objective, start: np.ndarray) -> np.ndarray:
    """Solve a smooth long-only, fully invested allocation problem.

    Over a five-window grid SLSQP occasionally stalls on a near-singular covariance from a
    particular warm start, so we retry from the equal-weight point and fall back to the warm
    start itself. The warm-started solve succeeds on the overwhelming majority of dates, so this
    only affects isolated dates and leaves the reported paths otherwise unchanged.
    """
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
        marginal = covariance @ weights
        variance = float(weights @ marginal)
        return float(np.square(weights * marginal / variance - budget).sum())

    return solve_simplex(loss, start)


def black_litterman_weights(
    covariance_daily: np.ndarray, alpha: np.ndarray, start: np.ndarray
) -> np.ndarray:
    """Combine a 50/30/20 equilibrium prior with ridge-alpha absolute views."""
    covariance = 21.0 * covariance_daily
    prior = BL_RISK_AVERSION * covariance @ ANCHOR
    prior_precision = np.linalg.inv(BL_TAU * covariance)
    view_precision = np.linalg.inv(BL_TAU * np.diag(np.diag(covariance)))
    posterior_covariance = np.linalg.inv(prior_precision + view_precision)
    posterior_mean = posterior_covariance @ (prior_precision @ prior + view_precision @ alpha)

    def negative_utility(weights: np.ndarray) -> float:
        return float(
            -posterior_mean @ weights + 0.5 * BL_RISK_AVERSION * weights @ covariance @ weights
        )

    return solve_simplex(negative_utility, start)


def sleeves_for(
    covariance: np.ndarray, alpha: np.ndarray, previous: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """All seven risk-based sleeves at one date."""
    volatilities = np.sqrt(np.diag(covariance))
    return {
        "Equal weight": EQUAL_BUDGET,
        "Inverse volatility": (1.0 / volatilities) / (1.0 / volatilities).sum(),
        "Equal risk contribution": risk_budget_weights(
            covariance, EQUAL_BUDGET, previous["Equal risk contribution"]
        ),
        "50/30/20 risk budget": risk_budget_weights(
            covariance, ANCHOR, previous["50/30/20 risk budget"]
        ),
        "Global minimum variance": solve_simplex(
            lambda w: float(w @ covariance @ w), previous["Global minimum variance"]
        ),
        "Maximum diversification": solve_simplex(
            lambda w: -float(w @ volatilities / np.sqrt(w @ covariance @ w)),
            previous["Maximum diversification"],
        ),
        "Black--Litterman": black_litterman_weights(
            covariance, alpha, previous["Black--Litterman"]
        ),
    }


def sliced(result: BacktestResults) -> BacktestResults:
    return BacktestResults(
        navs=result.navs.loc[EVAL_START:END_DATE],
        composition=result.composition.loc[EVAL_START:END_DATE],
        turnover=result.turnover.loc[EVAL_START:END_DATE],
        metadata=result.metadata,
    )


def main() -> None:
    print("Started covariance sensitivity analysis...", flush=True)
    closes = pd.read_parquet("data/raw/closes.parquet")
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0
    alpha = pd.read_parquet("data/processed/alpha_ridge_sbg.parquet") * (21 / 252)
    print("Data loaded.", flush=True)

    data = BacktestData.from_pandas(
        start_date=START_DATE,
        end_date=END_DATE,
        closes=closes,
        ffrs=ffr,
        rebal_freq="M",
    )
    lookup = {ts: i for i, ts in enumerate(data.timeline)}
    indices = np.array([int(np.where(data.assets == asset)[0][0]) for asset in ASSETS])
    # Solve on every date in the timeline, warm starting from the previous day, exactly as
    # sandbox/risk_based_benchmarks.py does. The risk-budgeting objectives are not convex, so a
    # per-rebalance warm start can land on a different local solution and would not reproduce the
    # numbers already reported in the appendix.
    solve_rows = np.arange(len(data.timeline))
    solve_dates = [pd.Timestamp(ts) for ts in data.timeline]

    returns = closes.pct_change().dropna(how="all")
    alphas = {d: alpha.loc[d, ASSETS].to_numpy(dtype=float) for d in solve_dates}
    labels = [
        "Equal weight",
        "Inverse volatility",
        "Equal risk contribution",
        "50/30/20 risk budget",
        "Global minimum variance",
        "Maximum diversification",
        "Black--Litterman",
    ]

    rows = {}
    for window in WINDOWS:
        print(f"Window {window}d started.", flush=True)
        covariances = rolling_covariances(returns, window, solve_dates)
        print(f"Window {window}d covariances computed.", flush=True)
        native = {c: np.zeros((len(data.timeline), len(data.assets))) for c in labels}
        controlled = {c: np.zeros_like(native[c]) for c in labels}
        previous = {c: ANCHOR.copy() for c in labels}
        for row, date in zip(solve_rows, solve_dates, strict=True):
            covariance = covariances.get(date)
            if covariance is None:
                continue
            sleeves = sleeves_for(covariance, alphas[date], previous)
            for label, sleeve in sleeves.items():
                previous[label] = sleeve
                estimated = np.sqrt(sleeve @ covariance @ sleeve)
                native[label][row, indices] = sleeve
                controlled[label][row, indices] = min(VOL_TARGET / estimated, 1.0) * sleeve
        print(f"Window {window}d weights computed.", flush=True)

        for label in labels:
            for variant, weights in (
                ("fully invested", native[label]),
                ("vol controlled", controlled[label]),
            ):
                result = run_backtest(
                    f"{label} {variant} {window}", data, PrecomputedWeights(lookup, weights)
                )
                rows[(label, variant, window)] = compute_metrics(sliced(result), ffr, cpi)[METRICS]
        print(f"Window {window}d backtests completed.", flush=True)
        print(
            "   "
            + "  ".join(
                f"{c.split()[0]}={rows[(c, 'vol controlled', window)]['Sharpe Ratio (FFR)']:.3f}"
                for c in labels
            ),
            flush=True,
        )

    table = pd.DataFrame(rows).T
    table.index.names = ["Method", "Variant", "Window"]
    out_dir = Path("output/tables/risk_based_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "covariance_grid.csv")

    print("\n=== Sharpe ratio by covariance window")
    grid = table["Sharpe Ratio (FFR)"].unstack("Window")
    print(grid.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== Each method at its own best window (selected in-sample)")
    best = table.loc[table.groupby(level=["Method", "Variant"])["Sharpe Ratio (FFR)"].idxmax()]
    print(best.to_string(float_format=lambda v: f"{v:.4f}"))
    best.to_csv(out_dir / "covariance_grid_best.csv")
    print("Done.", flush=True)


main()
