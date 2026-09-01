"""Black--Litterman hyper-parameter sweep for the risk-based benchmark appendix.

The appendix reports Black--Litterman with risk aversion ``delta=2.5``, prior
covariance ``tau * Sigma``, and view covariance ``tau * diag(Sigma)``. In that
(canonical) specification ``tau`` cancels from the posterior mean, so it has no
effect on the portfolio; this script verifies that numerically and then sweeps
the two parameters that do matter:

* ``delta``  -- risk aversion, which scales both the equilibrium returns
  ``pi = delta * Sigma * w_tgt`` and the penalty in the mean-variance utility;
* ``omega``  -- a view-confidence multiplier, ``Omega = omega * tau * diag(Sigma)``.
  Small ``omega`` trusts the views (the ridge alpha), large ``omega`` shrinks the
  posterior back to the equilibrium prior.

Every grid point is run in both the fully invested and the 7% volatility-controlled
variant, over the same universe, period, covariance estimate, and transaction
costs as the rest of the paper. Results go to ``output/tables/risk_based_benchmarks/``.
"""

import pickle
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
VOL_TARGET = 0.07 / np.sqrt(252)
BL_TAU = 0.05
DELTAS = [0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0]
OMEGAS = [0.0001, 0.001, 0.01, 0.1, 0.25, 1.0, 4.0, 10.0, 100.0]
TAU_CHECK = [0.005, 0.05, 0.5]
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
    equal_weights = np.full(len(start), 1.0 / len(start))
    for guess in (start, equal_weights):
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


def black_litterman_weights(
    covariance_daily: np.ndarray,
    alpha: np.ndarray,
    start: np.ndarray,
    delta: float,
    omega: float,
    tau: float,
) -> np.ndarray:
    """Combine a 50/30/20 equilibrium prior with ridge-alpha absolute views."""
    covariance = 252.0 * covariance_daily
    prior = delta * covariance @ ANCHOR
    prior_precision = np.linalg.inv(tau * covariance)
    view_precision = np.linalg.inv(omega * tau * np.diag(np.diag(covariance)))
    posterior_covariance = np.linalg.inv(prior_precision + view_precision)
    posterior_mean = posterior_covariance @ (prior_precision @ prior + view_precision @ alpha)

    def negative_utility(weights: np.ndarray) -> float:
        return float(-posterior_mean @ weights + 0.5 * delta * weights @ covariance @ weights)

    return solve_simplex(negative_utility, start)


def sliced(result: BacktestResults) -> BacktestResults:
    return BacktestResults(
        navs=result.navs.loc[EVAL_START:END_DATE],
        composition=result.composition.loc[EVAL_START:END_DATE],
        turnover=result.turnover.loc[EVAL_START:END_DATE],
        metadata=result.metadata,
    )


def main() -> None:
    print("Started Black-Litterman sensitivity analysis...", flush=True)
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
    indices = np.array([int(np.where(data.assets == asset)[0][0]) for asset in ASSETS])

    # The simulator calls the constructor on rebalance days and on day 0 only.
    solve_rows = np.flatnonzero(data.rebal_schedule)
    solve_rows = np.union1d(solve_rows, [0])
    covariances = {
        row: covariance_at(risk_model, pd.Timestamp(data.timeline[row]), indices)
        for row in solve_rows
    }
    alphas = {
        row: alpha.loc[pd.Timestamp(data.timeline[row]), ASSETS].to_numpy(dtype=float)
        for row in solve_rows
    }

    def sleeves(delta: float, omega: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Fully invested and volatility-controlled weights on each solve date."""
        out = np.zeros((len(data.timeline), len(data.assets)))
        controlled = np.zeros_like(out)
        previous = ANCHOR.copy()
        for row in solve_rows:
            covariance = covariances[row]
            sleeve = black_litterman_weights(covariance, alphas[row], previous, delta, omega, tau)
            previous = sleeve
            estimated_volatility = np.sqrt(sleeve @ covariance @ sleeve)
            out[row, indices] = sleeve
            controlled[row, indices] = min(VOL_TARGET / estimated_volatility, 1.0) * sleeve
        return out, controlled

    def evaluate(name: str, weights: np.ndarray) -> pd.Series:
        result = run_backtest(name, data, PrecomputedWeights(lookup, weights))
        return compute_metrics(sliced(result), ffr, cpi)[METRICS]

    out_dir = Path("output/tables/risk_based_benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. tau invariance check at the reported delta.
    print("Started tau invariance check...", flush=True)
    tau_rows = {}
    for tau in TAU_CHECK:
        _, controlled = sleeves(2.5, 1.0, tau)
        tau_rows[f"tau={tau}"] = evaluate(f"bl_tau_{tau}", controlled)
    tau_table = pd.DataFrame(tau_rows).T
    print(tau_table.to_string(float_format=lambda value: f"{value:.6f}"))
    tau_table.to_csv(out_dir / "bl_tau_invariance.csv")
    print("Done.", flush=True)

    # 2. delta / omega sweep.
    rows = {}
    print("Started delta-omega sweep...", flush=True)
    for delta in DELTAS:
        for omega in OMEGAS:
            native, controlled = sleeves(delta, omega, BL_TAU)
            key = (delta, omega)
            rows[(*key, "fully invested")] = evaluate(f"bl_{delta}_{omega}_native", native)
            rows[(*key, "vol controlled")] = evaluate(f"bl_{delta}_{omega}_vc", controlled)
            print(
                f"delta={delta:<5} omega={omega:<6} "
                f"VC Sharpe={rows[(*key, 'vol controlled')]['Sharpe Ratio (FFR)']:.3f}  "
                f"FI Sharpe={rows[(*key, 'fully invested')]['Sharpe Ratio (FFR)']:.3f}",
                flush=True,
            )

    table = pd.DataFrame(rows).T
    table.index.names = ["delta", "omega", "variant"]
    table.to_csv(out_dir / "bl_sweep.csv")
    print("Done.", flush=True)

    print("\nBest by Sharpe ratio, each variant")
    best: dict[str, tuple[float, float]] = {}
    for variant in ("vol controlled", "fully invested"):
        sub = table.xs(variant, level="variant").sort_values("Sharpe Ratio (FFR)", ascending=False)
        best[variant] = sub.index[0]
        print(f"\n{variant}: {len(sub)} specifications, Sharpe ratio ranges ")
        print(f"  {sub['Sharpe Ratio (FFR)'].min():.3f} to {sub['Sharpe Ratio (FFR)'].max():.3f}")
        print(sub.head(5).to_string(float_format=lambda value: f"{value:.4f}"))

    # Average relative weights of the best volatility-controlled specification.
    delta, omega = best["vol controlled"]
    _, controlled = sleeves(delta, omega, BL_TAU)
    result = run_backtest("bl_best_vc", data, PrecomputedWeights(lookup, controlled))
    composition = result.composition.loc[EVAL_START:END_DATE, ASSETS]
    gross = composition.sum(axis=1)
    relative = composition.div(gross, axis=0)[gross > 1e-8]
    print(f"\nBest volatility-controlled specification: delta={delta}, omega={omega}")
    print("Average relative weights:")
    print(relative.mean().to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"Average cash weight: {1.0 - gross.mean():.4f}")
    print("Done.", flush=True)


main()
