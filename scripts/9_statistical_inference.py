"""Statistical inference for the paper: conventional Sharpe, bootstrap CIs, deflated Sharpe.

Produces the numbers a referee would ask for, all as post-processing of the existing
backtest NAV series -- no strategy is re-run and no methodology changes.

Outputs (to ``output/tables/inference/``):

* ``sharpe_definitions.csv``  -- the paper's geometric (excess-CAGR / vol) Sharpe next to the
  conventional arithmetic (mean excess / stdev) Sharpe.
* ``bootstrap_metrics.csv``   -- stationary-block-bootstrap 95% confidence intervals for CAGR,
  volatility, Sharpe, and maximum drawdown.
* ``sharpe_differences.csv``  -- paired bootstrap confidence intervals and one-sided p-values for
  Sharpe differences against the Markowitz portfolio.
* ``subperiods.csv``          -- annualized return, volatility, and Sharpe by subperiod.
* ``deflated_sharpe.csv``     -- probabilistic and deflated Sharpe ratios for the Markowitz
  portfolio, accounting for the documented specification search.

The bootstrap is a stationary bootstrap (Politis and Romano) with mean block length 21 trading
days, resampled once per replication and applied to every portfolio with the SAME index, so that
Sharpe differences are paired and the cross-sectional correlation of the portfolios is preserved.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from simple_portfolio.simulator import BacktestResults

START, END = "2006-01-01", "2026-01-01"
RESULTS_DIR = Path("results")
OUT_DIR = Path("output/tables/inference")
MEAN_BLOCK = 21
N_BOOT = 10_000
SEED = 0
EULER = 0.5772156649015329

PORTFOLIOS = {
    "sbg_alpha_vc": "Markowitz",
    "sbg_simple_alpha_vc": "Simple Markowitz",
    "sbg_503020_vc": "50/30/20 VC",
    "sbg_6040_vc": "60/40 VC",
    "sbg_503020": "50/30/20",
    "sbg_6040": "60/40",
    "gold": "GLD",
    "bond": "AGG",
    "snp": "SPY",
}
SUBPERIODS = {
    "2006--2026": (START, END),
    "2006--2011": ("2006-01-01", "2011-01-01"),
    "2011--2016": ("2011-01-01", "2016-01-01"),
    "2016--2021": ("2016-01-01", "2021-01-01"),
    "2021--2026": ("2021-01-01", "2026-01-01"),
    "2006--2016": ("2006-01-01", "2016-01-01"),
    "2016--2026": ("2016-01-01", "2026-01-01"),
    "2011--2026": ("2011-01-01", "2026-01-01"),
}


def geometric_sharpe(returns: np.ndarray, rf: np.ndarray) -> float:
    """Excess CAGR over compounded cash, divided by annualized volatility (the paper's metric)."""
    n = len(returns)
    growth = np.prod(1.0 + returns)
    growth_rf = np.prod(1.0 + rf)
    return ((growth / growth_rf) ** (252.0 / n) - 1.0) / (returns.std(ddof=1) * np.sqrt(252.0))


def arithmetic_sharpe(returns: np.ndarray, rf: np.ndarray) -> float:
    """Conventional Sharpe: mean daily excess return over its standard deviation, annualized."""
    excess = returns - rf
    return np.sqrt(252.0) * excess.mean() / returns.std(ddof=1)


def cagr(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) ** (252.0 / len(returns)) - 1.0)


def max_drawdown(returns: np.ndarray) -> float:
    navs = np.cumprod(1.0 + returns)
    return float((1.0 - navs / np.maximum.accumulate(navs)).max())


def stationary_bootstrap_index(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis--Romano stationary bootstrap indices of length n."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    restart = rng.random(n) < p
    steps = rng.integers(0, n, size=n)
    for i in range(1, n):
        idx[i] = steps[i] if restart[i] else (idx[i - 1] + 1) % n
    return idx


def load_returns() -> tuple[pd.DataFrame, pd.Series]:
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    navs = {}
    for stem, label in PORTFOLIOS.items():
        navs[label] = BacktestResults.load(RESULTS_DIR / f"{stem}.pkl").navs.loc[START:END]
    returns = pd.DataFrame(navs).pct_change().dropna()
    return returns, ffr.loc[returns.index]


def main() -> None:
    print("Started statistical inference...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns, rf = load_returns()
    labels = list(returns.columns)
    r = {c: returns[c].to_numpy() for c in labels}
    f = rf.to_numpy()

    # ── 1. Sharpe definitions ────────────────────────────────────────────────
    definitions = pd.DataFrame(
        {
            c: {
                "Geometric (paper)": geometric_sharpe(r[c], f),
                "Arithmetic (conventional)": arithmetic_sharpe(r[c], f),
                "Volatility": returns[c].std(ddof=1) * np.sqrt(252.0),
            }
            for c in labels
        }
    ).T
    definitions.to_csv(OUT_DIR / "sharpe_definitions.csv")
    print("\n=== Sharpe ratio, paper's geometric definition vs conventional arithmetic")
    print(definitions.to_string(float_format=lambda v: f"{v:.3f}"))

    # ── 2. Stationary block bootstrap ────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    n = len(returns)
    draws = {
        c: {k: np.empty(N_BOOT) for k in ("cagr", "vol", "geo", "arith", "maxdd")} for c in labels
    }
    for b in range(N_BOOT):
        idx = stationary_bootstrap_index(n, MEAN_BLOCK, rng)
        f_b = f[idx]
        for c in labels:
            r_b = r[c][idx]
            draws[c]["cagr"][b] = cagr(r_b)
            draws[c]["vol"][b] = r_b.std(ddof=1) * np.sqrt(252.0)
            draws[c]["geo"][b] = geometric_sharpe(r_b, f_b)
            draws[c]["arith"][b] = arithmetic_sharpe(r_b, f_b)
            draws[c]["maxdd"][b] = max_drawdown(r_b)

    rows = {}
    for c in labels:
        point = {
            "cagr": cagr(r[c]),
            "vol": returns[c].std(ddof=1) * np.sqrt(252.0),
            "geo": geometric_sharpe(r[c], f),
            "arith": arithmetic_sharpe(r[c], f),
            "maxdd": max_drawdown(r[c]),
        }
        entry = {}
        for k, v in point.items():
            lo, hi = np.percentile(draws[c][k], [2.5, 97.5])
            entry[f"{k}"] = v
            entry[f"{k}_lo"] = lo
            entry[f"{k}_hi"] = hi
        rows[c] = entry
    bootstrap = pd.DataFrame(rows).T
    bootstrap.to_csv(OUT_DIR / "bootstrap_metrics.csv")
    print(f"\n=== Stationary bootstrap 95% CIs (mean block {MEAN_BLOCK}d, {N_BOOT} reps)")
    for c in labels:
        e = rows[c]
        print(
            f"{c:<18} CAGR {e['cagr']:6.2%} [{e['cagr_lo']:6.2%},{e['cagr_hi']:6.2%}]   "
            f"Sharpe(geo) {e['geo']:.2f} [{e['geo_lo']:.2f},{e['geo_hi']:.2f}]   "
            f"maxDD {e['maxdd']:6.2%} [{e['maxdd_lo']:6.2%},{e['maxdd_hi']:6.2%}]"
        )

    # ── 3. Paired Sharpe differences against Markowitz ───────────────────────
    base = "Markowitz"
    diff_rows = {}
    for c in labels:
        if c == base:
            continue
        for key, fn in (("geo", geometric_sharpe), ("arith", arithmetic_sharpe)):
            d = draws[base][key] - draws[c][key]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff_rows[(c, key)] = {
                "difference": fn(r[base], f) - fn(r[c], f),
                "lo": lo,
                "hi": hi,
                "p_le_0": float((d <= 0.0).mean()),
            }
    differences = pd.DataFrame(diff_rows).T
    differences.index.names = ["Portfolio", "Definition"]
    differences.to_csv(OUT_DIR / "sharpe_differences.csv")
    print("\n=== Sharpe differences, Markowitz minus each portfolio (paired bootstrap)")
    print(differences.to_string(float_format=lambda v: f"{v:.3f}"))

    # ── 4. Subperiods ────────────────────────────────────────────────────────
    sub_rows = {}
    for label, (a, b) in SUBPERIODS.items():
        window = returns.loc[a:b]
        f_w = rf.loc[window.index].to_numpy()
        for c in labels:
            r_w = window[c].to_numpy()
            sub_rows[(label, c)] = {
                "Return": cagr(r_w),
                "Volatility": r_w.std(ddof=1) * np.sqrt(252.0),
                "Sharpe (geo)": geometric_sharpe(r_w, f_w),
                "Sharpe (arith)": arithmetic_sharpe(r_w, f_w),
                "Max Drawdown": max_drawdown(r_w),
            }
    subperiods = pd.DataFrame(sub_rows).T
    subperiods.index.names = ["Period", "Portfolio"]
    subperiods.to_csv(OUT_DIR / "subperiods.csv")
    print("\n=== Subperiod Sharpe (geometric)")
    print(
        subperiods["Sharpe (geo)"]
        .unstack("Portfolio")
        .reindex(index=list(SUBPERIODS), columns=labels)
        .to_string(float_format=lambda v: f"{v:.2f}")
    )

    # ── 5. Probabilistic and deflated Sharpe for the Markowitz portfolio ─────
    # Bailey and Lopez de Prado. Everything in per-observation (daily) units.
    excess = r[base] - f
    sr = excess.mean() / r[base].std(ddof=1)
    skew = float(pd.Series(excess).skew())
    kurt = float(pd.Series(excess).kurtosis()) + 3.0  # non-excess kurtosis

    sweep = pd.read_csv("output/tables/ridge_hyperparameter_sweep_full.csv")
    trial_sharpes = sweep["Sharpe Ratio (FFR)"].to_numpy() / np.sqrt(252.0)
    n_trials = len(trial_sharpes)
    sr_var = float(trial_sharpes.var(ddof=1))

    sr0 = np.sqrt(sr_var) * (
        (1.0 - EULER) * norm.ppf(1.0 - 1.0 / n_trials)
        + EULER * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )

    def psr(benchmark: float) -> float:
        num = (sr - benchmark) * np.sqrt(n - 1.0)
        den = np.sqrt(1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2)
        return float(norm.cdf(num / den))

    deflated = pd.Series(
        {
            "Annualized Sharpe": sr * np.sqrt(252.0),
            "Daily Sharpe": sr,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Observations": n,
            "Documented trials": n_trials,
            "Trial Sharpe stdev (annualized)": np.sqrt(sr_var) * np.sqrt(252.0),
            "Expected max Sharpe under null (annualized)": sr0 * np.sqrt(252.0),
            "PSR vs 0": psr(0.0),
            "Deflated Sharpe (PSR vs expected max)": psr(sr0),
        }
    )
    deflated.to_csv(OUT_DIR / "deflated_sharpe.csv")
    print(f"\n=== Probabilistic / deflated Sharpe for {base}")
    print(deflated.to_string(float_format=lambda v: f"{v:.4f}"))
    print(
        "\nNote: the deflated Sharpe uses the "
        f"{n_trials} documented sweep specifications as the trial count, which is a lower bound "
        "on the true number of specifications considered."
    )
    print("Done.", flush=True)


main()
