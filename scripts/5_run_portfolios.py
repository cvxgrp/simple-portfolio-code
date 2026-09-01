"""Build base portfolios."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from simple_portfolio.optimizer import (
    AnchoredAlphaPortfolioConstructor,
    AnchoredVolControlPortfolioConstructor,
    FixedWeightMatrixVolControlPortfolioConstructor,
    FixedWeightPortfolioConstructor,
    FixedWeightVolControlPortfolioConstructor,
)
from simple_portfolio.simulator import BacktestData, run_backtest

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2026-01-01")
LEVERAGE = 1.0
STOCK_VOL_TARGET = 0.10 / np.sqrt(252)
BOND_VOL_TARGET = 0.05 / np.sqrt(252)
GOLD_VOL_TARGET = 0.14 / np.sqrt(252)
SBG_6040_VOL_TARGET = 0.07 / np.sqrt(252)
SBG_503020_VOL_TARGET = 0.07 / np.sqrt(252)
SBG_ALPHA_VOL_TARGET = 0.07 / np.sqrt(252)

# Capital-gains tax brackets as (long_term_rate, short_term_rate). Must match the
# TAX_BRACKETS list in scripts/4_metrics_and_plots.py. Bracket 1 is the zero-tax
# (vanilla) case, which is already produced by the untaxed backtests above, so the
# tax-bracket loop below only re-runs the latter three brackets.
TAX_BRACKETS: list[tuple[float, float]] = [
    (0.0, 0.0),  # B1: Single 0 - 50k
    (0.15, 0.22),  # B2: Single 50k - 100k
    (0.15, 0.35),  # B3: Single 256k - 545k
    (0.2, 0.37),  # B4: Single 640k +
]

# The net investment income tax is a 3.8% surtax on net investment income above a
# modified AGI of $200k for a single filer, so it applies in B3 and B4 but not in B2.
NIIT_RATE = 0.038
NIIT_APPLIES = [False, False, True, True]
# Long-term gains on a grantor trust holding physical metal (GLD) are collectibles
# gains, taxed at ordinary rates capped at 28%, not at the 15%/20% rate for stock.
COLLECTIBLES_CAP = 0.28

# ── Preflight check ───────────────────────────────────────────────────────────

_sbg = Path("data/processed/risk_model/sbg.pkl")
if not _sbg.exists():
    raise FileNotFoundError(
        f"{_sbg} not found. Run scripts/1_download_data.py first to generate it."
    )

_alpha_sbg = Path("data/processed/alpha_ridge_sbg.parquet")
if not _alpha_sbg.exists():
    raise FileNotFoundError(
        f"{_alpha_sbg} not found. Run scripts/2_generate_alphas.py first to generate it."
    )
# with _full_k8.open("rb") as f:
#     risk_model_full = pickle.load(f)

with _sbg.open("rb") as f:
    risk_model_sbg = pickle.load(f)  # noqa: S301

proc_data_dir = Path("data/processed")
proc_data_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
closes = pd.read_parquet("data/raw/closes.parquet")
ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]

weekly_data = BacktestData.from_pandas(
    start_date=START_DATE,
    end_date=END_DATE,
    closes=closes,
    ffrs=ffr,
    rebal_freq="W",
)

monthly_data = BacktestData.from_pandas(
    start_date=START_DATE,
    end_date=END_DATE,
    closes=closes,
    ffrs=ffr,
    rebal_freq="M",
)

quartely_data = BacktestData.from_pandas(
    start_date=START_DATE,
    end_date=END_DATE,
    closes=closes,
    ffrs=ffr,
    rebal_freq="Q",
)

yearly_data = BacktestData.from_pandas(
    start_date=START_DATE,
    end_date=END_DATE,
    closes=closes,
    ffrs=ffr,
    rebal_freq="Y",
)

lookup = {ts: i for i, ts in enumerate(monthly_data.timeline)}
n_time = len(monthly_data.timeline)
n_assets = len(monthly_data.assets)

# ── Universes and weights ─────────────────────────────────────────────────────

assets = closes.columns
spy_idx = np.where(monthly_data.assets == "SPY")[0][0]
agg_idx = np.where(monthly_data.assets == "AGG")[0][0]
gld_idx = np.where(monthly_data.assets == "GLD")[0][0]

snp_universe = assets == "SPY"
bond_universe = assets == "AGG"
gold_universe = assets == "GLD"
sbg_6040_universe = snp_universe | bond_universe
sbg_503020_universe = snp_universe | bond_universe | gold_universe

full_universe = np.ones_like(assets, dtype=bool)

equal_weights = np.ones((n_time, n_assets))

spy60_agg40_weights = np.zeros((n_time, n_assets))
spy60_agg40_weights[:, spy_idx] = 0.6
spy60_agg40_weights[:, agg_idx] = 0.4

spy50_agg30_gld20_weights = np.zeros((n_time, n_assets))
spy50_agg30_gld20_weights[:, spy_idx] = 0.5
spy50_agg30_gld20_weights[:, agg_idx] = 0.3
spy50_agg30_gld20_weights[:, gld_idx] = 0.2

# Average Markowitz weights (from table t-avg-weights), held as a static
# fixed-weight benchmark. The constructor renormalizes weights to sum to
# `leverage`, so a leverage equal to the risky-weight sum leaves the residual
# (9.3%) in cash.
MARKOWITZ_AVG_WEIGHTS = (0.417, 0.248, 0.242)  # SPY, AGG, GLD
MARKOWITZ_AVG_LEVERAGE = sum(MARKOWITZ_AVG_WEIGHTS)
markowitz_avg_weights = np.zeros((n_time, n_assets))
markowitz_avg_weights[:, spy_idx] = MARKOWITZ_AVG_WEIGHTS[0]
markowitz_avg_weights[:, agg_idx] = MARKOWITZ_AVG_WEIGHTS[1]
markowitz_avg_weights[:, gld_idx] = MARKOWITZ_AVG_WEIGHTS[2]

# Trailing window (trading days) for the volatility-controlled benchmarks'
# volatility estimate. Matches SBG_HALFLIFE in scripts/1_download_data.py: the
# same reactive-estimate argument applies to the scalar volatility estimate as
# to the covariance matrix used by the optimization-based portfolios.
sbg_window = 11
returns = closes.pct_change().dropna(how="all")
snp_vols = returns["SPY"].rolling(window=sbg_window).std()
gld_vols = returns["GLD"].rolling(window=sbg_window).std()
agg_vols = returns["AGG"].rolling(window=sbg_window).std()

spy60_agg40_rets = returns.mul(spy60_agg40_weights[-1], axis=1).sum(axis=1)
spy50_agg30_gld20_rets = returns.mul(spy50_agg30_gld20_weights[-1], axis=1).sum(axis=1)

spy60_agg40_vols = spy60_agg40_rets.rolling(window=sbg_window).std()
spy50_agg30_gld20_vols = spy50_agg30_gld20_rets.rolling(window=sbg_window).std()

snp_vols = snp_vols.loc[monthly_data.timeline].to_numpy()
gld_vols = gld_vols.loc[monthly_data.timeline].to_numpy()
agg_vols = agg_vols.loc[monthly_data.timeline].to_numpy()

spy60_agg40_vols = spy60_agg40_vols.loc[monthly_data.timeline].to_numpy()
spy50_agg30_gld20_vols = spy50_agg30_gld20_vols.loc[monthly_data.timeline].to_numpy()

# 50/30/20 anchor used only by the scratch momentum-without-vol-control test
anchor_503020 = np.zeros(n_assets)
anchor_503020[spy_idx] = 0.5
anchor_503020[agg_idx] = 0.3
anchor_503020[gld_idx] = 0.2

# ── Strategies ────────────────────────────────────────────────────────────────

# Individual assets
snp_strategy = FixedWeightPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=equal_weights,
    universe=snp_universe,
    leverage=LEVERAGE,
)
bond_strategy = FixedWeightPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=equal_weights,
    universe=bond_universe,
    leverage=LEVERAGE,
)
gold_strategy = FixedWeightPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=equal_weights,
    universe=gold_universe,
    leverage=LEVERAGE,
)

# Individual assets with vol control
snp_vc_strategy = FixedWeightVolControlPortfolioConstructor(
    ts_lookup=lookup,
    vols=snp_vols,
    fixed_weights=equal_weights,
    vol_target=STOCK_VOL_TARGET,
    universe=snp_universe,
    leverage=LEVERAGE,
)
bond_vc_strategy = FixedWeightVolControlPortfolioConstructor(
    ts_lookup=lookup,
    vols=agg_vols,
    fixed_weights=equal_weights,
    vol_target=BOND_VOL_TARGET,
    universe=bond_universe,
    leverage=LEVERAGE,
)
gold_vc_strategy = FixedWeightVolControlPortfolioConstructor(
    ts_lookup=lookup,
    vols=gld_vols,
    fixed_weights=equal_weights,
    vol_target=GOLD_VOL_TARGET,
    universe=gold_universe,
    leverage=LEVERAGE,
)

# Blended portfolios
sbg_6040_strategy = FixedWeightPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=spy60_agg40_weights,
    universe=sbg_6040_universe,
    leverage=LEVERAGE,
)
sbg_503020_strategy = FixedWeightPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=spy50_agg30_gld20_weights,
    universe=sbg_503020_universe,
    leverage=LEVERAGE,
)

# Static fixed-weight benchmark holding the average Markowitz weights (with cash)
sbg_markowitz_fw_strategy = FixedWeightPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=markowitz_avg_weights,
    universe=sbg_503020_universe,
    leverage=MARKOWITZ_AVG_LEVERAGE,
)

# Blended portfolios with vol control
sbg_6040_vc_strategy = FixedWeightVolControlPortfolioConstructor(
    ts_lookup=lookup,
    vols=spy60_agg40_vols,
    fixed_weights=spy60_agg40_weights,
    vol_target=SBG_6040_VOL_TARGET,
    universe=sbg_6040_universe,
    leverage=LEVERAGE,
)
sbg_503020_vc_strategy = FixedWeightVolControlPortfolioConstructor(
    ts_lookup=lookup,
    vols=spy50_agg30_gld20_vols,
    fixed_weights=spy50_agg30_gld20_weights,
    vol_target=SBG_503020_VOL_TARGET,
    universe=sbg_503020_universe,
    leverage=LEVERAGE,
)
sbg_6040_mat_vc_strategy = FixedWeightMatrixVolControlPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=spy60_agg40_weights,
    risk_model=risk_model_sbg,
    vol_target=SBG_6040_VOL_TARGET,
    universe=sbg_6040_universe,
    leverage=LEVERAGE,
)
sbg_503020_mat_vc_strategy = FixedWeightMatrixVolControlPortfolioConstructor(
    ts_lookup=lookup,
    fixed_weights=spy50_agg30_gld20_weights,
    risk_model=risk_model_sbg,
    vol_target=SBG_503020_VOL_TARGET,
    universe=sbg_503020_universe,
    leverage=LEVERAGE,
)

# Markowitz portfolio: ridge alpha with a 50/30/20 relative-allocation trust region.
sbg_alpha_strategy = AnchoredVolControlPortfolioConstructor(
    ts_lookup=lookup,
    alphas=pd.read_parquet(proc_data_dir / "alpha_ridge_sbg.parquet")
    .loc[monthly_data.timeline]
    .to_numpy(),
    risk_model=risk_model_sbg,
    vol_target=SBG_ALPHA_VOL_TARGET,
    anchor=anchor_503020,
    universe=sbg_503020_universe,
    leverage=LEVERAGE,
)

# Simple Markowitz: EWMA-of-returns alpha with the same trust region.
sbg_simple_alpha_strategy = AnchoredVolControlPortfolioConstructor(
    ts_lookup=lookup,
    alphas=pd.read_parquet(proc_data_dir / "alpha_simple_sbg.parquet")
    .loc[monthly_data.timeline]
    .to_numpy(),
    risk_model=risk_model_sbg,
    vol_target=SBG_ALPHA_VOL_TARGET,
    anchor=anchor_503020,
    universe=sbg_503020_universe,
    leverage=LEVERAGE,
)

# Simple Anchored momentum: EWMA-of-returns alpha with the same trust region
sbg_simple_alpha_strategy_no_vc = AnchoredAlphaPortfolioConstructor(
    ts_lookup=lookup,
    alphas=pd.read_parquet(proc_data_dir / "alpha_simple_sbg.parquet")
    .loc[monthly_data.timeline]
    .to_numpy(),
    risk_model=risk_model_sbg,
    vol_target=SBG_ALPHA_VOL_TARGET,
    anchor=anchor_503020,
    universe=sbg_503020_universe,
    leverage=LEVERAGE,
)

# sbg_r4_alpha = pd.read_parquet(proc_data_dir / "r4_sbg.parquet")
# sbg_r4_strategy = VolControlPortfolioConstructor(
#     ts_lookup=lookup,
#     alphas=sbg_r4_alpha.loc[monthly_data.timeline].to_numpy(),
#     risk_model=risk_model_sbg,
#     vol_target=SBG_ALPHA_VOL_TARGET,
#     universe=sbg_503020_universe,
#     leverage=LEVERAGE,
# )

# full_alpha_strategy = VolControlPortfolioConstructor(
#     ts_lookup=lookup,
#     alphas=pd.read_parquet(proc_data_dir / "alpha_factor_model_full.parquet")
#         .loc[monthly_data.timeline]
#         .to_numpy(),
#     risk_model=risk_model_full,
#     vol_target=SBG_ALPHA_VOL_TARGET,
#     universe=full_universe,
#     leverage=LEVERAGE,
# )

# ── Backtesting ───────────────────────────────────────────────────────────────

backtests = [
    ("snp", snp_strategy, yearly_data),
    ("bond", bond_strategy, yearly_data),
    ("gold", gold_strategy, yearly_data),
    ("sbg_6040", sbg_6040_strategy, yearly_data),
    ("sbg_503020", sbg_503020_strategy, yearly_data),
    ("sbg_markowitz_fw", sbg_markowitz_fw_strategy, yearly_data),
    ("sbg_6040_vc", sbg_6040_vc_strategy, monthly_data),
    ("sbg_503020_vc", sbg_503020_vc_strategy, monthly_data),
    ("sbg_alpha_vc", sbg_alpha_strategy, monthly_data),
    ("sbg_simple_alpha_vc", sbg_simple_alpha_strategy, monthly_data),
    # ("sbg_r4_vc_w", sbg_r4_strategy, weekly_data),
    # ("sbg_r4_vc_m", sbg_r4_strategy, monthly_data),
    # ("sbg_r4_vc_y", sbg_r4_strategy, yearly_data),
    # ("full_alpha_vc_w", full_alpha_strategy, weekly_data),
    # ("full_alpha_vc_m", full_alpha_strategy, monthly_data),
    # ("full_alpha_vc_y", full_alpha_strategy, yearly_data),
]

results = {}
results_dir = Path("results/")
results_dir.mkdir(parents=True, exist_ok=True)

for name, strategy, data in backtests:
    print(f"Running backtest: {name}")
    results[name] = run_backtest(backtest_name=name, data=data, portfolio_constructor=strategy)
    results[name].save(results_dir / f"{name}.pkl")


# -------------- Tax-bracket results -----------------
# Re-run the 8 headline strategies (the ones reported in the after-tax table in
# scripts/4_metrics_and_plots.py) under each non-zero tax bracket. The zero-tax
# bracket B1 is already covered by the vanilla {name}.pkl results above. Results
# are saved as results/tax/{name}_b{j}.pkl where j is the 1-indexed bracket.
tax_results_dir = Path("results/tax")
tax_results_dir.mkdir(parents=True, exist_ok=True)

# (name, strategy, rebal_freq) for the 8 headline backtests.
tax_backtests = [
    ("snp", snp_strategy, "Y"),
    ("bond", bond_strategy, "Y"),
    ("gold", gold_strategy, "Y"),
    ("sbg_6040", sbg_6040_strategy, "Y"),
    ("sbg_503020", sbg_503020_strategy, "Y"),
    ("sbg_6040_vc", sbg_6040_vc_strategy, "M"),
    ("sbg_503020_vc", sbg_503020_vc_strategy, "M"),
    ("sbg_alpha_vc", sbg_alpha_strategy, "M"),
    ("sbg_simple_alpha_vc", sbg_simple_alpha_strategy, "M"),
]

income_returns = pd.read_parquet("data/raw/income_returns.parquet")

for bracket_idx, (lt_base, st_base) in enumerate(TAX_BRACKETS[1:], start=2):
    surtax = NIIT_RATE if NIIT_APPLIES[bracket_idx - 1] else 0.0
    lt_rate = lt_base + surtax
    st_rate = st_base + surtax
    gld_rate = min(COLLECTIBLES_CAP, st_base) + surtax
    print(
        f"\n=== Tax bracket B{bracket_idx}: long_term={lt_rate:.3f}, "
        f"short_term={st_rate:.3f}, GLD long_term={gld_rate:.3f} ==="
    )

    # Tax-aware data per rebal frequency. The timeline is identical to the untaxed
    # data (tax does not change it), so the strategies' ts_lookup keys still align.
    # SPY pays qualified dividends, taxed at the long-term rate; AGG's distributions
    # are interest, taxed as ordinary income; GLD makes no distributions.
    freq_data = {
        freq: BacktestData.from_pandas(
            start_date=START_DATE,
            end_date=END_DATE,
            closes=closes,
            ffrs=ffr,
            rebal_freq=freq,
            long_term_rate=lt_rate,
            short_term_rate=st_rate,
            long_term_rates={"GLD": gld_rate},
            dividend_rates={"SPY": lt_rate, "AGG": st_rate, "GLD": 0.0},
            income_returns=income_returns,
        )
        for freq in ("Y", "M")
    }

    for name, strategy, freq in tax_backtests:
        result_name = f"{name}_b{bracket_idx}"
        print(f"Running tax backtest: {result_name}")
        result = run_backtest(
            backtest_name=result_name,
            data=freq_data[freq],
            portfolio_constructor=strategy,
        )
        result.save(tax_results_dir / f"{result_name}.pkl")


# -------------- Pareto curve results ----------------
# ── Backtesting ───────────────────────────────────────────────────────────────

results = {}
results_dir = Path("results/pareto")
results_dir.mkdir(parents=True, exist_ok=True)

VOL_TARGETS_ANN = np.arange(0.03, 0.1201, 0.005)
VOL_TARGETS_DAILY = VOL_TARGETS_ANN / np.sqrt(252)

for vol_target_ann, vol_target_daily in zip(VOL_TARGETS_ANN, VOL_TARGETS_DAILY, strict=True):
    vol_label = f"{100 * vol_target_ann:.1f}".replace(".", "p")

    print(
        f"\n=== Annualized vol target: {100 * vol_target_ann:.1f}% "
        f"(daily={vol_target_daily:.6f}) ==="
    )

    # Individual assets with vol control
    snp_vc_strategy = FixedWeightVolControlPortfolioConstructor(
        ts_lookup=lookup,
        vols=snp_vols,
        fixed_weights=equal_weights,
        vol_target=vol_target_daily,
        universe=snp_universe,
        leverage=LEVERAGE,
    )
    bond_vc_strategy = FixedWeightVolControlPortfolioConstructor(
        ts_lookup=lookup,
        vols=agg_vols,
        fixed_weights=equal_weights,
        vol_target=vol_target_daily,
        universe=bond_universe,
        leverage=LEVERAGE,
    )
    gold_vc_strategy = FixedWeightVolControlPortfolioConstructor(
        ts_lookup=lookup,
        vols=gld_vols,
        fixed_weights=equal_weights,
        vol_target=vol_target_daily,
        universe=gold_universe,
        leverage=LEVERAGE,
    )

    # Blended portfolios with vol control
    sbg_6040_vc_strategy = FixedWeightVolControlPortfolioConstructor(
        ts_lookup=lookup,
        vols=spy60_agg40_vols,
        fixed_weights=spy60_agg40_weights,
        vol_target=vol_target_daily,
        universe=sbg_6040_universe,
        leverage=LEVERAGE,
    )
    sbg_503020_vc_strategy = FixedWeightVolControlPortfolioConstructor(
        ts_lookup=lookup,
        vols=spy50_agg30_gld20_vols,
        fixed_weights=spy50_agg30_gld20_weights,
        vol_target=vol_target_daily,
        universe=sbg_503020_universe,
        leverage=LEVERAGE,
    )

    # Markowitz portfolio: ridge alpha with the 50/30/20 trust region.
    sbg_alpha_strategy = AnchoredVolControlPortfolioConstructor(
        ts_lookup=lookup,
        alphas=pd.read_parquet(proc_data_dir / "alpha_ridge_sbg.parquet")
        .loc[monthly_data.timeline]
        .to_numpy(),
        risk_model=risk_model_sbg,
        vol_target=vol_target_daily,
        anchor=anchor_503020,
        universe=sbg_503020_universe,
        leverage=LEVERAGE,
    )

    sbg_simple_alpha_strategy = AnchoredVolControlPortfolioConstructor(
        ts_lookup=lookup,
        alphas=pd.read_parquet(proc_data_dir / "alpha_simple_sbg.parquet")
        .loc[monthly_data.timeline]
        .to_numpy(),
        risk_model=risk_model_sbg,
        vol_target=vol_target_daily,
        anchor=anchor_503020,
        universe=sbg_503020_universe,
        leverage=LEVERAGE,
    )

    backtests = [
        ("sbg_6040_vc", sbg_6040_vc_strategy, monthly_data),
        ("sbg_503020_vc", sbg_503020_vc_strategy, monthly_data),
        ("sbg_alpha_vc", sbg_alpha_strategy, monthly_data),
        ("sbg_simple_alpha_vc", sbg_simple_alpha_strategy, monthly_data),
    ]

    for name, strategy, data in backtests:
        result_name = f"{name}_vol_{vol_label}"

        print(f"Running backtest: {result_name}")

        results[result_name] = run_backtest(
            backtest_name=result_name,
            data=data,
            portfolio_constructor=strategy,
        )

        results[result_name].save(results_dir / f"{result_name}.pkl")

print(f"\nDone. Saved Pareto results to {results_dir}")
