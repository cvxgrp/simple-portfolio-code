"""Compute metrics and generate plots for the paper."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simple_portfolio.metrics import compute_metrics  # noqa: E402
from simple_portfolio.simulator import BacktestResults  # noqa: E402

START_DATE = "2006-01-01"
END_DATE = "2026-01-01"

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("output/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
Path("output/tables").mkdir(parents=True, exist_ok=True)

WEIGHT_EPS = 1e-6  # threshold for treating an asset's weight as nonzero

BACKTESTS: dict[str, str] = {
    "snp": "SPY",
    "bond": "AGG",
    "gold": "GLD",
    "sbg_6040": "60/40",
    "sbg_503020": "50/30/20",
    "sbg_6040_vc": "60/40 VC",
    "sbg_503020_vc": "50/30/20 VC",
    "sbg_alpha_vc": "Markowitz",
    "sbg_simple_alpha_vc": "Simple Markowitz",
}

# (long_term_rate, short_term_rate)
TAX_BRACKETS: list[tuple[float, float]] = [
    (0.0, 0.0),  # Single 0 - 50k
    (0.15, 0.22),  # Single 50k - 100k
    (0.15, 0.35),  # Single 256k - 545k
    (0.2, 0.37),  # Single 640k +
]

# ── Load reference data ──────────────────────────────────────────────────────

ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0

# ── Load results and compute metrics ─────────────────────────────────────────

rows = {}
all_navs = {}
sliced_results: dict[str, BacktestResults] = {}

for pkl_name, paper_name in BACKTESTS.items():
    res = BacktestResults.load(RESULTS_DIR / f"{pkl_name}.pkl")
    res = BacktestResults(
        navs=res.navs.loc[START_DATE:END_DATE],
        composition=res.composition.loc[START_DATE:END_DATE],
        turnover=res.turnover.loc[START_DATE:END_DATE],
        metadata=res.metadata,
    )
    rows[paper_name] = compute_metrics(res, ffr, cpi)
    all_navs[paper_name] = res.navs
    sliced_results[paper_name] = res

results_df = pd.DataFrame(rows).T
results_df.index.name = "Portfolio"

# ── Save CSV ─────────────────────────────────────────────────────────────────

csv_cols = [
    "Return",
    "Volatility",
    "Sharpe Ratio (FFR)",
    "Max Drawdown",
    "Avg. Drawdown",
    "Turnover",
    "Consistency",
]
headline_df = results_df[csv_cols].copy()
cash_ffr = ffr.loc[START_DATE:END_DATE]
cash_return = (1.0 + cash_ffr).prod() ** (252 / len(cash_ffr)) - 1.0
headline_df.loc["Cash"] = [cash_return, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan]
headline_df.to_csv("output/tables/results.csv")
print("Saved output/tables/results.csv")
print(headline_df.to_string(float_format="{:.4f}".format))

# CPI-adjusted (real) return and Sharpe ratio, for the inflation table.
cpi_cols = ["Return - CPI", "Sharpe Ratio (CPI)"]
results_df[cpi_cols].to_csv("output/tables/results_cpi.csv")
print("Saved output/tables/results_cpi.csv")
print(results_df[cpi_cols].to_string(float_format="{:.4f}".format))

# ── After-tax return table ───────────────────────────────────────────────────
# Rows are the 8 backtests, columns the 4 tax brackets (B1..B4). Each cell is the
# annualized return of the strategy under that bracket. B1 is the zero-tax
# (vanilla) run; B2..B4 are separately simulated runs (scripts/3_base_portfolios.py
# tax loop) whose NAV already has the bracket's capital-gains tax subtracted along
# the way, including the terminal full-liquidation tax. The annualization matches
# the raw "Return" formula in compute_metrics, so B1 reproduces it exactly; higher
# brackets give lower returns.
TAX_RESULTS_DIR = RESULTS_DIR / "tax"
bracket_cols = [f"B{j}" for j in range(1, len(TAX_BRACKETS) + 1)]


def annualized_return(navs: pd.Series) -> float:
    """Annualized return from terminal NAV, matching compute_metrics' "Return"."""
    n_returns = len(navs) - 1  # matches len(navs.pct_change().dropna())
    return (navs.iloc[-1] / navs.iloc[0]) ** (252 / n_returns) - 1


tax_rows = {}
for pkl_name, paper_name in BACKTESTS.items():
    # B1 is the vanilla (zero-tax) run already loaded above.
    cells = {"B1": annualized_return(sliced_results[paper_name].navs)}
    for bracket_idx in range(2, len(TAX_BRACKETS) + 1):
        bres = BacktestResults.load(TAX_RESULTS_DIR / f"{pkl_name}_b{bracket_idx}.pkl")
        navs = bres.navs.loc[START_DATE:END_DATE]
        cells[f"B{bracket_idx}"] = annualized_return(navs)
    tax_rows[paper_name] = cells

tax_df = pd.DataFrame(tax_rows).T[bracket_cols]
tax_df.index.name = "Portfolio"
tax_df.to_csv("output/tables/results_tax.csv")
print("Saved output/tables/results_tax.csv")
print(tax_df.to_string(float_format="{:.4f}".format))

# ── After-tax FFR-adjusted Sharpe table ──────────────────────────────────────
# Same layout (8 strategies x brackets B1..B4). The numerator is each bracket's
# after-tax annualized return in excess of the FFR (from the taxed NAV, matching
# compute_metrics). The denominator is the zero-tax (B1) volatility, held fixed
# across brackets: taxes change the realized NAV but we do not treat the tax drag
# as portfolio risk. B1 therefore reproduces the headline "Sharpe Ratio (FFR)".


def annualized_excess_ffr_return(navs: pd.Series) -> float:
    """Annualized return in excess of the FFR, matching compute_metrics."""
    returns = navs.pct_change().dropna()
    cumu_rets = navs.iloc[-1] / navs.iloc[0]
    cumu_ffr = (1.0 + ffr.loc[returns.index]).prod()
    return (cumu_rets / cumu_ffr) ** (252 / len(returns)) - 1.0


sharpe_rows = {}
for pkl_name, paper_name in BACKTESTS.items():
    vol = results_df.loc[
        paper_name, "Volatility"
    ]  # zero-tax volatility, fixed across brackets
    # B1 is the vanilla (zero-tax) run already loaded above.
    cells = {"B1": annualized_excess_ffr_return(sliced_results[paper_name].navs) / vol}
    for bracket_idx in range(2, len(TAX_BRACKETS) + 1):
        bres = BacktestResults.load(TAX_RESULTS_DIR / f"{pkl_name}_b{bracket_idx}.pkl")
        navs = bres.navs.loc[START_DATE:END_DATE]
        cells[f"B{bracket_idx}"] = annualized_excess_ffr_return(navs) / vol
    sharpe_rows[paper_name] = cells

sharpe_df = pd.DataFrame(sharpe_rows).T[bracket_cols]
sharpe_df.index.name = "Portfolio"
sharpe_df.to_csv("output/tables/results_tax_sharpe.csv")
print("Saved output/tables/results_tax_sharpe.csv")
print(sharpe_df.to_string(float_format="{:.4f}".format))

# ── Standalone benchmark: fixed weights = average Markowitz weights ───────────
# Reported in the text only (no plot or table entry). Holding the average
# Markowitz weights statically isolates the value of the dynamic allocation.
mkw_fw = BacktestResults.load(RESULTS_DIR / "sbg_markowitz_fw.pkl")
mkw_fw = BacktestResults(
    navs=mkw_fw.navs.loc[START_DATE:END_DATE],
    composition=mkw_fw.composition.loc[START_DATE:END_DATE],
    turnover=mkw_fw.turnover.loc[START_DATE:END_DATE],
    metadata=mkw_fw.metadata,
)
mkw_fw_metrics = compute_metrics(mkw_fw, ffr, cpi)
print(
    "Fixed-weight (avg Markowitz weights) benchmark: "
    f"Return={mkw_fw_metrics['Return']:.4f}, "
    f"Volatility={mkw_fw_metrics['Volatility']:.4f}, "
    f"Sharpe (FFR)={mkw_fw_metrics['Sharpe Ratio (FFR)']:.4f}"
)

# ── Cumulative return plot (2x2) ─────────────────────────────────────────────

navs_df = pd.DataFrame(all_navs)
navs_df = navs_df / navs_df.iloc[0]
monthly_mask = (
    ~pd.Series(navs_df.index).dt.to_period("Q").duplicated(keep="first").to_numpy()
)
navs_df = navs_df.iloc[monthly_mask]

LINES: list[list[tuple[str, str, str, float]]] = [
    # (label, color, linestyle, linewidth)
    [
        ("SPY", "#0072B2", "-", 1.2),
        ("AGG", "#009E73", "-", 1.2),
        ("GLD", "#E69F00", "-", 1.2),
    ],
    [("60/40", "#E8694A", "-", 1.2), ("60/40 VC", "#B22000", "--", 1.2)],
    [("50/30/20", "#B39DDB", "-", 1.2), ("50/30/20 VC", "#6A3D9A", "--", 1.2)],
    [("Markowitz", "#000000", "--", 1.2), ("Simple Markowitz", "#808080", "-", 1.2)],
]
TITLES = [
    "Individual assets",
    "60/40 portfolios",
    "50/30/20 portfolios",
    "Markowitz portfolios",
]

yticks = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
_plotted_labels = [label for panel in LINES for label, *_ in panel]
ymin = navs_df[_plotted_labels].min(axis=None) * 0.9
ymax = navs_df[_plotted_labels].max(axis=None) * 1.1


def _plot_cumulative(fname: str) -> None:
    """Draw the 2x2 cumulative-return plot."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)

    for ax, panel, title in zip(axes.flat, LINES, TITLES, strict=True):
        for label, color, ls, lw in panel:
            ax.plot(
                navs_df.index,
                navs_df[label],
                label=label,
                color=color,
                linestyle=ls,
                linewidth=lw,
            )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
        ax.grid(True, which="minor", linewidth=0.3, alpha=0.2)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:g}" for v in yticks])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylim(ymin, ymax)

    handles = []
    labels = []
    for panel in LINES:
        for label, color, ls, lw in panel:
            handles.append(plt.Line2D([], [], color=color, linestyle=ls, linewidth=lw))
            labels.append(label)

    axes[1, 0].set_xlabel("Date")
    axes[1, 1].set_xlabel("Date")
    axes[0, 0].set_ylabel("Cumulative return")
    axes[1, 0].set_ylabel("Cumulative return")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(PLOTS_DIR / fname, bbox_inches="tight")
    print(f"Saved {PLOTS_DIR / fname}")
    plt.close(fig)


_plot_cumulative("cumulative_returns.pdf")

# ── CPI-adjusted (real) cumulative return plot (2x2) ─────────────────────────

# Deflate each NAV by the cumulative CPI index, then renormalize to start at 1.
navs_real = pd.DataFrame(all_navs)
cpi_index = (1.0 + cpi.reindex(navs_real.index).fillna(0.0)).cumprod()
navs_real = navs_real.div(cpi_index, axis=0)
navs_real = navs_real / navs_real.iloc[0]
navs_real = navs_real.iloc[monthly_mask]

ymin_real = navs_real[_plotted_labels].min(axis=None) * 0.9
ymax_real = navs_real[_plotted_labels].max(axis=None) * 1.1

fig_real, axes_real = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)

for ax, panel, title in zip(axes_real.flat, LINES, TITLES, strict=True):
    for label, color, ls, lw in panel:
        ax.plot(
            navs_real.index,
            navs_real[label],
            label=label,
            color=color,
            linestyle=ls,
            linewidth=lw,
        )
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(ymin_real, ymax_real)

axes_real[1, 0].set_xlabel("Date")
axes_real[1, 1].set_xlabel("Date")
axes_real[0, 0].set_ylabel("Real cumulative return")
axes_real[1, 0].set_ylabel("Real cumulative return")

handles = []
labels = []
for panel in LINES:
    for label, color, ls, lw in panel:
        handles.append(plt.Line2D([], [], color=color, linestyle=ls, linewidth=lw))
        labels.append(label)

fig_real.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=7,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig_real.tight_layout(rect=[0, 0.05, 1, 1])
fig_real.savefig(PLOTS_DIR / "cumulative_returns_real.pdf", bbox_inches="tight")
print(f"Saved {PLOTS_DIR / 'cumulative_returns_real.pdf'}")
plt.close(fig_real)

# ── Composition stackplots ───────────────────────────────────────────────────

STACK_COLORS = {"SPY": "#0072B2", "AGG": "#009E73", "GLD": "#E69F00", "Cash": "#CCCCCC"}
STACK_ORDER = ["SPY", "AGG", "GLD", "Cash"]

STACK_BACKTESTS: dict[str, str] = {
    "sbg_6040": "60/40",
    "sbg_503020": "50/30/20",
    "sbg_6040_vc": "60/40 VC",
    "sbg_503020_vc": "50/30/20 VC",
    "sbg_alpha_vc": "Markowitz",
    "sbg_simple_alpha_vc": "Simple Markowitz",
}

for pkl_name, paper_name in STACK_BACKTESTS.items():
    res = BacktestResults.load(RESULTS_DIR / f"{pkl_name}.pkl")
    comp = res.composition.loc[START_DATE:END_DATE].rename(columns={"Cash": "Cash"})

    # The stackplot is drawn on a quarterly subsample (a daily stackplot over 20
    # years is illegible); the reported averages below use every trading day.
    quarterly_idx = (
        ~pd.Series(comp.index).dt.to_period("Q").duplicated(keep="first").to_numpy()
    )
    comp_plot = comp.iloc[quarterly_idx]

    cols = [
        c for c in STACK_ORDER if c in comp.columns and comp[c].abs().sum() > WEIGHT_EPS
    ]
    colors = [STACK_COLORS[c] for c in cols]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(
        comp_plot.index,
        *[np.abs(comp_plot[c]) for c in cols],
        labels=cols,
        colors=colors,
    )
    # ax.set_ylim(0, 1.3)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_title(paper_name, fontsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(cols),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()

    fname = f"composition_{pkl_name}.pdf"
    fig.savefig(PLOTS_DIR / fname, bbox_inches="tight")
    print(f"Saved {PLOTS_DIR / fname}")
    plt.close(fig)

    # print average allocation across time
    avg_alloc = comp.mean()
    print(f"Average allocation for {paper_name}:")
    for asset in cols:
        print(f"  {asset}: {avg_alloc[asset]:.1%}")


# ── Pareto plot: realized annualized risk vs annualized return ────────────────

PARETO_RESULTS_DIR = RESULTS_DIR / "pareto"

PARETO_BACKTESTS: dict[str, str] = {
    "sbg_6040_vc": "60/40 VC",
    "sbg_503020_vc": "50/30/20 VC",
    "sbg_alpha_vc": "Markowitz",
    "sbg_simple_alpha_vc": "Simple Markowitz",
    # "sbg_r4_vc": "R4",
}

PARETO_LINES: dict[str, tuple[str, str, float]] = {
    # label: (color, linestyle, linewidth)
    "SPY VC": ("#0072B2", "-", 1.2),
    "AGG VC": ("#009E73", "-", 1.2),
    "GLD VC": ("#E69F00", "-", 1.2),
    "60/40 VC": ("#B22000", "-", 1.2),
    "50/30/20 VC": ("#6A3D9A", "-", 1.2),
    "Markowitz": ("#000000", "-", 1.2),
    "Simple Markowitz": ("#808080", "-", 1.2),
    "R4": ("#666666", ":", 1.2),
}

pareto_rows = []

for vol_target_ann in np.arange(0.03, 0.1201, 0.005):
    vol_label = f"{100 * vol_target_ann:.1f}".replace(".", "p")

    for pkl_prefix, paper_name in PARETO_BACKTESTS.items():
        pkl_path = PARETO_RESULTS_DIR / f"{pkl_prefix}_vol_{vol_label}.pkl"

        if not pkl_path.exists():
            print(f"Skipping missing Pareto result: {pkl_path}")
            continue

        res = BacktestResults.load(pkl_path)
        res = BacktestResults(
            navs=res.navs.loc[START_DATE:END_DATE],
            composition=res.composition.loc[START_DATE:END_DATE],
            turnover=res.turnover.loc[START_DATE:END_DATE],
            metadata=res.metadata,
        )

        metrics = compute_metrics(res, ffr, cpi)

        pareto_rows.append(
            {
                "Portfolio": paper_name,
                "Target Volatility": vol_target_ann,
                "Return": metrics["Return"],
                "Volatility": metrics["Volatility"],
                "Sharpe Ratio (FFR)": metrics["Sharpe Ratio (FFR)"],
            }
        )

pareto_df = pd.DataFrame(pareto_rows)
pareto_df.to_csv("output/tables/pareto_results.csv", index=False)
print("Saved output/tables/pareto_results.csv")

# Single-point benchmarks with no volatility knob: the raw assets (no vol
# control) and the fixed-weight portfolios. Metrics come from results_df
# (computed above); each point is placed at its realized volatility.
POINT_BENCHMARKS = ["SPY", "AGG", "GLD", "60/40", "50/30/20"]
POINT_COLORS = {
    "SPY": "#0072B2",
    "AGG": "#009E73",
    "GLD": "#E69F00",
    "60/40": "#E8694A",
    "50/30/20": "#B39DDB",
}

fig3, ax3 = plt.subplots(figsize=(5, 3.5))

for paper_name, group in pareto_df.groupby("Portfolio", sort=False):
    sorted_group = group.sort_values("Volatility")

    color, ls, lw = PARETO_LINES[paper_name]

    ax3.plot(
        sorted_group["Volatility"],
        sorted_group["Return"],
        label=paper_name,
        color=color,
        linestyle=ls,
        linewidth=lw,
        marker="o",
        markersize=2.5,
    )

for name in POINT_BENCHMARKS:
    ax3.scatter(
        results_df.loc[name, "Volatility"],
        results_df.loc[name, "Return"],
        color=POINT_COLORS[name],
        marker="*",
        s=70,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
        label=name,
    )

ax3.set_xlabel("Volatility")
ax3.set_ylabel("Return")
ax3.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax3.grid(True, which="minor", linewidth=0.3, alpha=0.2)

ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
ax3.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))

ax3.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    fontsize=7,
    frameon=False,
)

fig3.tight_layout()
fig3.savefig(PLOTS_DIR / "pareto_risk_return.pdf", bbox_inches="tight")
print(f"Saved {PLOTS_DIR / 'pareto_risk_return.pdf'}")
plt.close(fig3)

# Sharpe ratio versus realized volatility.
fig5, ax5 = plt.subplots(figsize=(5, 3.5))

for paper_name, group in pareto_df.groupby("Portfolio", sort=False):
    sorted_group = group.sort_values("Volatility")
    color, ls, lw = PARETO_LINES[paper_name]
    ax5.plot(
        sorted_group["Volatility"],
        sorted_group["Sharpe Ratio (FFR)"],
        label=paper_name,
        color=color,
        linestyle=ls,
        linewidth=lw,
        marker="o",
        markersize=2.5,
    )

for name in POINT_BENCHMARKS:
    ax5.scatter(
        results_df.loc[name, "Volatility"],
        results_df.loc[name, "Sharpe Ratio (FFR)"],
        color=POINT_COLORS[name],
        marker="*",
        s=70,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
        label=name,
    )

ax5.set_xlabel("Volatility")
ax5.set_ylabel("Sharpe ratio")
ax5.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
ax5.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
ax5.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    fontsize=7,
    frameon=False,
)
fig5.tight_layout()
fig5.savefig(PLOTS_DIR / "pareto_sharpe_vs_vol.pdf", bbox_inches="tight")
plt.close(fig5)

# ── Alpha signal plot (3x1: one panel per asset) ─────────────────────────────

ALPHA_PATH = Path("data/processed/alpha_ridge_sbg.parquet")
SIMPLE_ALPHA_PATH = Path("data/processed/alpha_simple_sbg.parquet")

alpha_cols = ["SPY", "AGG", "GLD"]
EVAL_HORIZON = 100  # forward horizon of the realized-return comparison (trading days),
# matching the alpha models' own fitting target h (see scripts/2_generate_alphas.py).
# Note this is longer than one rebalance period, which is a month (~21 days).

# True cumulative forward return over the next EVAL_HORIZON days, annualized
# with the same scaling convention as the alpha models' target. The realized
# return for the final EVAL_HORIZON days of the window needs closes past
# END_DATE; use the evaluation-only extended closes when available
# (scripts/1b_download_eval_closes.py) so the target covers the whole window.
EVAL_CLOSES_PATH = Path("data/raw/closes_eval.parquet")
if EVAL_CLOSES_PATH.exists():
    closes_sbg = pd.read_parquet(EVAL_CLOSES_PATH)[alpha_cols]
else:
    print(f"{EVAL_CLOSES_PATH} not found; target ends {EVAL_HORIZON} days early")
    closes_sbg = pd.read_parquet("data/raw/closes.parquet")[alpha_cols]
alpha_target = (
    closes_sbg.pct_change(EVAL_HORIZON).shift(-EVAL_HORIZON) * 252 / (EVAL_HORIZON)
)

# Trailing EWMA alpha used by the simple Markowitz portfolio, rescaled from
# per-day units into the target's units (a pure rescaling; the optimizer is
# invariant to the scale of alpha).
simple_alphas = pd.read_parquet(SIMPLE_ALPHA_PATH)[alpha_cols] * 252

# Ridge alpha used by the Markowitz portfolio. Already in the target's
# (annualized) units, so no rescaling is needed.
alphas = pd.read_parquet(ALPHA_PATH)[alpha_cols]

# One panel per asset, with the realized forward return and the two forecasts
# overlaid. Colors identify the series, not the asset, and match the portfolio
# colors used elsewhere (Markowitz black, simple Markowitz grey).
ALPHA_SERIES: list[tuple[str, pd.DataFrame, str, str]] = [
    # (label, frame, color, linestyle)
    ("Realized forward return", alpha_target, "#CC79A7", "--"),
    ("Simple Markowitz alpha", simple_alphas, "#808080", "-"),
    ("Markowitz alpha", alphas, "#000000", "-"),
]


fig_alpha, axes_alpha = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=False)

for ax, col in zip(axes_alpha, alpha_cols, strict=True):
    for label, df, color, ls in ALPHA_SERIES:
        s = df[col].loc[START_DATE:END_DATE]
        # The forecast emits exactly zero during its warm-up; drop those
        # days so the warm-up is not drawn as a flat line.
        s = s.where(s != 0.0)
        ax.plot(s.index, s, label=label, color=color, linestyle=ls, linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_title(col, fontsize=9)
    ax.set_ylabel("Annualized return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.2)

axes_alpha[-1].set_xlabel("Date")
axes_alpha[-1].set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))

handles_alpha = [
    plt.Line2D([], [], color=color, linestyle=ls, linewidth=1.3)
    for _, _, color, ls in ALPHA_SERIES
]
fig_alpha.legend(
    handles_alpha,
    [label for label, *_ in ALPHA_SERIES],
    loc="lower center",
    ncol=len(ALPHA_SERIES),
    fontsize=11,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

fig_alpha.tight_layout()
fig_alpha.savefig(PLOTS_DIR / "alphas_sbg.pdf", bbox_inches="tight")
print(f"Saved {PLOTS_DIR / 'alphas_sbg.pdf'}")
plt.close(fig_alpha)

# ── Alpha-target correlation table ───────────────────────────────────────────
# For each forecast and asset: the correlation between the alpha and the
# realized EVAL_HORIZON-day forward return, and the fraction of
# days on which the forecast's sign matches the realized return's, over the
# simulation window. Days without a forecast (alpha exactly zero, e.g. the
# model's warm-up period) or without a realized return (the final
# EVAL_HORIZON days) are excluded.

corr_rows = {}
for forecast_name, alpha_df in [
    ("Simple Markowitz", simple_alphas),
    ("Markowitz", alphas),
]:
    cells = {}
    for col in alpha_cols:
        a = alpha_df[col].loc[START_DATE:END_DATE]
        t = alpha_target[col].loc[START_DATE:END_DATE]
        valid = t.notna() & (a != 0.0)
        cells[f"{col} corr"] = a[valid].corr(t[valid])
        cells[f"{col} sign acc"] = (np.sign(a[valid]) == np.sign(t[valid])).mean()
    corr_rows[forecast_name] = cells

corr_df = pd.DataFrame(corr_rows).T
corr_df.index.name = "Portfolio"
corr_df.to_csv("output/tables/alpha_target_correlations.csv")
print("Saved output/tables/alpha_target_correlations.csv")
print(corr_df.to_string(float_format="{:.4f}".format))

# ── Annual cross-sectional cosine similarity ────────────────────────────────
# For each date, compute the cosine similarity between the three-dimensional
# alpha vector and the corresponding realized forward-return vector:
#
#     cosine_t = alpha_t^T target_t
#                / (||alpha_t||_2 ||target_t||_2).
#
# The daily cosine similarities are then averaged within each calendar year.
# A value near 1 means that the forecast and realized-return vectors point in
# similar cross-sectional directions; a value near -1 means that they point in
# opposite directions.


def cross_sectional_cosine_similarity(
    alpha_df: pd.DataFrame,
    target_df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """Compute cross-sectional cosine similarity for each date."""
    alpha_aligned, target_aligned = alpha_df[columns].align(
        target_df[columns],
        join="inner",
        axis=0,
    )

    alpha_values = alpha_aligned.to_numpy(dtype=float)
    target_values = target_aligned.to_numpy(dtype=float)

    valid_rows = np.isfinite(alpha_values).all(axis=1) & np.isfinite(target_values).all(
        axis=1
    )

    alpha_norm = np.linalg.norm(alpha_values, axis=1)
    target_norm = np.linalg.norm(target_values, axis=1)

    valid_rows &= (alpha_norm > 0.0) & (target_norm > 0.0)

    cosine = np.full(len(alpha_aligned), np.nan, dtype=float)
    cosine[valid_rows] = np.sum(
        alpha_values[valid_rows] * target_values[valid_rows],
        axis=1,
    ) / (alpha_norm[valid_rows] * target_norm[valid_rows])

    return pd.Series(
        cosine,
        index=alpha_aligned.index,
        name="Cosine similarity",
    )


cosine_daily = pd.DataFrame(
    {
        "Simple Markowitz": cross_sectional_cosine_similarity(
            simple_alphas.loc[START_DATE:END_DATE],
            alpha_target.loc[START_DATE:END_DATE],
            alpha_cols,
        ),
        "Markowitz": cross_sectional_cosine_similarity(
            alphas.loc[START_DATE:END_DATE],
            alpha_target.loc[START_DATE:END_DATE],
            alpha_cols,
        ),
    }
)

# Average daily cross-sectional cosine similarity within each year.
cosine_annual = cosine_daily.groupby(cosine_daily.index.year).mean()
cosine_annual.index.name = "Year"

# Save the annual values.
cosine_annual.to_csv("output/tables/alpha_target_cosine_similarity_by_year.csv")
print("Saved output/tables/alpha_target_cosine_similarity_by_year.csv")
print(cosine_annual.to_string(float_format="{:.4f}".format))

# Plot both alpha models in one figure.
fig_cosine, ax_cosine = plt.subplots(figsize=(10, 4))

COSINE_STYLES = [
    ("Markowitz", "#000000", "--", 1.2),
    ("Simple Markowitz", "#808080", "-", 1.2),
]

for label, color, linestyle, linewidth in COSINE_STYLES:
    ax_cosine.plot(
        cosine_daily.index,
        cosine_daily[label].rolling(100).mean(),
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

ax_cosine.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=0.8,
    alpha=0.6,
)

ax_cosine.set_xlabel("Year")
ax_cosine.set_ylabel("Cosine similarity")
ax_cosine.set_ylim(-1.0, 1.0)

ax_cosine.grid(
    True,
    which="major",
    linewidth=0.5,
    alpha=0.4,
)

ax_cosine.legend(
    fontsize=8,
    frameon=False,
)

fig_cosine.tight_layout()

plot_path = PLOTS_DIR / "alpha_target_cosine_similarity.pdf"
fig_cosine.savefig(plot_path, bbox_inches="tight")

print(f"Saved {plot_path}")
plt.close(fig_cosine)

# ── Yearly return and volatility bar plots ───────────────────────────────────

yearly_navs = pd.DataFrame(all_navs)
yearly_navs = yearly_navs.loc[START_DATE:END_DATE]

# Daily or rebalance-period returns, depending on the saved NAV frequency.
yearly_rets = yearly_navs.pct_change().dropna(how="all")

# Calendar-year annual returns.
annual_returns = yearly_navs.resample("YE").last().pct_change().dropna(how="all")
annual_returns.index = annual_returns.index.year

# Calendar-year annualized volatility.
annual_vols = yearly_rets.groupby(yearly_rets.index.year).std() * np.sqrt(252)

# Keep only years that exist in both.
common_years = annual_returns.index.intersection(annual_vols.index)
annual_returns = annual_returns.loc[common_years]
annual_vols = annual_vols.loc[common_years]


PLOT_ORDER = [
    # "SPY",
    # "AGG",
    # "GLD",
    "60/40",
    "50/30/20",
    "60/40 VC",
    "50/30/20 VC",
    "Markowitz",
    "Simple Markowitz",
]

PLOT_COLORS = {
    "SPY": "#0072B2",
    "AGG": "#009E73",
    "GLD": "#E69F00",
    "60/40": "#E8694A",
    "50/30/20": "#B39DDB",
    "60/40 VC": "#B22000",
    "50/30/20 VC": "#6A3D9A",
    "Markowitz": "#000000",
    "Simple Markowitz": "#808080",
}

# Markowitz is drawn dashed everywhere except the Pareto plots.
PLOT_LINESTYLES = {name: ("--" if name == "Markowitz" else "-") for name in PLOT_COLORS}

annual_returns = annual_returns[[c for c in PLOT_ORDER if c in annual_returns.columns]]
annual_vols = annual_vols[[c for c in PLOT_ORDER if c in annual_vols.columns]]

# ── Plot yearly realized volatility

# ── Plot yearly realized volatility ──────────────────────────────────────────

# 1x3 panels grouped as in the cumulative-return plot (same colors/linestyles).
VOL_PANELS: list[list[tuple[str, str, str, float]]] = [
    # (label, color, linestyle, linewidth)
    [("60/40", "#E8694A", "-", 1.2), ("60/40 VC", "#B22000", "--", 1.2)],
    [("50/30/20", "#B39DDB", "-", 1.2), ("50/30/20 VC", "#6A3D9A", "--", 1.2)],
    [("Markowitz", "#000000", "--", 1.2), ("Simple Markowitz", "#808080", "-", 1.2)],
]
VOL_TITLES = ["60/40 portfolios", "50/30/20 portfolios", "Markowitz portfolios"]

fig_vol, axes_vol = plt.subplots(1, 3, figsize=(7, 3), sharey=True)

for ax, panel, title in zip(axes_vol, VOL_PANELS, VOL_TITLES, strict=True):
    for label, color, ls, lw in panel:
        ax.plot(
            annual_vols.index,
            annual_vols[label],
            label=label,
            color=color,
            linestyle=ls,
            linewidth=lw,
        )

    # Dashed horizontal line at the 7% annualized volatility target.
    ax.axhline(0.07, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Year")
    ax.set_ylim(0.0, 0.25)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.2)
    ax.set_xticks(annual_vols.index[::3])
    ax.set_xticklabels(
        [str(int(y)) for y in annual_vols.index[::3]], rotation=45, ha="right"
    )

axes_vol[0].set_ylabel("Realized volatility")
axes_vol[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))

handles_vol = []
labels_vol = []
for panel in VOL_PANELS:
    for label, color, ls, lw in panel:
        handles_vol.append(plt.Line2D([], [], color=color, linestyle=ls, linewidth=lw))
        labels_vol.append(label)

fig_vol.legend(
    handles_vol,
    labels_vol,
    loc="lower center",
    ncol=6,
    fontsize=7,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
)

fig_vol.tight_layout()
fig_vol.savefig(PLOTS_DIR / "yearly_volatilities.pdf", bbox_inches="tight")
print(f"Saved {PLOTS_DIR / 'yearly_volatilities.pdf'}")
plt.close(fig_vol)
