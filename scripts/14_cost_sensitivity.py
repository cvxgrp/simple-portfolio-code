"""Transaction-cost and cash-rate sensitivity for the six portfolios.

Two referee objections are addressed here, both by re-running the existing strategies with a
different simulation assumption; no strategy or forecast is changed.

* Trading costs. The paper charges a 5 basis point spread. We re-run at 10 and 20 basis points.
* Cash. The paper accrues cash at the effective federal funds rate, which an individual investor
  cannot obtain directly. We re-run with cash accruing at the federal funds rate less 25 basis
  points annualized, which is conservative relative to a Treasury bill fund. The Sharpe ratio is
  still measured in excess of the federal funds rate itself, so the haircut is a pure penalty.

Results go to ``output/tables/sensitivity/``.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from simple_portfolio.metrics import compute_metrics
from simple_portfolio.optimizer import (
    AnchoredVolControlPortfolioConstructor,
    FixedWeightPortfolioConstructor,
    FixedWeightVolControlPortfolioConstructor,
)
from simple_portfolio.simulator import BacktestData, BacktestResults, run_backtest

START_DATE = pd.Timestamp("2005-01-01")
END_DATE = pd.Timestamp("2026-01-01")
EVAL_START = "2006-01-01"
EVAL_END = "2026-01-01"
LEVERAGE = 1.0
VOL_TARGET = 0.07 / np.sqrt(252)
SBG_WINDOW = 11
SPREADS = [5e-4, 1e-3, 2e-3]
CASH_HAIRCUTS = [0.0, 0.0025]
METRICS = ["Return", "Volatility", "Sharpe Ratio (FFR)", "Max Drawdown", "Turnover"]


def sliced(result: BacktestResults) -> BacktestResults:
    return BacktestResults(
        navs=result.navs.loc[EVAL_START:EVAL_END],
        composition=result.composition.loc[EVAL_START:EVAL_END],
        turnover=result.turnover.loc[EVAL_START:EVAL_END],
        metadata=result.metadata,
    )


def main() -> None:
    print("Started cost sensitivity analysis...", flush=True)
    closes = pd.read_parquet("data/raw/closes.parquet")
    ffr = pd.read_parquet("data/raw/ffr.parquet")["FFR"]
    cpi = pd.read_parquet("data/raw/cpi_econ.parquet")["CPI"] - 1.0
    with Path("data/processed/risk_model/sbg.pkl").open("rb") as file:
        risk_model = pickle.load(file)  # noqa: S301

    rows = {}
    for haircut in CASH_HAIRCUTS:
        cash_rate = ffr - haircut / 252.0
        monthly = BacktestData.from_pandas(
            start_date=START_DATE, end_date=END_DATE, closes=closes, ffrs=cash_rate, rebal_freq="M"
        )
        yearly = BacktestData.from_pandas(
            start_date=START_DATE, end_date=END_DATE, closes=closes, ffrs=cash_rate, rebal_freq="Y"
        )
        lookup = {ts: i for i, ts in enumerate(monthly.timeline)}
        n_time, n_assets = len(monthly.timeline), len(monthly.assets)
        assets = closes.columns
        spy = int(np.where(monthly.assets == "SPY")[0][0])
        agg = int(np.where(monthly.assets == "AGG")[0][0])
        gld = int(np.where(monthly.assets == "GLD")[0][0])
        u_6040 = (assets == "SPY") | (assets == "AGG")
        u_503020 = u_6040 | (assets == "GLD")

        w6040 = np.zeros((n_time, n_assets))
        w6040[:, spy], w6040[:, agg] = 0.6, 0.4
        w503020 = np.zeros((n_time, n_assets))
        w503020[:, spy], w503020[:, agg], w503020[:, gld] = 0.5, 0.3, 0.2
        anchor = np.zeros(n_assets)
        anchor[spy], anchor[agg], anchor[gld] = 0.5, 0.3, 0.2

        returns = closes.pct_change()
        vol6040 = (
            returns.mul(w6040[-1], axis=1)
            .sum(axis=1)
            .rolling(SBG_WINDOW)
            .std()
            .loc[monthly.timeline]
            .to_numpy()
        )
        vol503020 = (
            returns.mul(w503020[-1], axis=1)
            .sum(axis=1)
            .rolling(SBG_WINDOW)
            .std()
            .loc[monthly.timeline]
            .to_numpy()
        )

        def alpha_constructor(
            name: str,
            lookup: dict = lookup,
            monthly: BacktestData = monthly,
            anchor: np.ndarray = anchor,
            universe: np.ndarray = u_503020,
        ) -> AnchoredVolControlPortfolioConstructor:
            # The loop-carried values are bound as defaults so each iteration's
            # constructor keeps that iteration's data rather than the last one's.
            return AnchoredVolControlPortfolioConstructor(
                ts_lookup=lookup,
                alphas=pd.read_parquet(f"data/processed/{name}.parquet")
                .loc[monthly.timeline]
                .to_numpy(),
                risk_model=risk_model,
                vol_target=VOL_TARGET,
                anchor=anchor,
                universe=universe,
                leverage=LEVERAGE,
            )

        strategies = {
            "Markowitz": (alpha_constructor("alpha_ridge_sbg"), monthly),
            "Simple Markowitz": (alpha_constructor("alpha_simple_sbg"), monthly),
            "50/30/20 VC": (
                FixedWeightVolControlPortfolioConstructor(
                    ts_lookup=lookup,
                    vols=vol503020,
                    fixed_weights=w503020,
                    vol_target=VOL_TARGET,
                    universe=u_503020,
                    leverage=LEVERAGE,
                ),
                monthly,
            ),
            "60/40 VC": (
                FixedWeightVolControlPortfolioConstructor(
                    ts_lookup=lookup,
                    vols=vol6040,
                    fixed_weights=w6040,
                    vol_target=VOL_TARGET,
                    universe=u_6040,
                    leverage=LEVERAGE,
                ),
                monthly,
            ),
            "50/30/20": (
                FixedWeightPortfolioConstructor(
                    ts_lookup=lookup,
                    fixed_weights=w503020,
                    universe=u_503020,
                    leverage=LEVERAGE,
                ),
                yearly,
            ),
            "60/40": (
                FixedWeightPortfolioConstructor(
                    ts_lookup=lookup, fixed_weights=w6040, universe=u_6040, leverage=LEVERAGE
                ),
                yearly,
            ),
        }

        for spread in SPREADS:
            if haircut > 0.0 and spread != SPREADS[0]:
                continue  # only the base spread is re-run under the cash haircut
            for label, (constructor, data) in strategies.items():
                result = run_backtest(label, data, constructor, bid_ask_spread=spread)
                rows[(label, spread, haircut)] = compute_metrics(sliced(result), ffr, cpi)[METRICS]
            print(f"done spread={spread}, haircut={haircut}", flush=True)

    table = pd.DataFrame(rows).T
    table.index.names = ["Portfolio", "Spread", "Cash haircut"]
    out_dir = Path("output/tables/sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "cost_and_cash.csv")

    print("\n=== Sharpe ratio by trading cost (no cash haircut)")
    base = table.xs(0.0, level="Cash haircut")["Sharpe Ratio (FFR)"].unstack("Spread")
    print(base.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== Return by trading cost (no cash haircut)")
    print(
        table.xs(0.0, level="Cash haircut")["Return"]
        .unstack("Spread")
        .to_string(float_format=lambda v: f"{v:.4f}")
    )
    print("\n=== Cash haircut of 25bp at the base spread")
    cut = table.xs(SPREADS[0], level="Spread")
    print(cut.to_string(float_format=lambda v: f"{v:.4f}"))
    print("Done.", flush=True)


main()
