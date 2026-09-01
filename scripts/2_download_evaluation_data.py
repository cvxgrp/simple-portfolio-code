"""Download evaluation-only closes extending past the simulation window.

The alpha figure and the alpha-target correlation table need the realized
100-day forward return for every day of the simulation window; for the last
100 trading days of 2025 this requires closes from early 2026, which the main
pipeline (scripts/1_download_data.py, END_DATE 2026-01-01) deliberately does
not download. This script pulls SPY/AGG/GLD closes through today into a
separate file, data/raw/closes_eval.parquet, that is used only by the
evaluation plots and tables in scripts/4_metrics_and_plots.py -- the
simulation pipeline and its inputs are untouched.
"""

from pathlib import Path

import yfinance as yf

ASSETS = ["SPY", "AGG", "GLD"]
START_DATE = "1995-01-01"

out_path = Path("data/raw/closes_eval.parquet")

closes = yf.download(ASSETS, start=START_DATE, auto_adjust=True)["Close"][ASSETS]
closes = closes.loc[closes["SPY"].dropna().index]  # trading days, as in the main pipeline
closes.to_parquet(out_path)
print(
    f"Saved {out_path}: {len(closes)} rows, {closes.index[0].date()} to {closes.index[-1].date()}"
)
