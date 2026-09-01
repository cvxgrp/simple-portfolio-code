"""Generate the two return forecasts used in the paper."""

from pathlib import Path

import pandas as pd

from simple_portfolio.alpha.geo_alpha import get_ridge_alpha_over_time

ASSETS = ["AGG", "SPY", "GLD"]

print("Started generating alphas...", flush=True)
processed = Path("data/processed")

closes = pd.read_parquet("data/raw/closes.parquet")
external_features = pd.read_parquet(processed / "ext_features.parquet")
etf_features = pd.read_parquet(processed / "etf_features.parquet")
etf_features = etf_features[
    [column for column in etf_features if any(f"_{asset}" in column for asset in ASSETS)]
]

simple_alpha = (
    closes[ASSETS]
    .pct_change()
    .ewm(halflife=252, min_periods=63)
    .mean()
    .reindex(columns=closes.columns, fill_value=0.0)
    .fillna(0.0)
)
simple_alpha.to_parquet(processed / "alpha_simple_sbg.parquet")

ridge_alpha = get_ridge_alpha_over_time(
    closes,
    external_features,
    etf_features,
    ASSETS,
    h=100,
    min_history=512,
    halflife=252,
    ridge=10.0,
    min_window=63,
    verbose=True,
)
ridge_alpha.to_parquet(processed / "alpha_ridge_sbg.parquet")
print("Done.", flush=True)
