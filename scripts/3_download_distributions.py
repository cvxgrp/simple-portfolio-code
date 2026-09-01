"""Download the distribution history of SPY, AGG, and GLD for the tax analysis.

The backtests run on adjusted close prices, whose returns silently reinvest every
distribution, so the post-tax analysis of \\S4.2 would otherwise treat dividend and
coupon income as deferred capital appreciation. It is not: it is taxable in the year it
is received. This script writes the daily *income* component of each asset's total
return, so the simulator can tax it separately.

For each asset the income return on day ``t`` is the cash distribution per share with
ex-date ``t``, divided by the unadjusted close on the previous trading day.

Writes ``data/raw/income_returns.parquet``, a daily frame aligned to the trading
calendar of ``data/raw/closes.parquet``, with one column per asset.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

ASSETS = ["SPY", "AGG", "GLD"]
START_DATE = "2004-01-01"
END_DATE = "2026-01-01"
OUT_PATH = Path("data/raw/income_returns.parquet")


def main() -> None:
    closes = pd.read_parquet("data/raw/closes.parquet")
    calendar = closes.loc[START_DATE:END_DATE].index

    income = pd.DataFrame(0.0, index=calendar, columns=ASSETS)
    for asset in ASSETS:
        ticker = yf.Ticker(asset)

        # Unadjusted closes: the denominator must be the price a holder actually saw,
        # not a price restated for later distributions.
        raw = yf.download(asset, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
        price = raw["Close"]
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
        price.index = pd.DatetimeIndex(price.index).tz_localize(None).normalize()
        price = price.reindex(calendar).ffill()

        dividends = ticker.dividends
        if dividends.empty:
            print(f"{asset}: no distributions on record")
            continue
        dividends.index = pd.DatetimeIndex(dividends.index).tz_localize(None).normalize()
        dividends = dividends.groupby(level=0).sum()
        dividends = dividends.reindex(calendar).fillna(0.0)

        income[asset] = (dividends / price.shift(1)).fillna(0.0)
        paid = int((income[asset] > 0).sum())
        annual = income[asset].groupby(income.index.year).sum()
        print(
            f"{asset}: {paid} distribution days, "
            f"average annual income yield {annual.loc[2006:2025].mean():.2%}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    income.to_parquet(OUT_PATH)
    print(f"\nWrote {OUT_PATH} with shape {income.shape}")

    # Sanity check: total return should be price return plus income return.
    total = closes.loc[calendar, ASSETS].pct_change()
    print("\nCheck: mean |total return - (price return + income return)| by asset")
    for asset in ASSETS:
        raw = yf.download(asset, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
        price = raw["Close"]
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
        price.index = pd.DatetimeIndex(price.index).tz_localize(None).normalize()
        price = price.reindex(calendar).ffill()
        residual = (total[asset] - (price.pct_change() + income[asset])).abs()
        print(f"  {asset}: {residual.mean():.2e} (max {residual.max():.2e})")


main()
