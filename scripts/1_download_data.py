"""Download raw data and compute shared processed artifacts."""

import io
import pickle
import re
import time
import warnings
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Parameters ────────────────────────────────────────────────────────────────

START_DATE = "1995-01-01"
END_DATE = "2026-01-01"

VOL_HALFLIFE = 42
COV_HALFLIFE = 126
UNIVERSE_MIN_PERIODS = 126
MIN_UNIVERSE_SIZE = 15
K_FACTORS = 8

# NEW: FRED series to use as macro/market-state features.
FRED_FEATURE_SERIES = {
    # Rates
    "DGS3MO": "y_3m",
    "DGS2": "y_2y",
    "DGS5": "y_5y",
    "DGS10": "y_10y",
    "DGS30": "y_30y",
    "DFII10": "real_y_10y",
    "T10YIE": "breakeven_10y",
    # Risk / FX
    "VIXCLS": "vix",
    "DTWEXBGS": "dollar_index",
}

# ── Stage 1: Download raw data ─────────────────────────────────────────────────

raw_data_dir = Path("data/raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

print("=== Stage 1: Downloading raw data ===")

print("Downloading ETF data...")
etf_path = Path("data/universe.txt")
with etf_path.open("r") as f:
    etfs = f.read().splitlines()
etfs = [etf for etf in etfs if etf and not etf.startswith("#")]
yf_data = yf.download(etfs, start=START_DATE, end=END_DATE, auto_adjust=True)
closes = yf_data["Close"]
volumes = yf_data["Volume"]
trading_days = closes["SPY"].dropna().index
closes = closes.loc[trading_days]
volumes = volumes.loc[trading_days]
closes.to_parquet(raw_data_dir / "closes.parquet")
volumes.to_parquet(raw_data_dir / "volumes.parquet")
print("Saved closes.parquet and volumes.parquet")


fred_key_path = Path("keys/fred_api.key")
if not fred_key_path.exists():
    raise FileNotFoundError(
        f"FRED API key not found at '{fred_key_path}'. Please write your key to that file."
    )
fred = Fred(api_key=fred_key_path.read_text().strip())


def get_fred_series_with_retry(series_id: str, attempts: int = 5) -> pd.Series:
    """Fetch a FRED series, retrying transient gateway/API failures."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=END_DATE,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts:
                break
            wait_seconds = 2**attempt
            print(f"FRED download for {series_id} failed ({exc}); retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"FRED download for {series_id} failed after {attempts} attempts."
    ) from last_error


def _download_french_daily_zip(zip_filename: str) -> pd.DataFrame:
    """Download one Kenneth French daily factor zip and return decimal daily returns."""
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{zip_filename}"

    with urlopen(url) as response:  # noqa: S310
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        raw = zf.read(csv_name).decode("latin1")

    # Find the header line: the line just before rows beginning with YYYYMMDD.
    lines = raw.splitlines()
    first_data_line = next(i for i, line in enumerate(lines) if re.match(r"^\s*\d{8}\s*,", line))
    header_line = first_data_line - 1

    ff = pd.read_csv(io.StringIO(raw), skiprows=header_line)

    # Keep rows whose first column is a YYYYMMDD date.
    date_col = ff.columns[0]
    ff = ff[ff[date_col].astype(str).str.strip().str.match(r"^\d{8}$")].copy()

    ff[date_col] = pd.to_datetime(
        ff[date_col].astype(str).str.strip(),
        format="%Y%m%d",
    )
    ff = ff.set_index(date_col)

    ff.columns = [c.strip().replace("-", "_").replace(" ", "_").lower() for c in ff.columns]

    # Kenneth French files are in percent units.
    ff = ff.astype(float) / 100.0

    return ff


def get_fama_french_factors_daily(include_rf: bool = False) -> pd.DataFrame:
    """Download daily Fama-French 8 factors and return decimal daily returns.

    Factors returned by default:
        mkt_rf, smb, hml, rmw, cma, mom, st_rev, lt_rev

    If include_rf=True, also includes:
        rf
    """
    ff5 = _download_french_daily_zip("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    mom = _download_french_daily_zip("F-F_Momentum_Factor_daily_CSV.zip")
    st_rev = _download_french_daily_zip("F-F_ST_Reversal_Factor_daily_CSV.zip")
    lt_rev = _download_french_daily_zip("F-F_LT_Reversal_Factor_daily_CSV.zip")

    # Standardize column names in case French uses slightly different spelling.
    rename_map = {
        "mkt_rf": "mkt_rf",
        "smb": "smb",
        "hml": "hml",
        "rmw": "rmw",
        "cma": "cma",
        "rf": "rf",
        "mom": "mom",
        "st_rev": "st_rev",
        "lt_rev": "lt_rev",
    }

    ff5 = ff5.rename(columns=rename_map)
    mom = mom.rename(columns=rename_map)
    st_rev = st_rev.rename(columns=rename_map)
    lt_rev = lt_rev.rename(columns=rename_map)

    factors = pd.concat(
        [
            ff5[["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]],
            mom[["mom"]],
            st_rev[["st_rev"]],
            lt_rev[["lt_rev"]],
        ],
        axis=1,
        join="inner",
    ).sort_index()

    factor_cols = [
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
        "st_rev",
        "lt_rev",
    ]

    if include_rf:
        return factors[[*factor_cols, "rf"]]

    return factors[factor_cols]


print("Downloading Fed Funds Rate...")
ffr_raw = get_fred_series_with_retry("DFF").to_frame("FFR")
ffr_aligned = ffr_raw.reindex(closes.index).ffill()
ffr_daily = (1 + ffr_aligned / 100) ** (1 / 252) - 1.0
ffr_daily.to_parquet(raw_data_dir / "ffr.parquet")
print("Saved ffr.parquet")

cpi_raw = get_fred_series_with_retry("CPILFESL").to_frame("CPI")
union_index = closes.index.union(cpi_raw.index)
cpi_daily_ml = (1 + cpi_raw.shift(1).pct_change()) ** (1 / 21)
cpi_daily_econ = (1 + cpi_raw.pct_change()) ** (1 / 21)
cpi_daily_ml = cpi_daily_ml.reindex(union_index).ffill().reindex(closes.index)
cpi_daily_econ = cpi_daily_econ.reindex(union_index).ffill().reindex(closes.index)
cpi_daily_ml.to_parquet(raw_data_dir / "cpi_ml.parquet")
cpi_daily_econ.to_parquet(raw_data_dir / "cpi_econ.parquet")
print("Saved cpi_ml.parquet and cpi_econ.parquet")


# NEW ── Stage 1b: Download macro feature data ─────────────────────────────────

print("Downloading macro feature data...")

macro_raw = {}

for series_id, name in FRED_FEATURE_SERIES.items():
    macro_raw[name] = get_fred_series_with_retry(series_id)

macro_raw = pd.DataFrame(macro_raw)
macro_raw.index = pd.to_datetime(macro_raw.index)
macro_raw = macro_raw.reindex(trading_days).ffill()

macro_raw.to_parquet(raw_data_dir / "macro_raw.parquet")
print("Saved macro_raw.parquet")


# NEW ── Stage 1b.2: Download Fama-French factor data ──────────────────────────

print("Downloading Fama-French factor data...")

ff_factors_raw = get_fama_french_factors_daily()
ff_factors_raw = ff_factors_raw.loc[START_DATE:END_DATE]
ff_factors_raw = ff_factors_raw.reindex(trading_days).ffill()

ff_factors_raw.to_parquet(raw_data_dir / "fama_french_raw.parquet")
print("Saved fama_french_raw.parquet")


# NEW ── Stage 1c: Build features and next-day targets ─────────────────────────

print("\n=== Stage 1c: Building features ===")

proc_data_dir = Path("data/processed")
proc_data_dir.mkdir(parents=True, exist_ok=True)

# Features are used in natural units: log transforms on positive heavy-tailed
# series, trailing-mean smoothing of noisy series, and a few constructed
# features. No ad-hoc scaling or centering — the factor-model fit is invariant
# to per-feature rescaling and handles scale internally.

returns = closes.pct_change()

# ETF-derived features.
feature_blocks = {}

for h in [63, 126, 252]:
    feature_blocks[f"ret_{h}d"] = closes.pct_change(h)  # trailing return
    feature_blocks["vol_adjusted_ret"] = (
        closes.pct_change(h) / returns.rolling(h).std()
    )  # volatility-adjusted return

rets = closes.pct_change()

vol_21 = rets.rolling(21).std()
vol_252 = rets.rolling(252).std()
feature_blocks["vol_21_vs_252"] = np.log(vol_21) - np.log(vol_252)

feature_blocks["log volume_21d"] = np.log(volumes).rolling(21).mean()
feature_blocks["log volume_63d"] = np.log(volumes).rolling(63).mean()
feature_blocks["log volume_126d"] = np.log(volumes).rolling(126).mean()

etf_features = pd.concat(feature_blocks, axis=1)
etf_features.columns = [f"{feature}__{ticker}" for feature, ticker in etf_features.columns]

# Macro features.
macro_features = pd.DataFrame(index=closes.index)

for c in ["y_3m", "y_2y", "y_5y", "y_10y", "y_30y", "real_y_10y", "breakeven_10y"]:
    macro_features[c] = macro_raw[c]  # yield levels
macro_features["log_vix"] = np.log(macro_raw["vix"]).rolling(63).mean()
macro_features["dollar_index"] = macro_raw["dollar_index"].rolling(512).mean()
macro_features["slope_3m_2y"] = macro_features["y_2y"] - macro_features["y_3m"]


# Fama-French features: trailing means of the daily factor returns.
ff_features = ff_factors_raw.add_prefix("ff_").rolling(63).mean()

# Final feature matrix.
ext_features = pd.concat([macro_features, ff_features], axis=1)

ext_features = ext_features.reindex(trading_days)
ext_features.to_parquet(proc_data_dir / "ext_features.parquet")

etf_features = etf_features.reindex(trading_days)
etf_features.to_parquet(proc_data_dir / "etf_features.parquet")

print("Saved features")


# ── Stage 2: Full-universe risk model ─────────────────────────────────────────

print("\n=== Stage 2: Building full-universe risk model ===")

returns = closes.pct_change().dropna(how="all")
assets = returns.columns
factors = list(range(K_FACTORS))

# Fs: dict = {}
# D_halves: dict = {}

# for ts in tqdm(returns.index[UNIVERSE_MIN_PERIODS:], desc="Computing Fs and D_halves"):
#     returns_so_far = returns.loc[:ts].copy()
#     universe = returns_so_far.tail(UNIVERSE_MIN_PERIODS).dropna(axis=1).columns
#     if len(universe) < MIN_UNIVERSE_SIZE:
#         continue
#     returns_so_far = returns_so_far[universe].dropna()
#     F, D_half = get_F_D_half(
#         returns_so_far,
#         k=K_FACTORS,
#         vol_halflife=VOL_HALFLIFE,
#         cov_halflife=COV_HALFLIFE,
#     )
#     Fs[ts] = F.reindex(index=assets, columns=factors, fill_value=0.0)
#     D_halves[ts] = D_half.reindex(index=assets, fill_value=0.0)

Fs_sbg: dict = {}
D_halves_sbg: dict = {}

sbg_tickers = ["SPY", "AGG", "GLD"]
sbg_factors = list(range(len(sbg_tickers)))
# Trailing window (trading days) for the SPY/AGG/GLD covariance estimate. Short
# by conventional standards, but sweeping 11/21/42/63/126 days showed that the
# more reactive estimate improves return, Sharpe, and drawdown monotonically,
# at the cost of higher turnover.
SBG_HALFLIFE = 11

for ts in tqdm(returns.index[SBG_HALFLIFE:], desc="SPY/AGG/GLD Fs and D_halves"):
    returns_so_far = returns.loc[:ts, sbg_tickers].dropna().copy()

    if len(returns_so_far) < SBG_HALFLIFE:
        continue

    vols = np.sqrt(np.square(returns_so_far.tail(SBG_HALFLIFE)).mean())
    corr = returns_so_far.tail(SBG_HALFLIFE).div(vols, axis=1).cov()
    F_corr = np.linalg.cholesky(corr.to_numpy())

    F = pd.DataFrame(
        data=np.diag(vols.to_numpy()) @ F_corr,
        index=vols.index,
        columns=range(len(vols)),
    )
    D_half = pd.Series(data=1e-7, index=vols.index, dtype=float)
    Fs_sbg[ts] = F.reindex(index=assets, columns=sbg_factors, fill_value=0.0)
    D_halves_sbg[ts] = D_half.reindex(index=assets, fill_value=0.0)

risk_model_dir = proc_data_dir / "risk_model"
risk_model_dir.mkdir(parents=True, exist_ok=True)

# with (risk_model_dir / f"full_K{K_FACTORS}.pkl").open("wb") as f:
#     pickle.dump({"Fs": Fs, "D_halves": D_halves}, f)

with (risk_model_dir / "sbg.pkl").open("wb") as f:
    pickle.dump({"Fs": Fs_sbg, "D_halves": D_halves_sbg}, f)

print("Risk models complete.")

print("Setup complete.")
