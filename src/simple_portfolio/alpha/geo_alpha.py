"""Geometric factor-model alpha."""

import numpy as np
import pandas as pd


def geo_factor_decomp(
    G: np.ndarray, k: int, n_iters: int = 40, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a PSD matrix G as F F^T + diag(D) by alternating updates."""
    c = np.sqrt(np.maximum(np.diag(G), 1e-300))
    G_t = G / c[:, None] / c[None, :]
    D = np.ones(G.shape[0])
    F = np.zeros((G.shape[0], k))
    for _ in range(n_iters):
        d_half = np.sqrt(D)
        lam, V = np.linalg.eigh(G_t / d_half[:, None] / d_half[None, :])
        F = d_half[:, None] * (V[:, -k:] * np.sqrt(np.maximum(lam[-k:] - 1.0, 0.0)))
        D = np.maximum(np.diag(G_t - F @ F.T), eps)
    return c[:, None] * F, c**2 * D


def get_geo_alpha_over_time(
    closes: pd.DataFrame,
    ext_features: pd.DataFrame,
    etf_features: pd.DataFrame,
    assets: list[str] | None = None,
    n_factors: int = 3,
    h: int = 100,
    min_history: int = 512,
    halflife: int = 252,
    n_iters: int = 40,
    min_window: int = 63,
    verbose: bool = False,
) -> pd.DataFrame:
    """Get the geometric factor-model alpha over time."""
    if not closes.index.equals(ext_features.index):
        raise ValueError("Closes and external features must have the same index.")
    if not closes.index.equals(etf_features.index):
        raise ValueError("Closes and ETF features must have the same index.")
    if assets is None:
        assets = list(closes.columns)

    etf_features = etf_features[
        [col for col in etf_features.columns if any(f"_{asset}" in col for asset in assets)]
    ]
    features = pd.concat([ext_features, etf_features], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)

    targets = closes[assets].pct_change(h).shift(-h) * 252 / h
    targets = targets.replace([np.inf, -np.inf], np.nan)

    X = features.to_numpy(dtype=np.float64)
    Y = targets.to_numpy(dtype=np.float64)
    n_time, n_assets = Y.shape

    # run_len[j]: number of consecutive complete (all-asset) target rows ending
    # at row j inclusive.
    valid = ~np.isnan(Y).any(axis=1)
    run_len = np.zeros(n_time, dtype=int)
    for j in range(n_time):
        run_len[j] = run_len[j - 1] + 1 if valid[j] and j > 0 else int(valid[j])

    # Full-length exponential weights; each window uses the trailing slice so
    # that the newest observation always has the largest weight.
    w_full = np.exp(-np.log(2) / halflife * np.arange(min_history)[::-1])

    out = np.zeros_like(Y)
    for i in range(min_window + h - 1, n_time):
        hi = i + 1 - h
        window = min(run_len[hi - 1], min_history)
        if window < min_window:
            continue
        lo = hi - window
        Yw = Y[lo:hi]
        x_now = X[i]
        Xw = X[lo:hi]
        ft_mask = ~np.isnan(Xw).any(axis=0) & ~np.isnan(x_now)
        if ft_mask.sum() < n_factors:
            continue
        row_w = w_full[-window:] / w_full[-window:].sum()
        Z = np.hstack([Yw, Xw[:, ft_mask]])
        G = (Z * row_w[:, None]).T @ Z
        F, D = geo_factor_decomp(G, n_factors, n_iters=n_iters)
        F_1, F_2 = F[:n_assets], F[n_assets:]
        D_2 = D[n_assets:]
        A = F_2.T @ (F_2 / D_2[:, None])
        w = np.linalg.solve(A, F_2.T @ (x_now[ft_mask] / D_2))
        out[i] = F_1 @ w
        if verbose and i % 500 == 0:
            print(f"{targets.index[i].date()}: alpha = {np.round(out[i], 4)}")

    alphas = pd.DataFrame(out, index=targets.index, columns=targets.columns)
    return alphas.reindex(columns=closes.columns, fill_value=0.0)


def get_ridge_alpha_over_time(
    closes: pd.DataFrame,
    ext_features: pd.DataFrame,
    etf_features: pd.DataFrame,
    assets: list[str] | None = None,
    h: int = 100,
    min_history: int = 512,
    halflife: int = 252,
    ridge: float = 1e-3,
    min_window: int = 63,
    verbose: bool = False,
) -> pd.DataFrame:
    """Get exponentially weighted ridge-regression alpha over time."""
    if not closes.index.equals(ext_features.index):
        raise ValueError("Closes and external features must have the same index.")
    if not closes.index.equals(etf_features.index):
        raise ValueError("Closes and ETF features must have the same index.")
    if h <= 0:
        raise ValueError("h must be positive.")
    if min_history <= 0:
        raise ValueError("min_history must be positive.")
    if halflife <= 0:
        raise ValueError("halflife must be positive.")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative.")
    if min_window <= 0:
        raise ValueError("min_window must be positive.")

    if assets is None:
        assets = list(closes.columns)

    missing_assets = [asset for asset in assets if asset not in closes.columns]
    if missing_assets:
        raise ValueError(f"Assets not present in closes: {missing_assets}")

    etf_features = etf_features[
        [col for col in etf_features.columns if any(f"_{asset}" in col for asset in assets)]
    ]

    features = pd.concat(
        [ext_features, etf_features],
        axis=1,
    )
    features = features.replace([np.inf, -np.inf], np.nan)

    targets = closes[assets].pct_change(h, fill_method=None).shift(-h) * 252 / h
    targets = targets.replace([np.inf, -np.inf], np.nan)

    X = features.to_numpy(dtype=np.float64)
    Y = targets.to_numpy(dtype=np.float64)

    n_time, _ = Y.shape

    # run_len[j] is the number of consecutive complete all-asset target rows
    # ending at row j, inclusive.
    valid = ~np.isnan(Y).any(axis=1)
    run_len = np.zeros(n_time, dtype=int)

    for j in range(n_time):
        if not valid[j]:
            run_len[j] = 0
        elif j == 0:
            run_len[j] = 1
        else:
            run_len[j] = run_len[j - 1] + 1

    # Each window takes a trailing slice, so its newest observation receives
    # the largest weight.
    decay = np.log(2.0) / halflife
    ages = np.arange(min_history - 1, -1, -1)
    w_full = np.exp(-decay * ages)

    out = np.zeros_like(Y)

    for i in range(min_window + h - 1, n_time):
        # Targets in [lo, hi) are fully observable by prediction day i.
        hi = i + 1 - h

        if hi <= 0:
            continue

        window = min(run_len[hi - 1], min_history)

        if window < min_window:
            continue

        lo = hi - window

        Yw = Y[lo:hi]
        Xw = X[lo:hi]
        x_now = X[i]

        # Keep features observed throughout the window and on prediction day.
        ft_mask = ~np.isnan(Xw).any(axis=0) & ~np.isnan(x_now)

        if not ft_mask.any():
            continue

        feature_scale = np.maximum(np.std(Xw[:, ft_mask], axis=0), 1e-12)
        X_fit = Xw[:, ft_mask] / feature_scale
        x_pred = x_now[ft_mask] / feature_scale

        row_w = w_full[-window:].copy()
        row_w /= row_w.sum()

        # Exponentially weighted feature Gram and feature-target cross moment.
        gram = X_fit.T @ (row_w[:, None] * X_fit)
        cross = X_fit.T @ (row_w[:, None] * Yw)

        penalty = ridge * np.eye(X_fit.shape[1])

        try:
            beta = np.linalg.solve(
                gram + penalty,
                cross,
            )
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(
                gram + penalty,
                cross,
                rcond=None,
            )[0]

        out[i] = x_pred @ beta

        if verbose and i % 500 == 0:
            print(f"{targets.index[i].date()}: alpha = {np.round(out[i], 4)}")

    alphas = pd.DataFrame(
        out,
        index=targets.index,
        columns=targets.columns,
    )

    return alphas.reindex(
        columns=closes.columns,
        fill_value=0.0,
    )
