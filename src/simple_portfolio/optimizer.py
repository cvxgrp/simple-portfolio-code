"""Factor-based portfolio optimizer."""

from typing import Protocol

import cvxpy as cp
import numpy as np
import pandas as pd

# ruff: noqa: ARG002


EPSILON = 1e-6


class PortfolioConstructor(Protocol):
    """Portfolio constructor."""

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        raise NotImplementedError


class FixedWeightPortfolioConstructor:
    """Fixed weight portfolio constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        fixed_weights: np.ndarray,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the fixed weight portfolio constructor."""
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(fixed_weights), 1))

        self.leverage = leverage
        self.ts_lookup = ts_lookup
        self.fixed_weights = fixed_weights
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        weights = self.fixed_weights[idx]
        universe = universe if self.universe is None else universe * self.universe[idx]

        unnorm_weights = weights * universe
        if np.abs(unnorm_weights).sum() < EPSILON:
            return np.zeros_like(weights)

        return self.leverage * unnorm_weights / np.abs(unnorm_weights).sum()


class FixedWeightVolControlPortfolioConstructor:
    """Fixed weight portfolio with volatility control constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        vols: np.ndarray,
        fixed_weights: np.ndarray,
        vol_target: float,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the fixed weight portfolio with volatility control constructor."""
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(fixed_weights), 1))

        self.leverage = leverage
        self.vol_target = vol_target
        self.ts_lookup = ts_lookup
        self.vols = vols
        self.fixed_weights = fixed_weights
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        fixed_weights = self.fixed_weights[idx]

        universe = universe if self.universe is None else universe * self.universe[idx]

        if np.abs(fixed_weights[universe]).sum() < EPSILON:
            return np.zeros_like(fixed_weights)

        weights = (fixed_weights * universe) / np.abs(fixed_weights[universe]).sum()
        vol = self.vols[idx]

        if vol < EPSILON:
            return np.zeros_like(fixed_weights)

        scaling = np.clip(self.vol_target / vol, 0.0, self.leverage)
        return weights * scaling


class FixedWeightMatrixVolControlPortfolioConstructor:
    """Fixed weight portfolio with volatility control constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        fixed_weights: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the fixed weight portfolio with volatility control constructor."""
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(fixed_weights), 1))

        self.leverage = leverage
        self.vol_target = vol_target
        self.ts_lookup = ts_lookup
        self.fixed_weights = fixed_weights
        self.risk_model = risk_model
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        fixed_weights = self.fixed_weights[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]

        if np.abs(fixed_weights[universe]).sum() < EPSILON:
            return np.zeros_like(fixed_weights)

        weights = (fixed_weights * universe) / np.abs(fixed_weights[universe]).sum()
        f = np.concatenate([F.T @ weights, D_half * weights])
        total_vol = np.linalg.norm(f)

        if total_vol < EPSILON:
            return np.zeros_like(fixed_weights)

        scaling = np.clip(self.vol_target / total_vol, 0.0, self.leverage)
        return weights * scaling


class VolControlPortfolioConstructor:
    """Volatility control portfolio constructor."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the volatility control portfolio constructor."""
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(alphas), 1))

        self.leverage = leverage
        self.vol_target = vol_target
        self.ts_lookup = ts_lookup
        self.risk_model = risk_model
        self._alphas = alphas
        self.n = None
        self.k = None
        self.Sig_half = None
        self.alpha_param = None
        self.mask = None
        self.weights = None
        self.problem = None
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        if (n != self.n) or (k != self.k):
            self.n = n
            self.k = k
            self.Sig_half = cp.Parameter(shape=(n + k, n))
            self.alpha_param = cp.Parameter(shape=(n,))
            self.mask = cp.Parameter(shape=(n,))
            self.weights = cp.Variable(n)
            self.objective = cp.Maximize(cp.scalar_product(self.alpha_param, self.weights))
            self.factor_risk = cp.norm2(self.Sig_half @ self.weights)
            self.constraints = [
                cp.sum(self.weights) <= self.leverage,
                self.weights >= 0.0,
                self.factor_risk <= self.vol_target,
                cp.multiply(self.weights, self.mask) == 0.0,
            ]
            self.problem = cp.Problem(self.objective, self.constraints)

        self.Sig_half.value = np.vstack([F.T, np.diag(D_half)])
        self.alpha_param.value = alphas
        self.mask.value = (~universe).astype(float)
        self.problem.solve(solver=cp.CLARABEL, verbose=False)
        return self.weights.value


class AnchoredVolControlPortfolioConstructor(VolControlPortfolioConstructor):
    """Anchored volatility control with optional cash and trading-cost terms."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        anchor: np.ndarray,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
        bid_ask_spread: float = 0.0,
        cash_rate_horizon_days: int | None = None,
    ) -> None:
        """Initialize the anchored volatility control portfolio constructor."""
        super().__init__(
            ts_lookup=ts_lookup,
            alphas=alphas,
            risk_model=risk_model,
            vol_target=vol_target,
            universe=universe,
            leverage=leverage,
        )
        if bid_ask_spread < 0.0:
            raise ValueError("Bid-ask spread must be nonnegative.")
        if cash_rate_horizon_days is not None and cash_rate_horizon_days <= 0:
            raise ValueError("Cash-rate horizon must be positive.")
        self.anchor = anchor
        self.bid_ask_spread = bid_ask_spread
        self.cash_rate_horizon_days = cash_rate_horizon_days
        self.curr_weights_param = None
        self.cash_return_param = None

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        if (n != self.n) or (k != self.k):
            self.n = n
            self.k = k
            self.Sig_half = cp.Parameter(shape=(n + k, n))
            self.alpha_param = cp.Parameter(shape=(n,))
            self.mask = cp.Parameter(shape=(n,))
            self.curr_weights_param = cp.Parameter(shape=(n,))
            self.cash_return_param = cp.Parameter()
            self.weights = cp.Variable(n)
            gross = cp.sum(self.weights)
            cash = 1.0 - gross
            expected_net_return = cp.scalar_product(self.alpha_param, self.weights)
            expected_net_return += self.cash_return_param * cash
            expected_net_return -= (
                0.5 * self.bid_ask_spread * cp.norm1(self.weights - self.curr_weights_param)
            )
            self.objective = cp.Maximize(expected_net_return)
            self.factor_risk = cp.norm2(self.Sig_half @ self.weights)
            self.constraints = [
                gross <= self.leverage,
                self.weights >= 0.0,
                self.factor_risk <= self.vol_target,
                cp.multiply(self.weights, self.mask) == 0.0,
                cp.norm1(self.weights - self.anchor * gross) <= gross,
            ]
            self.problem = cp.Problem(self.objective, self.constraints)

        self.Sig_half.value = np.vstack([F.T, np.diag(D_half)])
        self.alpha_param.value = alphas
        self.mask.value = (~universe).astype(float)
        self.curr_weights_param.value = curr_weights
        self.cash_return_param.value = (
            0.0
            if self.cash_rate_horizon_days is None
            else np.expm1(self.cash_rate_horizon_days * np.log1p(ffr))
        )
        self.problem.solve(solver=cp.CLARABEL, verbose=False)
        return self.weights.value


class VolControlMinWeightsPortfolioConstructor:
    """Volatility control portfolio constructor with minimum weight constraints."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        min_weights: np.ndarray,
        rel_min_weights: bool = True,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the volatility control portfolio constructor."""
        if vol_target <= 0.0:
            raise ValueError("Volatility target must be positive.")
        if leverage <= 0.0:
            raise ValueError("Leverage must be positive.")
        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (len(alphas), 1))

        self.leverage = leverage
        self.vol_target = vol_target
        self.min_weights = min_weights
        self.rel_min_weights = rel_min_weights
        self.ts_lookup = ts_lookup
        self.risk_model = risk_model
        self._alphas = alphas
        self.n = None
        self.k = None
        self.Sig_half = None
        self.alpha_param = None
        self.mask = None
        self.weights = None
        self.problem = None
        self.universe = universe

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        if (n != self.n) or (k != self.k):
            self.n = n
            self.k = k
            self.Sig_half = cp.Parameter(shape=(n + k, n))
            self.alpha_param = cp.Parameter(shape=(n,))
            self.mask = cp.Parameter(shape=(n,))
            self.weights = cp.Variable(n)
            self.objective = cp.Maximize(cp.scalar_product(self.alpha_param, self.weights))
            self.factor_risk = cp.norm2(self.Sig_half @ self.weights)
            self.constraints = [
                cp.sum(self.weights) <= self.leverage,
                self.weights >= 0.0,
                self.factor_risk <= self.vol_target,
                cp.multiply(self.weights, self.mask) == 0.0,
            ]
            if self.rel_min_weights:
                self.constraints.append(self.weights >= self.min_weights * cp.sum(self.weights))
            else:
                self.constraints.append(self.weights >= self.min_weights)
            self.problem = cp.Problem(self.objective, self.constraints)

        self.Sig_half.value = np.vstack([F.T, np.diag(D_half)])
        self.alpha_param.value = alphas
        self.mask.value = (~universe).astype(float)
        self.problem.solve(solver=cp.CLARABEL, verbose=False)
        return self.weights.value


class Markowitz70PortfolioConstructor:
    """Factor-based Markowitz-style portfolio constructor (markowitz70)."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        max_weights: np.ndarray | None = None,
        universe: np.ndarray | None = None,
        bid_ask_spread: float = 5e-4,
        t_lim: float = 1.0,
        target_vol: float = 0.01,
        max_leverage: float = 1.5,
        min_weights: float = 0.0,
        kappa_risk: float = 0.02,
        gamma_risk: float = 1e1,
    ) -> None:
        """Initialize the Markowitz70 portfolio constructor."""
        if target_vol <= 0.0:
            raise ValueError("Target volatility must be positive.")
        if max_leverage <= 0.0:
            raise ValueError("Max leverage must be positive.")

        a = np.asarray(alphas, dtype=float)
        n_rows, n_assets = a.shape

        if universe is not None and universe.ndim == 1:
            universe = np.tile(universe, (n_rows, 1))
        if max_weights is not None:
            max_weights = np.asarray(max_weights, dtype=float)
            if max_weights.ndim == 1:
                max_weights = np.tile(max_weights, (n_rows, 1))

        self.ts_lookup = ts_lookup
        self._alphas = a
        self.risk_model = risk_model
        self._max_weights = (
            max_weights if max_weights is not None else np.ones((n_rows, n_assets), dtype=float)
        )
        self.universe = universe
        self.bid_ask_spread = bid_ask_spread
        self.t_lim = t_lim
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.min_weights = min_weights
        self.kappa_risk = kappa_risk
        self.gamma_risk = gamma_risk
        self.first_solve = True

        self.n: int | None = None
        self.k: int | None = None
        self.weights: cp.Variable | None = None
        self.cash: cp.Variable | None = None
        self.risk_slack: cp.Variable | None = None
        self.alpha_param: cp.Parameter | None = None
        self.curr_w_param: cp.Parameter | None = None
        self.ffr_param: cp.Parameter | None = None
        self.mask_param: cp.Parameter | None = None
        self.max_w_param: cp.Parameter | None = None
        self.Sig_half_param: cp.Parameter | None = None
        self.sqrt_Sig_diag_param: cp.Parameter | None = None
        self.curr_cash_param: cp.Parameter | None = None
        self.problem: cp.Problem | None = None

    def _ensure_problem(self, n: int, k: int) -> None:
        if self.n == n and self.k == k and self.problem is not None:
            return

        self.n = n
        self.k = k
        self.weights = cp.Variable(n)
        self.cash = cp.Variable()
        self.risk_slack = cp.Variable(nonneg=True)
        self.alpha_param = cp.Parameter(shape=(n,))
        self.curr_w_param = cp.Parameter(shape=(n,))
        self.ffr_param = cp.Parameter()
        self.mask_param = cp.Parameter(shape=(n,))
        self.max_w_param = cp.Parameter(shape=(n,))
        self.Sig_half_param = cp.Parameter(shape=(n + k, n))
        self.sqrt_Sig_diag_param = cp.Parameter(shape=(n,), nonneg=True)
        self.curr_cash_param = cp.Parameter()

        expected_returns = (
            cp.scalar_product(self.alpha_param, self.weights) + self.cash * self.ffr_param
        )
        sqrt_kappa = float(np.sqrt(self.kappa_risk))
        diag_risk_term = sqrt_kappa * cp.sum(
            cp.multiply(self.sqrt_Sig_diag_param, cp.abs(self.weights))
        )
        factor_risk = cp.norm2(
            cp.hstack([cp.norm2(self.Sig_half_param @ self.weights), diag_risk_term])
        )
        constraints: list[cp.Constraint] = [
            cp.norm1(self.weights) <= self.max_leverage,
            cp.sum(self.weights) + self.cash == 1.0,
            self.weights >= self.min_weights,
            self.weights <= self.max_w_param,
            factor_risk <= self.target_vol + self.risk_slack,
            cp.multiply(self.weights, self.mask_param) == 0.0,
        ]

        objective = expected_returns - self.gamma_risk * self.risk_slack
        if not self.first_solve:
            t_cost = self.bid_ask_spread * cp.norm1(self.weights - self.curr_w_param)
            objective -= t_cost
            w_diff = self.weights - self.curr_w_param
            double_turnover = cp.norm1(w_diff) + cp.abs(self.cash - self.curr_cash_param)
            constraints.append(double_turnover <= 2.0 * self.t_lim)

        self.problem = cp.Problem(cp.Maximize(objective), constraints)

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        max_weights = self._max_weights[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        self._ensure_problem(n, k)

        Sig_half = np.vstack([F.T, np.diag(D_half)])
        Sig_diag = np.sum(F**2, axis=1) + D_half**2

        self.Sig_half_param.value = Sig_half
        self.alpha_param.value = alphas
        self.curr_w_param.value = curr_weights
        self.ffr_param.value = ffr
        self.mask_param.value = (~universe).astype(float)
        self.max_w_param.value = max_weights
        self.sqrt_Sig_diag_param.value = np.sqrt(Sig_diag)
        self.curr_cash_param.value = float(1.0 - curr_weights.sum())

        self.problem.solve(solver=cp.CLARABEL, verbose=False)
        if self.weights.value is None:
            raise RuntimeError("weights variable missing after solve.")
        return self.weights.value


class AnchoredAlphaPortfolioConstructor(VolControlPortfolioConstructor):
    """Volatility control portfolio constructor with an l1 trust region."""

    def __init__(
        self,
        ts_lookup: dict[pd.Timestamp, int],
        alphas: np.ndarray,
        risk_model: dict[str, dict[pd.Timestamp, np.ndarray]],
        vol_target: float,
        anchor: np.ndarray,
        universe: np.ndarray | None = None,
        leverage: float = 1.0,
    ) -> None:
        """Initialize the anchored volatility control portfolio constructor."""
        super().__init__(
            ts_lookup=ts_lookup,
            alphas=alphas,
            risk_model=risk_model,
            vol_target=vol_target,
            universe=universe,
            leverage=leverage,
        )
        self.anchor = anchor

    def __call__(
        self,
        ts: pd.Timestamp,
        curr_weights: np.ndarray,
        universe: np.ndarray,
        ffr: float,
    ) -> np.ndarray:
        """Construct a portfolio."""
        idx = self.ts_lookup[ts]
        alphas = self._alphas[idx]
        F = self.risk_model["Fs"][ts]
        D_half = self.risk_model["D_halves"][ts]
        universe = universe if self.universe is None else universe * self.universe[idx]
        n, k = F.shape

        if (n != self.n) or (k != self.k):
            self.n = n
            self.k = k
            self.Sig_half = cp.Parameter(shape=(n + k, n))
            self.alpha_param = cp.Parameter(shape=(n,))
            self.mask = cp.Parameter(shape=(n,))
            self.weights = cp.Variable(n)
            gross = cp.sum(self.weights)
            self.objective = cp.Maximize(cp.scalar_product(self.alpha_param - ffr, self.weights))
            self.factor_risk = cp.norm2(self.Sig_half @ self.weights)
            self.constraints = [
                gross <= self.leverage,
                self.weights >= 0.0,
                cp.multiply(self.weights, self.mask) == 0.0,
            ]
            self.problem = cp.Problem(self.objective, self.constraints)

        self.Sig_half.value = np.vstack([F.T, np.diag(D_half)])
        self.alpha_param.value = alphas
        self.mask.value = (~universe).astype(float)
        self.problem.solve(solver=cp.CLARABEL, verbose=False)
        return self.weights.value
