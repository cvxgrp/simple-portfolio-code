"""Tests for portfolio constructors."""

import unittest

import numpy as np
import pandas as pd

from simple_portfolio.optimizer import AnchoredVolControlPortfolioConstructor


class TestAnchoredVolControlPortfolioConstructor(unittest.TestCase):
    """Tests for the anchored Markowitz objective terms."""

    def make_constructor(
        self, alphas: np.ndarray, *, spread: float = 0.0, cash_days: int | None = None
    ) -> tuple[AnchoredVolControlPortfolioConstructor, pd.Timestamp]:
        """Build a loose-risk, two-asset problem for objective tests."""
        ts = pd.Timestamp("2020-01-31")
        risk_model = {
            "Fs": {ts: np.zeros((2, 1))},
            "D_halves": {ts: np.full(2, 1e-3)},
        }
        constructor = AnchoredVolControlPortfolioConstructor(
            ts_lookup={ts: 0},
            alphas=alphas[None, :],
            risk_model=risk_model,
            vol_target=1.0,
            anchor=np.array([0.5, 0.5]),
            bid_ask_spread=spread,
            cash_rate_horizon_days=cash_days,
        )
        return constructor, ts

    def solve(
        self,
        constructor: AnchoredVolControlPortfolioConstructor,
        ts: pd.Timestamp,
        current: np.ndarray,
        daily_ffr: float = 0.0,
    ) -> np.ndarray:
        """Solve a fully available synthetic portfolio."""
        return constructor(ts, current, np.ones(2, dtype=bool), daily_ffr)

    def test_monthly_cash_return_can_dominate_risky_assets(self) -> None:
        """Cash is selected when its compounded horizon return exceeds alpha."""
        constructor, ts = self.make_constructor(np.array([0.001, 0.001]), cash_days=21)
        weights = self.solve(constructor, ts, np.zeros(2), daily_ffr=1e-4)
        np.testing.assert_allclose(weights, 0.0, atol=1e-6)

    def test_half_spread_penalty_uses_current_weights(self) -> None:
        """A forecast edge smaller than the half-spread does not justify trading."""
        current = np.array([0.5, 0.5])
        constructor, ts = self.make_constructor(
            np.array([0.0011, 0.0010]), spread=5e-4, cash_days=21
        )
        weights = self.solve(constructor, ts, current)
        np.testing.assert_allclose(weights, current, atol=1e-6)

    def test_invalid_objective_parameters_are_rejected(self) -> None:
        """Financial objective parameters must have valid domains."""
        with self.assertRaises(ValueError):
            self.make_constructor(np.ones(2), spread=-1e-4)
        with self.assertRaises(ValueError):
            self.make_constructor(np.ones(2), cash_days=0)


if __name__ == "__main__":
    unittest.main()
