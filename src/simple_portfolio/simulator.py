"""Performs portfolio backtesting."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from .optimizer import PortfolioConstructor

# Numerical tolerances for lot bookkeeping.
_DOLLAR_EPS = 1e-12
_SHARE_EPS = 1e-12
_WASH_SALE_DAYS = 30  # calendar days either side of a loss sale (IRC section 1091)


@dataclass
class _Lot:
    """A single tax lot for one asset (fractional shares allowed)."""

    acq_date: pd.Timestamp
    shares: float
    cumu_acq: float


def _sell_lots(
    asset_lots: list[_Lot],
    dollars_to_sell: float,
    cnow: float,
    ts: object,
    lt_rate: float,
    st_rate: float,
) -> tuple[float, float, float, float]:
    """Greedily realize ``dollars_to_sell`` of one asset from its lots."""
    if dollars_to_sell <= 0.0 or not asset_lots:
        return 0.0, 0.0, 0.0, 0.0

    ts_pd = pd.Timestamp(ts)

    def _is_long_term(lot: _Lot) -> bool:
        return (lot.acq_date + pd.DateOffset(years=1)) < ts_pd

    # Single-step optimal: sell lots in increasing per-share tax burden
    # rate * (cnow - basis), so the largest rate-weighted losses are realized
    # first and the largest rate-weighted gains are avoided.
    def _tax_burden(lot: _Lot) -> float:
        rate = lt_rate if _is_long_term(lot) else st_rate
        return rate * (cnow - lot.cumu_acq)

    ordered = sorted(asset_lots, key=_tax_burden)

    lt_gain = 0.0
    st_gain = 0.0
    lt_loss = 0.0
    st_loss = 0.0
    remaining = dollars_to_sell
    for lot in ordered:
        if remaining <= _DOLLAR_EPS:
            break
        lot_value = lot.shares * cnow
        take = min(remaining, lot_value)
        shares_taken = take / cnow
        gain = shares_taken * (cnow - lot.cumu_acq)
        if _is_long_term(lot):
            if gain >= 0.0:
                lt_gain += gain
            else:
                lt_loss += gain
        elif gain >= 0.0:
            st_gain += gain
        else:
            st_loss += gain
        lot.shares -= shares_taken
        remaining -= take

    asset_lots[:] = [lot for lot in asset_lots if lot.shares > _SHARE_EPS]
    return lt_gain, st_gain, lt_loss, st_loss


@dataclass
class _PendingLoss:
    """A realized loss whose deductibility is not yet settled under the wash-sale rule."""

    sale_date: pd.Timestamp
    proceeds: float
    lt_loss: float
    st_loss: float


class BacktestResults(NamedTuple):
    """Results of a backtest."""

    navs: pd.Series
    composition: pd.DataFrame
    turnover: pd.Series
    metadata: pd.Series

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestResults":
        """Load results from a dictionary."""
        return cls(
            navs=data["navs"],
            composition=data["composition"],
            turnover=data["turnover"],
            metadata=data["metadata"],
        )

    def to_dict(self) -> dict:
        """Save results to a dictionary."""
        return {
            "navs": self.navs,
            "composition": self.composition,
            "turnover": self.turnover,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save results to a pickle file."""
        with path.open("wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Path) -> "BacktestResults":
        """Load results from a pickle file."""
        with path.open("rb") as f:
            return cls.from_dict(pickle.load(f))  # noqa: S301


class BacktestData(NamedTuple):
    """Data for a backtest."""

    timeline: np.ndarray
    rebal_schedule: np.ndarray
    assets: np.ndarray
    returns: np.ndarray
    universe: np.ndarray
    ffrs: np.ndarray
    long_term_rate: float = 0.0
    short_term_rate: float = 0.0
    long_term_rates: np.ndarray | None = None
    dividend_rates: np.ndarray | None = None
    income_returns: np.ndarray | None = None

    @classmethod
    def from_pandas(
        cls,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        closes: pd.DataFrame,
        ffrs: pd.Series,
        rebal_freq: Literal["D", "W", "M", "Q", "Y"],
        long_term_rate: float = 0.0,
        short_term_rate: float = 0.0,
        long_term_rates: dict[str, float] | None = None,
        dividend_rates: dict[str, float] | None = None,
        income_returns: pd.DataFrame | None = None,
    ) -> "BacktestData":
        """Load data from a pandas DataFrame."""
        closes = closes.loc[start_date:end_date]
        ffrs = ffrs.loc[start_date:end_date]

        timeline = closes.index  # pd.DatetimeIndex
        assets = closes.columns.to_numpy()
        t_series = pd.Series(timeline)
        rebal_schedule = np.array(
            ~t_series.dt.to_period(rebal_freq).duplicated(keep="last"), dtype=bool
        )

        universe_np = closes.notna().to_numpy()

        # Filled returns: ffill then bfill closes, compute daily returns
        filled_closes = closes.ffill().bfill()
        prev_closes = filled_closes.shift(1)
        filled_returns_np = np.nan_to_num(
            (filled_closes - prev_closes).to_numpy() / prev_closes.to_numpy()
        )

        # FFR: zero on the first day (no carry before the simulation starts)
        ffrs_np = ffrs.to_numpy().flatten()
        ffrs_np[0] = 0.0

        times_np = timeline.to_numpy()

        lt_rates_np = np.full(len(assets), long_term_rate, dtype=float)
        div_rates_np = np.full(len(assets), long_term_rate, dtype=float)
        for position, asset in enumerate(assets):
            if long_term_rates is not None and asset in long_term_rates:
                lt_rates_np[position] = long_term_rates[asset]
            if dividend_rates is not None and asset in dividend_rates:
                div_rates_np[position] = dividend_rates[asset]

        income_np = np.zeros_like(filled_returns_np)
        if income_returns is not None:
            aligned = (
                income_returns.reindex(index=timeline, columns=closes.columns)
                .fillna(0.0)
                .to_numpy()
            )
            income_np = aligned

        return cls(
            timeline=times_np,
            rebal_schedule=rebal_schedule,
            assets=assets,
            universe=universe_np,
            returns=filled_returns_np,
            ffrs=ffrs_np,
            long_term_rate=long_term_rate,
            short_term_rate=short_term_rate,
            long_term_rates=lt_rates_np,
            dividend_rates=div_rates_np,
            income_returns=income_np,
        )


def run_backtest(
    backtest_name: str,
    data: BacktestData,
    portfolio_constructor: PortfolioConstructor,
    bid_ask_spread: float = 5e-4,
) -> BacktestResults:
    """Run a backtest for a given portfolio strategy."""
    timeline = data.timeline
    rebal_schedule = data.rebal_schedule
    assets = data.assets
    ffr_1p_np = 1.0 + data.ffrs
    filled_returns_1p_np = 1.0 + data.returns
    universes_np = data.universe
    n_time = len(data.timeline)
    n_assets = len(assets)
    st_rate = data.short_term_rate
    lt_rates = (
        np.full(n_assets, data.long_term_rate, dtype=float)
        if data.long_term_rates is None
        else np.asarray(data.long_term_rates, dtype=float)
    )
    div_rates = (
        np.full(n_assets, data.long_term_rate, dtype=float)
        if data.dividend_rates is None
        else np.asarray(data.dividend_rates, dtype=float)
    )
    income_np = (
        np.zeros((n_time, n_assets))
        if data.income_returns is None
        else np.asarray(data.income_returns, dtype=float)
    )
    taxable = bool(st_rate) or bool(np.any(lt_rates)) or bool(np.any(div_rates))

    metadata = pd.Series(
        {
            "name": backtest_name,
            "start_date": data.timeline[0],
            "end_date": data.timeline[-1],
        }
    )

    # Pre-allocate output arrays
    navs_np = np.empty(n_time)
    composition_np = np.zeros((n_time, len(assets)))
    turnover_np = np.zeros(n_time, dtype=float)

    potential_liquidation_np: np.ndarray = ~np.all(universes_np, axis=1)

    cash = 1.0
    holdings_np = np.zeros(len(assets))
    weights_np = np.zeros(len(assets))  # reused buffer — never reassigned
    first_solve = True

    # Lot tracking: synthetic per-asset price index and per-asset lot lists.
    cumu_rets = np.ones(len(assets))
    lots: list[list[_Lot]] = [[] for _ in range(len(assets))]

    # Wash-sale bookkeeping: losses awaiting the close of their forward window, and
    # the most recent purchase of each asset for the backward half of the window.
    pending: list[list[_PendingLoss]] = [[] for _ in range(n_assets)]
    last_buy: list[tuple[pd.Timestamp, float, _Lot] | None] = [None] * n_assets

    def settle_matured(a: int, now: pd.Timestamp) -> float:
        """Return the (non-positive) tax effect of losses whose window has closed."""
        if not pending[a]:
            return 0.0
        effect = 0.0
        keep = []
        for p in pending[a]:
            if p.sale_date + pd.Timedelta(days=_WASH_SALE_DAYS) < now:
                effect += lt_rates[a] * p.lt_loss + st_rate * p.st_loss
            else:
                keep.append(p)
        pending[a] = keep
        return effect

    def disallow_against(a: int, dollars: float, now: pd.Timestamp, target: _Lot) -> None:
        """Disallow pending losses replaced by a purchase, rolling them into basis."""
        if dollars <= _DOLLAR_EPS or not pending[a]:
            return
        remaining = dollars
        disallowed = 0.0
        keep = []
        for p in pending[a]:
            window_open = p.sale_date + pd.Timedelta(days=_WASH_SALE_DAYS) >= now
            if remaining <= _DOLLAR_EPS or not window_open or p.proceeds <= _DOLLAR_EPS:
                keep.append(p)
                continue
            share = min(1.0, remaining / p.proceeds)
            disallowed -= (p.lt_loss + p.st_loss) * share
            remaining -= share * p.proceeds
            if share < 1.0:
                p.lt_loss *= 1.0 - share
                p.st_loss *= 1.0 - share
                p.proceeds *= 1.0 - share
                keep.append(p)
        pending[a] = keep
        if disallowed > _DOLLAR_EPS and target.shares > _SHARE_EPS:
            target.cumu_acq += disallowed / target.shares

    def record_loss(a: int, proceeds: float, lt_loss: float, st_loss: float, now) -> None:
        """Book a realized loss, applying the backward half of the wash-sale window."""
        now_ts = pd.Timestamp(now)
        if lt_loss >= -_DOLLAR_EPS and st_loss >= -_DOLLAR_EPS:
            return
        prior = last_buy[a]
        if prior is not None:
            bought_at, bought_dollars, bought_lot = prior
            within = now_ts - bought_at <= pd.Timedelta(days=_WASH_SALE_DAYS)
            if within and bought_lot.shares > _SHARE_EPS and proceeds > _DOLLAR_EPS:
                share = min(1.0, bought_dollars / proceeds)
                bought_lot.cumu_acq += -(lt_loss + st_loss) * share / bought_lot.shares
                lt_loss *= 1.0 - share
                st_loss *= 1.0 - share
        if lt_loss < -_DOLLAR_EPS or st_loss < -_DOLLAR_EPS:
            pending[a].append(_PendingLoss(now_ts, proceeds, lt_loss, st_loss))

    for i in range(n_time):
        universe_np = universes_np[i]

        # Interest accrued on cash at the FFR is realized as a short-term gain every
        # timestep (regardless of rebalancing) and taxed at the short-term rate.
        ffr_interest = cash * (ffr_1p_np[i] - 1.0)
        cash *= ffr_1p_np[i]
        cash -= st_rate * ffr_interest
        if taxable:
            # Distributions are income in the period received. Tax them, and step the
            # basis up by the amount taxed so the same dollars are not taxed again as
            # a capital gain when the position is sold.
            income_row = income_np[i]
            for a in np.nonzero(income_row)[0]:
                income = holdings_np[a] * income_row[a]
                if income <= _DOLLAR_EPS:
                    continue
                cash -= div_rates[a] * income
                step = cumu_rets[a] * income_row[a]
                for lot in lots[a]:
                    lot.cumu_acq += step
        holdings_np *= filled_returns_1p_np[i]
        cumu_rets *= filled_returns_1p_np[i]
        turnover_dollars = 0.0
        if potential_liquidation_np[i]:
            # Realize all lots of assets dropping out of the universe before
            # the holdings of those assets are zeroed.
            liq_mask = (~universe_np) & (holdings_np != 0.0)
            for a in np.nonzero(liq_mask)[0]:
                # Realize the full lot value (not holdings_np[a]) so float drift
                # never leaves a residual gain unrealized before clearing.
                lot_value = sum(lot.shares for lot in lots[a]) * cumu_rets[a]
                lt_g, st_g, lt_l, st_l = _sell_lots(
                    lots[a], lot_value, cumu_rets[a], timeline[i], lt_rates[a], st_rate
                )
                # Pay tax on the realized gains out of cash before this step's NAV
                # is marked, so liquidation tax hits the current NAV. Losses are
                # withheld until their wash-sale window closes.
                cash -= lt_rates[a] * lt_g + st_rate * st_g
                record_loss(a, lot_value, lt_l, st_l, timeline[i])
                lots[a].clear()

            prev_holdings_sum = holdings_np.sum()
            holdings_np *= universe_np
            liq_value = prev_holdings_sum - holdings_np.sum()
            cash += liq_value
            turnover_dollars = liq_value

        nav = cash + holdings_np.sum() - bid_ask_spread * turnover_dollars
        np.divide(holdings_np, nav, out=weights_np)
        navs_np[i] = nav
        turnover = turnover_dollars / nav

        if rebal_schedule[i] or first_solve:
            new_weights = portfolio_constructor(
                ts=timeline[i],
                curr_weights=weights_np,
                universe=data.universe[i],
                ffr=data.ffrs[i],
            )
            first_solve = False

            # Turnover counts traded (noncash) assets only; cash is excluded.
            new_cash_weight = 1.0 - new_weights.sum()
            tcost_base = np.abs(new_weights - weights_np).sum()
            turnover += 0.5 * tcost_base
            new_cash_weight -= 0.5 * bid_ask_spread * tcost_base

            # Lot bookkeeping: compare new target holdings against old holdings
            # (post-return, post-liquidation) and buy/sell the difference.
            new_holdings = new_weights * nav
            deltas = new_holdings - holdings_np
            ts_pd = pd.Timestamp(timeline[i])
            rebal_tax = 0.0
            for a in range(len(assets)):
                # Losses whose 30-day window closed before today become deductible.
                rebal_tax += settle_matured(a, ts_pd)
                d = deltas[a]
                if d > _DOLLAR_EPS:
                    bought = _Lot(ts_pd, d / cumu_rets[a], cumu_rets[a])
                    lots[a].append(bought)
                    # A purchase inside a pending loss's window disallows that loss,
                    # which is added to the basis of these replacement shares.
                    disallow_against(a, d, ts_pd, bought)
                    last_buy[a] = (ts_pd, d, bought)
                elif d < -_DOLLAR_EPS:
                    lt_g, st_g, lt_l, st_l = _sell_lots(
                        lots[a], -d, cumu_rets[a], timeline[i], lt_rates[a], st_rate
                    )
                    rebal_tax += lt_rates[a] * lt_g + st_rate * st_g
                    record_loss(a, -d, lt_l, st_l, ts_pd)

            # Pay rebalance tax out of cash going forward, mirroring how the
            # bid-ask tcost already reduces cash (reflected in the next NAV).
            cash = new_cash_weight * nav - rebal_tax
            np.multiply(new_weights, nav, out=holdings_np)
            weights_np[:] = new_weights

        composition_np[i] = weights_np
        turnover_np[i] = turnover

    # Terminal liquidation: realize all remaining lots at the final timestep and
    # subtract that tax from the terminal NAV. Even a fully passive holder must
    # liquidate (and pay the blanket gains tax) before the wealth can be used.
    final_ts = timeline[-1]
    terminal_tax = 0.0
    for a in range(len(assets)):
        if lots[a]:
            lot_value = sum(lot.shares for lot in lots[a]) * cumu_rets[a]
            lt_g, st_g, lt_l, st_l = _sell_lots(
                lots[a], lot_value, cumu_rets[a], final_ts, lt_rates[a], st_rate
            )
            terminal_tax += lt_rates[a] * lt_g + st_rate * st_g
            # No trading follows the terminal liquidation, so every loss realized
            # here, and every loss still pending, clears its wash-sale window.
            terminal_tax += lt_rates[a] * lt_l + st_rate * st_l
            lots[a].clear()
        for p in pending[a]:
            terminal_tax += lt_rates[a] * p.lt_loss + st_rate * p.st_loss
        pending[a].clear()
    navs_np[-1] -= terminal_tax

    composition = pd.DataFrame(composition_np, index=timeline, columns=assets)
    composition["Cash"] = 1.0 - composition.sum(axis=1)
    navs = pd.Series(navs_np, index=timeline)
    turnover = pd.Series(turnover_np, index=timeline)
    return BacktestResults(navs, composition, turnover, metadata)
