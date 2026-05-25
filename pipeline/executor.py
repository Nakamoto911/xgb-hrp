"""Drift-band executor + AVCO ledger — Module 7 of SPEC.md.

Walks the OOS window day-by-day. At each business day:

1. Mark portfolio to market.
2. Drain any pending risk-off action enqueued by yesterday's risk monitor
   (liquidate everything to risk-free, or re-enter per ``config.reenter_mode``).
3. Ask the risk monitor about today's universe-bear share. A trigger or a
   clearance is enqueued for tomorrow's close.
4. If today is a scheduled rebalance AND we're risk-on AND no action is
   pending, run the selector + allocator and trade to the target weights
   with drift-band gating (config.drift_threshold).
5. At year-end roll-over, settle the annual tax on net realized gains
   against the vintage-aged loss carryforward.

The ledger is **true AVCO**: transaction costs roll into the cost basis
(spec §9.3.1 / French PMP convention), partial sells leave avg_cost
unchanged, full sells reset it. Losses are tagged with the year they
realized and expire after ``config.loss_carryforward_years`` (default 10).
"""
from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.allocator import allocate
from pipeline.config import PipelineConfig
from pipeline.risk_monitor import RiskMonitor, Transition
from pipeline.selector import select

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# AVCO ledger
# -----------------------------------------------------------------------------
@dataclass
class AVCOLedger:
    """Average-cost ledger per asset. Cost basis stored as dollar total,
    units stored separately; avg_cost = basis / units.

    On a buy of Δu at price p with transaction cost tc (positive scalar):
        basis  += Δu * p + tc       # TC rolls into PMP (French convention)
        units  += Δu

    On a sell of Δu at price p:
        frac           = Δu / units
        basis_sold     = basis * frac
        realized_gain  = Δu * p - basis_sold
        basis  -= basis_sold
        units  -= Δu
        # avg_cost = basis/units is unchanged for a partial sell ✓
    """

    units: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    basis: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def buy(self, symbol: str, delta_units: float, price: float, tc: float) -> None:
        if delta_units <= 0:
            raise ValueError(f"buy expects delta_units > 0, got {delta_units}")
        self.units[symbol] += delta_units
        self.basis[symbol] += delta_units * price + tc

    def sell(self, symbol: str, delta_units: float, price: float) -> float:
        """Return the realized gain on the sold portion (gross, before tax/TC)."""
        if delta_units <= 0:
            raise ValueError(f"sell expects delta_units > 0, got {delta_units}")
        held = self.units.get(symbol, 0.0)
        if delta_units > held + 1e-9:
            raise ValueError(f"oversell {symbol}: held={held}, asked={delta_units}")
        frac = min(1.0, delta_units / held) if held > 0 else 1.0
        basis_sold = self.basis.get(symbol, 0.0) * frac
        proceeds = delta_units * price
        realized = proceeds - basis_sold
        self.units[symbol] = held - delta_units
        self.basis[symbol] = self.basis.get(symbol, 0.0) - basis_sold
        if self.units[symbol] <= 1e-9:
            # Full sale: zero out residuals.
            self.units[symbol] = 0.0
            self.basis[symbol] = 0.0
        return realized

    def avg_cost(self, symbol: str) -> float:
        u = self.units.get(symbol, 0.0)
        if u <= 0:
            return float("nan")
        return self.basis[symbol] / u


# -----------------------------------------------------------------------------
# Tax book — vintage-aged carryforward + annual settlement
# -----------------------------------------------------------------------------
@dataclass
class TaxBook:
    pfu_rate: float
    carryforward_years: int
    # ordered by vintage year (insertion order). Value is the *positive* loss amount.
    carryforward: "OrderedDict[int, float]" = field(default_factory=OrderedDict)
    # accumulators for the current calendar year
    year_realized: float = 0.0  # running sum of (gains - losses) for current year
    # historical record
    history: list[dict] = field(default_factory=list)

    def record_realized(self, year: int, amount: float) -> None:
        self.year_realized += amount  # positive=gain, negative=loss

    def settle_year(self, year: int) -> float:
        """Apply this year's net realized P&L to the carryforward and compute tax due.

        Returns the tax due (always >= 0). Side-effects update the carryforward
        ledger and append a row to ``history``.
        """
        net = self.year_realized
        tax_due = 0.0
        applied_carryforward = 0.0

        if net > 0:
            # Drain oldest carryforward first (FIFO across vintages).
            remaining = net
            to_drop: list[int] = []
            for vintage, loss_amt in self.carryforward.items():
                if remaining <= 0:
                    break
                used = min(loss_amt, remaining)
                self.carryforward[vintage] = loss_amt - used
                remaining -= used
                applied_carryforward += used
                if self.carryforward[vintage] <= 1e-9:
                    to_drop.append(vintage)
            for v in to_drop:
                del self.carryforward[v]
            tax_due = max(0.0, remaining) * self.pfu_rate
        elif net < 0:
            # Add this year's loss as a new vintage.
            self.carryforward[year] = self.carryforward.get(year, 0.0) + abs(net)

        # Expire old vintages.
        cutoff = year - self.carryforward_years
        expired: list[int] = [v for v in self.carryforward if v <= cutoff]
        for v in expired:
            del self.carryforward[v]

        self.history.append(
            {
                "year": year,
                "net_realized": net,
                "applied_carryforward": applied_carryforward,
                "tax_due": tax_due,
                "carryforward_outstanding": sum(self.carryforward.values()),
                "expired_vintages": expired,
            }
        )
        self.year_realized = 0.0
        return tax_due


# -----------------------------------------------------------------------------
# Backtest output bundle
# -----------------------------------------------------------------------------
@dataclass
class BacktestResult:
    nav: pd.Series                   # daily NAV (post-tax, post-TC)
    weights: pd.DataFrame            # date × symbol weight matrix at close
    trades: pd.DataFrame             # one row per executed leg
    risk_events: pd.DataFrame        # risk monitor transitions
    tax_history: pd.DataFrame        # year-end tax accruals
    total_tc: float
    total_tax: float
    final_carryforward: float


# -----------------------------------------------------------------------------
# Executor
# -----------------------------------------------------------------------------
@dataclass
class Executor:
    config: PipelineConfig
    prices: pd.DataFrame            # date × symbol close prices (incl. risk-free)
    forecast_panel: pd.DataFrame    # long-form forecast output from build_forecasts
    initial_capital: float = 100_000.0

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> set[pd.Timestamp]:
        freq = self.config.rebalance_frequency
        if freq == "daily":
            return set(index)
        freq_map = {
            "weekly": "W-FRI",
            "monthly": "BME",
            "quarterly": "BQE",
            "yearly": "BYE",
        }
        if freq == "semi-annually":
            # Pandas has no direct semi-annual business-end; take quarterly and keep Q2/Q4.
            anchors = pd.date_range(start=index.min(), end=index.max(), freq="BQE")
            anchors = anchors[anchors.month.isin((6, 12))]
        else:
            anchors = pd.date_range(start=index.min(), end=index.max(), freq=freq_map[freq])
        # Snap each anchor to the closest available business day on or before it.
        snapped: set[pd.Timestamp] = set()
        for a in anchors:
            valid = index[index <= a]
            if len(valid):
                snapped.add(valid[-1])
        return snapped

    def _p_bear_at(self, date: pd.Timestamp) -> dict[str, float]:
        sub = self.forecast_panel[self.forecast_panel["date"] == date]
        return dict(zip(sub["symbol"], sub["p_bear"], strict=False))

    def _selected_at(
        self, date: pd.Timestamp, returns: pd.DataFrame
    ) -> tuple[list[str], pd.Series]:
        """Run selector + allocator at ``date``. Returns (selected, weights)."""
        # Restrict to forecasts up to (and including) date.
        sub = self.forecast_panel[self.forecast_panel["date"] == date]
        if sub.empty:
            return [], pd.Series(dtype=float)
        sel_out = select(self.forecast_panel, self.config, on_dates=pd.DatetimeIndex([date]))
        selected = sel_out.by_date.get(date, [])
        # Keep only selected symbols that are tradeable (have a price today)
        tradeable_today = set(self.prices.columns[self.prices.loc[date].notna()])
        selected = [s for s in selected if s in tradeable_today]
        if not selected:
            return [], pd.Series(dtype=float)
        # Use returns up to t-1 to avoid using today's return in the cov.
        returns_up_to_yesterday = returns.loc[: date - pd.Timedelta(days=1)]
        w = allocate(returns_up_to_yesterday, selected, self.config)
        return selected, w

    def _trade_to_target(
        self,
        target_weights: pd.Series,
        prices_today: pd.Series,
        ledger: AVCOLedger,
        cash: float,
        portfolio_value: float,
        gate_drift: bool,
    ) -> tuple[float, list[dict], float, float]:
        """Adjust holdings toward ``target_weights`` (sum=1 over selected symbols).

        Drift-band gating applies symmetrically to every symbol in the universe
        currently or newly held. Returns (new_cash, trade_rows, total_tc, realized_gains).
        """
        bps = self.config.transaction_cost_bps / 1e4
        gate = self.config.drift_threshold if gate_drift else 0.0

        # Build "universe" = symbols I hold today ∪ symbols I'm targeting.
        held_symbols = {s for s, u in ledger.units.items() if u > 0}
        target_symbols = set(target_weights.index)
        universe = held_symbols | target_symbols

        # Two passes: sells first (to raise cash), then buys.
        trade_rows: list[dict] = []
        total_tc = 0.0
        realized_gains = 0.0

        def current_weight(s: str) -> float:
            if portfolio_value <= 0:
                return 0.0
            return ledger.units.get(s, 0.0) * prices_today.get(s, 0.0) / portfolio_value

        plan: list[tuple[str, float, float]] = []  # (symbol, delta_weight, target_weight)
        for s in universe:
            w_curr = current_weight(s)
            w_tgt = float(target_weights.get(s, 0.0))
            dw = w_tgt - w_curr
            if abs(dw) < gate:
                continue
            plan.append((s, dw, w_tgt))

        # Execute sells
        for s, dw, _w_tgt in plan:
            if dw >= 0:
                continue
            price = prices_today.get(s, np.nan)
            if not np.isfinite(price) or price <= 0:
                continue
            # Sell enough units to reduce weight by |dw|.
            delta_value = -dw * portfolio_value
            current_units = ledger.units.get(s, 0.0)
            delta_units = min(current_units, delta_value / price)
            if delta_units <= 1e-9:
                continue
            gross_proceeds = delta_units * price
            tc = gross_proceeds * bps
            realized = ledger.sell(s, delta_units, price)
            realized_gains += realized
            cash += gross_proceeds - tc
            total_tc += tc
            trade_rows.append(
                {"symbol": s, "side": "sell", "units": delta_units, "price": price,
                 "tc": tc, "realized_gain": realized}
            )

        # Execute buys (cash-constrained)
        for s, dw, w_tgt in plan:
            if dw <= 0:
                continue
            price = prices_today.get(s, np.nan)
            if not np.isfinite(price) or price <= 0:
                continue
            desired_value = dw * portfolio_value
            # Solve: gross + tc <= cash; gross = units*price; tc = gross*bps
            #        gross * (1+bps) <= cash → gross <= cash / (1+bps)
            max_gross = cash / (1.0 + bps)
            gross = min(desired_value, max_gross)
            if gross <= 1e-6:
                continue
            delta_units = gross / price
            tc = gross * bps
            ledger.buy(s, delta_units, price, tc)
            cash -= gross + tc
            total_tc += tc
            trade_rows.append(
                {"symbol": s, "side": "buy", "units": delta_units, "price": price,
                 "tc": tc, "realized_gain": 0.0}
            )

        return cash, trade_rows, total_tc, realized_gains

    def run(self) -> BacktestResult:
        cfg = self.config
        rf_symbol = cfg.risk_free_asset
        if rf_symbol not in self.prices.columns:
            raise ValueError(
                f"risk_free_asset {rf_symbol!r} missing from prices panel; "
                f"add it via load_risk_free + concat before running."
            )
        # Restrict to OOS window where the forecast panel has dates.
        oos_dates = sorted(set(self.forecast_panel["date"]) & set(self.prices.index))
        if not oos_dates:
            raise ValueError("forecast_panel and prices share no dates.")
        oos_index = pd.DatetimeIndex(oos_dates)
        prices = self.prices.loc[oos_index]
        returns = self.prices.pct_change(fill_method=None)

        ledger = AVCOLedger()
        tax = TaxBook(pfu_rate=cfg.pfu_rate, carryforward_years=cfg.loss_carryforward_years)
        monitor: Optional[RiskMonitor] = RiskMonitor(config=cfg) if cfg.risk_monitor_enabled else None
        rebal_dates = self._rebalance_dates(oos_index)

        cash = self.initial_capital
        nav_records: list[tuple[pd.Timestamp, float]] = []
        weight_records: list[dict] = []
        all_trades: list[dict] = []
        total_tc = 0.0
        total_tax = 0.0
        prev_year = oos_index[0].year
        pending_action: Optional[str] = None
        last_targets: pd.Series = pd.Series(dtype=float)

        for t in oos_index:
            prices_today = prices.loc[t]
            portfolio_value = cash + sum(
                u * prices_today.get(s, 0.0) for s, u in ledger.units.items() if u > 0
            )

            # 1) Drain pending risk-off action from yesterday's monitor.
            if pending_action == "liquidate":
                # Sell everything (including any residual non-rf positions) into rf.
                sells = [s for s, u in list(ledger.units.items()) if u > 0 and s != rf_symbol]
                # Sell, then buy rf with the freed cash.
                liquidation_targets = pd.Series({rf_symbol: 1.0})
                cash, trade_rows, tc, realized = self._trade_to_target(
                    liquidation_targets, prices_today, ledger, cash, portfolio_value,
                    gate_drift=False,
                )
                del sells  # diagnostic only
                for r in trade_rows:
                    r["date"] = t
                    r["reason"] = "risk_off_liquidate"
                all_trades.extend(trade_rows)
                total_tc += tc
                tax.record_realized(t.year, realized)
                pending_action = None

            elif pending_action == "reenter":
                if cfg.reenter_mode == "immediate_fresh":
                    selected, w = self._selected_at(t, returns)
                elif cfg.reenter_mode == "immediate_last_targets":
                    selected, w = list(last_targets.index), last_targets
                else:  # next_rebalance — stay risk-free, will reallocate at next scheduled date
                    selected, w = [], pd.Series(dtype=float)
                if not selected:
                    # No allocation possible → stay risk-free.
                    pass
                else:
                    portfolio_value_now = cash + sum(
                        u * prices_today.get(s, 0.0) for s, u in ledger.units.items() if u > 0
                    )
                    cash, trade_rows, tc, realized = self._trade_to_target(
                        w, prices_today, ledger, cash, portfolio_value_now,
                        gate_drift=False,
                    )
                    for r in trade_rows:
                        r["date"] = t
                        r["reason"] = f"risk_off_reenter_{cfg.reenter_mode}"
                    all_trades.extend(trade_rows)
                    total_tc += tc
                    tax.record_realized(t.year, realized)
                    last_targets = w
                pending_action = None

            # 2) Risk monitor check.
            if monitor is not None:
                transition = monitor.check(t, self._p_bear_at(t))
                if transition is Transition.TRIGGER:
                    pending_action = "liquidate"
                elif transition is Transition.CLEAR:
                    pending_action = "reenter"

            # 3) Scheduled rebalance (skip if risk-off or pending action).
            if (
                t in rebal_dates
                and (monitor is None or not monitor.is_risk_off)
                and pending_action is None
            ):
                selected, w = self._selected_at(t, returns)
                portfolio_value_now = cash + sum(
                    u * prices_today.get(s, 0.0) for s, u in ledger.units.items() if u > 0
                )
                if selected:
                    cash, trade_rows, tc, realized = self._trade_to_target(
                        w, prices_today, ledger, cash, portfolio_value_now,
                        gate_drift=True,
                    )
                    for r in trade_rows:
                        r["date"] = t
                        r["reason"] = "scheduled_rebalance"
                    all_trades.extend(trade_rows)
                    total_tc += tc
                    tax.record_realized(t.year, realized)
                    last_targets = w
                else:
                    # Empty selection on a scheduled rebalance → route to risk-free.
                    cash, trade_rows, tc, realized = self._trade_to_target(
                        pd.Series({rf_symbol: 1.0}), prices_today, ledger, cash,
                        portfolio_value_now, gate_drift=True,
                    )
                    for r in trade_rows:
                        r["date"] = t
                        r["reason"] = "empty_selection_to_rf"
                    all_trades.extend(trade_rows)
                    total_tc += tc
                    tax.record_realized(t.year, realized)

            # 4) Year-end tax settlement.
            if t.year != prev_year:
                tax_due = tax.settle_year(prev_year)
                if tax_due > 0:
                    cash -= tax_due
                    total_tax += tax_due
                prev_year = t.year

            # 5) Record NAV + weights at close.
            portfolio_value_close = cash + sum(
                u * prices_today.get(s, 0.0) for s, u in ledger.units.items() if u > 0
            )
            nav_records.append((t, portfolio_value_close))
            for s, u in ledger.units.items():
                if u > 0:
                    weight_records.append(
                        {
                            "date": t,
                            "symbol": s,
                            "weight": u * prices_today.get(s, 0.0) / max(portfolio_value_close, 1e-9),
                        }
                    )

        # Final-year settlement (so taxes on the last calendar year actually accrue).
        tax_due = tax.settle_year(prev_year)
        if tax_due > 0:
            cash -= tax_due
            total_tax += tax_due
            # Reflect the final tax in the last NAV point.
            if nav_records:
                last_t, last_nav = nav_records[-1]
                nav_records[-1] = (last_t, last_nav - tax_due)

        nav = pd.Series({d: v for d, v in nav_records}).sort_index()
        weights = (
            pd.DataFrame(weight_records)
            .pivot_table(index="date", columns="symbol", values="weight", fill_value=0.0)
            .sort_index()
            if weight_records else pd.DataFrame()
        )
        trades = pd.DataFrame(all_trades)
        risk_events = monitor.events_to_frame() if monitor is not None else pd.DataFrame()
        tax_history = pd.DataFrame(tax.history)

        return BacktestResult(
            nav=nav,
            weights=weights,
            trades=trades,
            risk_events=risk_events,
            tax_history=tax_history,
            total_tc=total_tc,
            total_tax=total_tax,
            final_carryforward=sum(tax.carryforward.values()),
        )


__all__ = ["AVCOLedger", "BacktestResult", "Executor", "TaxBook"]
