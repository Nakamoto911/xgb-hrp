"""Daily Risk Monitor — Module 8 of SPEC.md.

Stateful detector that classifies each trading day into risk-on / risk-off
based on the share of the universe whose XGB-predicted P(bear) crosses a
threshold. Two hysteresis params keep the regime from flip-flopping.

Trigger:
    count(p_bear > bear_prob_threshold) / N > universe_pct_threshold

Clearance:
    count(p_bear > bear_prob_threshold) / N < universe_pct_clear_threshold
    AND consecutive_clear_days >= risk_off_dwell_days
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class State(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


class Transition(str, Enum):
    NONE = "none"
    TRIGGER = "trigger"     # risk_on → risk_off
    CLEAR = "clear"         # risk_off → risk_on


@dataclass
class RiskEvent:
    date: pd.Timestamp
    transition: Transition
    state_after: State
    universe_bear_pct: float
    asset_bear_probs: dict[str, float]


@dataclass
class RiskMonitor:
    """Stateful daily monitor. One instance per backtest run."""

    config: PipelineConfig
    state: State = State.RISK_ON
    consecutive_clear_days: int = 0
    events: list[RiskEvent] = field(default_factory=list)

    def _bear_share(self, p_bear_today: dict[str, float]) -> float:
        if not p_bear_today:
            return 0.0
        threshold = self.config.bear_prob_threshold
        bear_count = sum(1 for p in p_bear_today.values() if p > threshold)
        return bear_count / len(p_bear_today)

    def check(
        self, date: pd.Timestamp, p_bear_today: dict[str, float]
    ) -> Transition:
        """Advance the state machine by one trading day. Returns the transition."""
        share = self._bear_share(p_bear_today)
        transition = Transition.NONE

        if self.state is State.RISK_ON:
            if share > self.config.universe_pct_threshold:
                self.state = State.RISK_OFF
                self.consecutive_clear_days = 0
                transition = Transition.TRIGGER
                logger.info(
                    "[%s] RISK-OFF TRIGGERED: %.0f%% of universe bear > %.2f (threshold %.0f%%)",
                    date.date(),
                    share * 100,
                    self.config.bear_prob_threshold,
                    self.config.universe_pct_threshold * 100,
                )

        elif self.state is State.RISK_OFF:
            if share < self.config.universe_pct_clear_threshold:
                self.consecutive_clear_days += 1
                if self.consecutive_clear_days >= self.config.risk_off_dwell_days:
                    self.state = State.RISK_ON
                    self.consecutive_clear_days = 0
                    transition = Transition.CLEAR
                    logger.info(
                        "[%s] RISK-OFF CLEARED: %.0f%% bear < %.0f%% clear threshold, "
                        "%d-day dwell satisfied (reenter_mode=%s)",
                        date.date(),
                        share * 100,
                        self.config.universe_pct_clear_threshold * 100,
                        self.config.risk_off_dwell_days,
                        self.config.reenter_mode,
                    )
            else:
                # Reset dwell counter if we bounce back above the clear threshold.
                self.consecutive_clear_days = 0

        if transition is not Transition.NONE:
            self.events.append(
                RiskEvent(
                    date=date,
                    transition=transition,
                    state_after=self.state,
                    universe_bear_pct=share,
                    asset_bear_probs=dict(p_bear_today),
                )
            )
        return transition

    @property
    def is_risk_off(self) -> bool:
        return self.state is State.RISK_OFF

    def events_to_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=["date", "transition", "state_after", "universe_bear_pct"]
            )
        return pd.DataFrame(
            {
                "date": [e.date for e in self.events],
                "transition": [e.transition.value for e in self.events],
                "state_after": [e.state_after.value for e in self.events],
                "universe_bear_pct": [e.universe_bear_pct for e in self.events],
            }
        )


def run_monitor(
    forecast_panel: pd.DataFrame, config: PipelineConfig
) -> tuple[RiskMonitor, pd.DataFrame]:
    """Run the monitor across every date in ``forecast_panel`` for ablation work.

    Returns the populated monitor plus a per-date state-history frame
    ``[date, state, universe_bear_pct]``. The executor uses ``RiskMonitor.check``
    directly day-by-day; this helper is for offline ablation runs.
    """
    monitor = RiskMonitor(config=config)
    states: list[tuple[pd.Timestamp, str, float]] = []
    # Pivot to date-indexed p_bear table for fast iteration.
    pivot = (
        forecast_panel.set_index(["date", "symbol"])["p_bear"]
        .unstack("symbol")
        .sort_index()
    )
    for date, row in pivot.iterrows():
        p_bear_today = row.dropna().to_dict()
        monitor.check(pd.Timestamp(date), p_bear_today)
        share = monitor._bear_share(p_bear_today)
        states.append((pd.Timestamp(date), monitor.state.value, share))
    history = pd.DataFrame(states, columns=["date", "state", "universe_bear_pct"])
    return monitor, history


__all__ = ["RiskMonitor", "RiskEvent", "State", "Transition", "run_monitor"]
