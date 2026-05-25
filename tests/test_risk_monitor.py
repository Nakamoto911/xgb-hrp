"""Risk monitor — trigger, hysteresis, dwell."""
from __future__ import annotations

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.risk_monitor import RiskMonitor, State, Transition


def _cfg(**overrides):
    base = dict(
        bear_prob_threshold=0.70,
        universe_pct_threshold=0.40,
        universe_pct_clear_threshold=0.20,
        risk_off_dwell_days=3,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_triggers_when_bear_share_exceeds_threshold():
    m = RiskMonitor(config=_cfg())
    # 3 of 4 above bear threshold (0.75 share > 0.40)
    p = {"A": 0.9, "B": 0.8, "C": 0.85, "D": 0.2}
    t = m.check(pd.Timestamp("2024-01-02"), p)
    assert t is Transition.TRIGGER
    assert m.state is State.RISK_OFF


def test_no_trigger_when_share_below_threshold():
    m = RiskMonitor(config=_cfg())
    p = {"A": 0.9, "B": 0.2, "C": 0.2, "D": 0.2}  # 25% bear < 40% threshold
    t = m.check(pd.Timestamp("2024-01-02"), p)
    assert t is Transition.NONE
    assert m.state is State.RISK_ON


def test_clearance_requires_dwell():
    m = RiskMonitor(config=_cfg(risk_off_dwell_days=3))
    # Trigger first.
    m.check(pd.Timestamp("2024-01-02"), {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.1})
    assert m.state is State.RISK_OFF
    # 2 consecutive clear days → still risk-off.
    for d in ["2024-01-03", "2024-01-04"]:
        t = m.check(pd.Timestamp(d), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1})
        assert t is Transition.NONE
        assert m.state is State.RISK_OFF
    # 3rd consecutive clear day → clearance fires.
    t = m.check(pd.Timestamp("2024-01-05"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1})
    assert t is Transition.CLEAR
    assert m.state is State.RISK_ON


def test_dwell_resets_on_bounce_back():
    m = RiskMonitor(config=_cfg(risk_off_dwell_days=3))
    m.check(pd.Timestamp("2024-01-02"), {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.1})
    # 2 clear days then a bounce back to bearish.
    m.check(pd.Timestamp("2024-01-03"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1})
    m.check(pd.Timestamp("2024-01-04"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1})
    m.check(pd.Timestamp("2024-01-05"), {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.1})
    # 3 more clear days now — only on the 3rd does clearance fire.
    transitions = [
        m.check(pd.Timestamp("2024-01-08"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}),
        m.check(pd.Timestamp("2024-01-09"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}),
        m.check(pd.Timestamp("2024-01-10"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}),
    ]
    assert transitions[0] is Transition.NONE
    assert transitions[1] is Transition.NONE
    assert transitions[2] is Transition.CLEAR


def test_event_log_records_transitions():
    m = RiskMonitor(config=_cfg(risk_off_dwell_days=1))
    m.check(pd.Timestamp("2024-01-02"), {"A": 0.9, "B": 0.9, "C": 0.9, "D": 0.1})
    m.check(pd.Timestamp("2024-01-03"), {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1})
    df = m.events_to_frame()
    assert len(df) == 2
    assert df.iloc[0]["transition"] == "trigger"
    assert df.iloc[1]["transition"] == "clear"
