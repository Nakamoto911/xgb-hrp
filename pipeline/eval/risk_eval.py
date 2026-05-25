"""Risk-monitor eval checks (spec §12.7)."""
from __future__ import annotations

import pandas as pd

from pipeline.eval._base import EvalCheck, EvalContext, EvalResult, failed, passed


def _check_hysteresis(ctx: EvalContext) -> EvalResult:
    """No oscillation: trigger→clear→trigger within 20 trading days is suspect."""
    if ctx.backtest_result is None or ctx.backtest_result.risk_events.empty:
        return passed("hysteresis", message="no risk events to evaluate")
    events = ctx.backtest_result.risk_events.sort_values("date").reset_index(drop=True)
    bad: list[str] = []
    for i in range(2, len(events)):
        a, b, c = events.iloc[i - 2], events.iloc[i - 1], events.iloc[i]
        if (
            a["transition"] == "trigger"
            and b["transition"] == "clear"
            and c["transition"] == "trigger"
        ):
            span = (pd.Timestamp(c["date"]) - pd.Timestamp(a["date"])).days
            if span < 20:
                bad.append(f"{a['date'].date()}→{b['date'].date()}→{c['date'].date()}")
    if bad:
        return failed("hysteresis", metric=len(bad),
                      message=f"oscillations <20d detected: {bad}")
    return passed("hysteresis", metric=len(events),
                  message=f"{len(events)} events, no <20d oscillations")


CHECKS: list[EvalCheck] = [
    EvalCheck("hysteresis", "risk_monitor", _check_hysteresis, critical=False),
]
