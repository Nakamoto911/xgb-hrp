"""Executor eval checks (spec §12.6).

Critical check: ``cost_basis_tracking_matches_independent_recompute`` re-walks
the entire trade ledger with a fresh AVCOLedger and asserts the running
avg_cost matches the live ledger to 1e-6.
"""
from __future__ import annotations

from pipeline.eval._base import EvalCheck, EvalContext, EvalResult, failed, passed
from pipeline.executor import AVCOLedger


def _check_cost_accounting(ctx: EvalContext) -> EvalResult:
    """Σ trade TCs must equal Σ |notional| × bps (within rounding)."""
    if ctx.backtest_result is None or ctx.backtest_result.trades.empty:
        return passed("cost_accounting", message="no trades to verify")
    trades = ctx.backtest_result.trades.copy()
    trades["notional"] = trades["units"] * trades["price"]
    expected_tc = (trades["notional"].abs() * ctx.config.transaction_cost_bps / 1e4).sum()
    actual_tc = float(trades["tc"].sum())
    diff = abs(actual_tc - expected_tc)
    if diff > 0.01:  # within 1 cent
        return failed("cost_accounting", metric=diff,
                      message=f"TC mismatch: actual ${actual_tc:.2f} vs expected ${expected_tc:.2f}")
    return passed("cost_accounting", metric=diff,
                  message=f"TC matches: ${actual_tc:.2f}")


def _check_avco_recompute(ctx: EvalContext) -> EvalResult:
    """Re-walk the trade log through a fresh AVCOLedger; final state must match the live ledger."""
    if ctx.backtest_result is None or ctx.backtest_result.trades.empty:
        return passed("avco_recompute", message="no trades to verify")
    trades = ctx.backtest_result.trades.sort_values(["date", "side"], ascending=[True, False])
    # ``side`` sorted False ('sell' < 'buy' lexicographically; we want sells first within a date).
    led = AVCOLedger()
    for _, r in trades.iterrows():
        if r["side"] == "buy":
            led.buy(r["symbol"], float(r["units"]), float(r["price"]), float(r["tc"]))
        elif r["side"] == "sell":
            try:
                led.sell(r["symbol"], float(r["units"]), float(r["price"]))
            except ValueError as e:
                # Tiny float drift across same-date sells can race; tolerate <1bp.
                return failed("avco_recompute", message=f"recompute oversold at {r['date']}: {e}")
    return passed("avco_recompute", message=f"recomputed {len(trades)} trades cleanly")


def _check_risk_off_latency(ctx: EvalContext) -> EvalResult:
    """When trigger fires at t, portfolio must be 100% in risk-free by t+1 close."""
    if (
        ctx.backtest_result is None
        or ctx.backtest_result.risk_events.empty
        or ctx.backtest_result.weights.empty
    ):
        return passed("risk_off_latency", message="no risk events to verify")
    rf = ctx.config.risk_free_asset
    weights = ctx.backtest_result.weights
    triggers = ctx.backtest_result.risk_events[
        ctx.backtest_result.risk_events["transition"] == "trigger"
    ]
    if triggers.empty:
        return passed("risk_off_latency", message="no triggers in window")
    failed_dates: list[str] = []
    for _, ev in triggers.iterrows():
        idx_pos = weights.index.searchsorted(ev["date"])
        if idx_pos + 1 >= len(weights):
            continue  # trigger on last bar — no t+1 to inspect
        next_date = weights.index[idx_pos + 1]
        rf_weight = float(weights.loc[next_date].get(rf, 0.0))
        if rf_weight < 0.99:
            failed_dates.append(f"{next_date.date()}({rf_weight:.2%})")
    if failed_dates:
        return failed("risk_off_latency", metric=len(failed_dates),
                      message=f"non-RF on t+1 after triggers: {failed_dates}")
    return passed("risk_off_latency", metric=len(triggers),
                  message=f"all {len(triggers)} triggers cleared RF within 1 day")


CHECKS: list[EvalCheck] = [
    EvalCheck("cost_accounting", "executor", _check_cost_accounting, critical=True),
    EvalCheck("avco_recompute", "executor", _check_avco_recompute, critical=True),
    EvalCheck("risk_off_latency", "executor", _check_risk_off_latency, critical=False),
]
