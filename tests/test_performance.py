"""Performance metrics + sub-period decomposition + report renderer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.executor import BacktestResult
from pipeline.performance import (
    PerformanceReport,
    build_buy_and_hold_nav,
    compute_metrics,
    compute_turnover_annual,
    sub_period_metrics,
)


def _linear_nav(start: float, end: float, n: int) -> pd.Series:
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(np.linspace(start, end, n), index=idx)


def test_metrics_on_monotonic_up_nav_have_positive_sharpe():
    nav = _linear_nav(100, 200, 252)  # ~100% over 1 year
    m = compute_metrics(nav)
    assert m["sharpe"] > 0
    assert m["cum_return"] == pytest.approx(1.0, abs=1e-6)
    assert m["mdd"] == pytest.approx(0.0, abs=1e-9)
    assert m["hit_rate"] >= 0.99  # every month is positive


def test_metrics_capture_drawdown():
    idx = pd.date_range("2020-01-02", periods=100, freq="B")
    nav = pd.Series(np.concatenate([np.linspace(100, 150, 50), np.linspace(150, 90, 50)]), index=idx)
    m = compute_metrics(nav)
    assert m["mdd"] == pytest.approx((90 - 150) / 150, rel=1e-4)
    assert m["sharpe"] < 0  # net loss → negative Sharpe


def test_empty_nav_returns_zero_metrics():
    m = compute_metrics(pd.Series(dtype=float))
    assert m["sharpe"] == 0.0
    assert m["mdd"] == 0.0
    assert m["n_days"] == 0


def test_sub_period_metrics_clip_to_window():
    nav = _linear_nav(100, 110, 500)  # Jan 2020 → Dec 2021
    subs = sub_period_metrics(nav)
    assert "COVID" in subs
    assert "Post_COVID" in subs
    assert "GFC" not in subs  # NAV doesn't span 2007-2009


def test_turnover_zero_when_no_trades():
    nav = _linear_nav(100, 110, 100)
    assert compute_turnover_annual(pd.DataFrame(), nav) == 0.0


def test_build_buy_and_hold_nav_rebases_to_initial_capital():
    idx = pd.date_range("2020-01-02", periods=5, freq="B")
    prices = pd.DataFrame({"^GSPC": [100, 110, 105, 120, 130]}, index=idx)
    nav = build_buy_and_hold_nav(prices, "^GSPC", initial_capital=10_000)
    assert nav.iloc[0] == pytest.approx(10_000)
    assert nav.iloc[-1] == pytest.approx(10_000 * 1.30)


def test_report_renders_markdown_without_errors():
    cfg = PipelineConfig()
    nav = _linear_nav(100_000, 110_000, 252)
    BacktestResult(
        nav=nav,
        weights=pd.DataFrame({"IVV": np.full(252, 0.5), "AGG": np.full(252, 0.5)}, index=nav.index),
        trades=pd.DataFrame(),
        risk_events=pd.DataFrame(),
        tax_history=pd.DataFrame(),
        total_tc=10.0,
        total_tax=5.0,
        final_carryforward=0.0,
    )
    report = PerformanceReport(
        config=cfg,
        strategy_metrics=compute_metrics(nav),
        strategy_sub_periods=sub_period_metrics(nav),
        benchmark_metrics={"S&P_BH": compute_metrics(nav * 0.9)},
        benchmark_sub_periods={},
    )
    md = report.render_markdown()
    assert "# Pipeline Report" in md
    assert "Strategy" in md
    assert "S&P_BH" in md
