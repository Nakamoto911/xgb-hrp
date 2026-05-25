"""Eval registry + --gate logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.eval import (
    REGISTRY,
    EvalContext,
    any_critical_failed,
    render_markdown,
    run_all,
)
from pipeline.executor import BacktestResult


@pytest.fixture
def clean_context() -> EvalContext:
    cfg = PipelineConfig()
    idx = pd.date_range("2020-01-02", periods=252, freq="B")
    prices = pd.DataFrame(
        {
            "IVV": 100 * (1 + 0.001 * np.arange(252)),
            "AGG": 100 * (1 + 0.0003 * np.arange(252)),
            "BIL": 100 * (1 + 0.0001 * np.arange(252)),
        },
        index=idx,
    )
    nav = pd.Series(100_000 * (1 + 0.0008 * np.arange(252)), index=idx)
    weights = pd.DataFrame(
        {"IVV": np.full(252, 0.5), "AGG": np.full(252, 0.5), "BIL": np.zeros(252)},
        index=idx,
    )
    bench_navs = {
        "S&P_BH": pd.Series(100_000 * (1 + 0.0005 * np.arange(252)), index=idx),
        "60_40":  pd.Series(100_000 * (1 + 0.0006 * np.arange(252)), index=idx),
    }
    forecast = pd.DataFrame({"date": idx, "symbol": "IVV", "p_bear": 0.2, "p_bull": 0.8,
                             "p_bull_smoothed": 0.8, "regime_forecast": "Bull"})
    result = BacktestResult(
        nav=nav, weights=weights, trades=pd.DataFrame(),
        risk_events=pd.DataFrame(), tax_history=pd.DataFrame(),
        total_tc=0.0, total_tax=0.0, final_carryforward=0.0,
    )
    return EvalContext(
        config=cfg, prices=prices, forecast_panel=forecast,
        backtest_result=result, benchmark_navs=bench_navs,
    )


def test_registry_non_empty():
    assert len(REGISTRY) >= 10
    categories = {c.category for c in REGISTRY}
    assert {"data", "allocator", "executor", "risk_monitor", "e2e"} <= categories


def test_critical_checks_marked():
    crit = [c for c in REGISTRY if c.critical]
    assert {c.name for c in crit} >= {"missing_data", "weight_sum", "avco_recompute", "causality_audit"}


def test_clean_context_all_pass(clean_context):
    results = run_all(clean_context)
    assert not any_critical_failed(results), "critical checks failed on a clean context"


def test_inject_bad_weight_fails_critical():
    cfg = PipelineConfig()
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    weights = pd.DataFrame({"IVV": np.full(10, 0.5), "AGG": np.full(10, -0.1)}, index=idx)
    result = BacktestResult(
        nav=pd.Series(100_000.0, index=idx),
        weights=weights, trades=pd.DataFrame(),
        risk_events=pd.DataFrame(), tax_history=pd.DataFrame(),
        total_tc=0.0, total_tax=0.0, final_carryforward=0.0,
    )
    ctx = EvalContext(config=cfg, backtest_result=result)
    results = run_all(ctx)
    failed = [r for r in results if r.name == "non_negativity"]
    assert failed and not failed[0].passed
    assert any_critical_failed(results)


def test_render_markdown_summarizes():
    cfg = PipelineConfig()
    ctx = EvalContext(config=cfg)  # no data → many checks fail informationally
    results = run_all(ctx)
    md = render_markdown(results)
    assert "# Eval report" in md
    assert "## data" in md
    assert "Summary:" in md
