"""Streamlit UI for the end-to-end pipeline.

Sidebar binds 1:1 to ``PipelineConfig`` (spec §17 acceptance criterion #5).
The Run button executes data → forecast → backtest → report (warm-cache
target < 30s per spec §17.5) and renders the equity curve vs benchmarks,
risk events, and the eval report.

Launch with:    streamlit run app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.charts import asset_regime_chart, portfolio_composition_chart
from pipeline.config import PipelineConfig
from pipeline.data import load_prices, load_risk_free
from pipeline.eval import EvalContext, any_critical_failed, render_markdown, run_all
from pipeline.executor import Executor
from pipeline.forecast import build_forecasts
from pipeline.performance import build_benchmark_navs, build_report

st.set_page_config(
    page_title="xgb-hrp pipeline", page_icon="📊", layout="wide",
)
st.title("📊 xgb-hrp — end-to-end quant pipeline")
st.caption("Per-asset JM regime → XGB forecast → HRP allocation → drift-band executor with daily risk monitor.")


# -----------------------------------------------------------------------------
# Sidebar — bind every PipelineConfig field
# -----------------------------------------------------------------------------
def _sidebar_config() -> PipelineConfig:
    with st.sidebar:
        st.header("Config")
        with st.expander("Pool & benchmarks", expanded=True):
            pool = st.selectbox("Pool", ["etf", "mutual_fund", "european"], index=0)
            risk_free = st.text_input("Risk-free ticker (blank = default)", value="")
        with st.expander("Regime (Module 3)", expanded=False):
            jm_lookback = st.slider("JM lookback (years)", 1, 20, 11)
        with st.expander("Forecast (Module 4)", expanded=False):
            rule = st.selectbox(
                "Selection rule",
                ["ewma_smoothed", "prob_threshold", "regime_and_prob", "trend", "last_day_regime"],
                index=0,
            )
            theta = st.slider("Bull-prob threshold θ", 0.30, 0.95, 0.60, step=0.01)
            trend_window = st.slider("Trend window (days)", 2, 30, 5)
        with st.expander("Allocator (Module 6)", expanded=False):
            allocator = st.selectbox("Allocator", ["hrp", "ew", "momentum_30d", "min_vol_30d"], index=0)
            hrp_lookback = st.slider("HRP lookback (years)", 1, 10, 4)
            hrp_linkage = st.selectbox("HRP linkage", ["single", "complete", "ward"], index=0)
        with st.expander("Execution (Module 7)", expanded=False):
            freq = st.selectbox(
                "Rebalance frequency",
                ["daily", "weekly", "monthly", "quarterly", "semi-annually", "yearly"],
                index=3,
            )
            drift = st.slider("Drift threshold", 0.0, 0.10, 0.015, step=0.001)
            tc_bps = st.number_input("Transaction cost (bps)", 0.0, 50.0, 5.0)
            pfu_rate = st.slider("PFU rate", 0.0, 0.50, 0.314, step=0.001)
        with st.expander("Risk monitor (Module 8)", expanded=True):
            rm_enabled = st.checkbox("Enable risk monitor", value=True)
            bear_th = st.slider("Bear-prob threshold (per asset)", 0.50, 0.95, 0.70, step=0.01)
            uni_th = st.slider("Universe trigger threshold (%)", 10, 90, 40, step=5) / 100
            uni_clear = st.slider("Universe clear threshold (%)", 5, 50, 25, step=5) / 100
            dwell = st.slider("Clearance dwell (days)", 1, 30, 5)
            reenter = st.selectbox(
                "Re-entry mode",
                ["immediate_fresh", "immediate_last_targets", "next_rebalance"],
                index=0,
            )
        with st.expander("Backtest window", expanded=True):
            start = st.text_input("Start date (YYYY-MM-DD)", value="2023-01-01")
            end = st.text_input("End date (YYYY-MM-DD; blank = today)", value="2024-06-30")

        st.divider()
        cache_dir = Path(st.text_input("Cache dir", value="cache"))
        force = st.checkbox("Force refresh", value=False)

    overrides = {
        "pool": pool,
        "jm_lookback_years": jm_lookback,
        "forecast_method": rule,
        "bull_prob_threshold": theta,
        "trend_window": trend_window,
        "allocator": allocator,
        "hrp_lookback_years": hrp_lookback,
        "hrp_linkage": hrp_linkage,
        "rebalance_frequency": freq,
        "drift_threshold": drift,
        "transaction_cost_bps": tc_bps,
        "pfu_rate": pfu_rate,
        "risk_monitor_enabled": rm_enabled,
        "bear_prob_threshold": bear_th,
        "universe_pct_threshold": uni_th,
        "universe_pct_clear_threshold": uni_clear,
        "risk_off_dwell_days": dwell,
        "reenter_mode": reenter,
        "start_date": start,
        "end_date": end or None,
        "cache_dir": cache_dir,
    }
    if risk_free:
        overrides["risk_free_asset"] = risk_free
    return PipelineConfig.model_validate(overrides), force


# -----------------------------------------------------------------------------
# Pipeline run (cached on config tuple)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _run_pipeline(config_json: str, force_refresh: bool):
    cfg = PipelineConfig.model_validate_json(config_json)
    prices = load_prices(cfg, force_refresh=force_refresh)
    forecasts = build_forecasts(cfg, force_refresh=force_refresh).panel
    rf = load_risk_free(cfg, force_refresh=force_refresh).rename(cfg.risk_free_asset)
    prices_rf = pd.concat([prices, rf], axis=1).ffill()
    result = Executor(config=cfg, prices=prices_rf, forecast_panel=forecasts).run()
    report = build_report(cfg, result, prices_with_rf=prices_rf, force_refresh=False)
    bench_navs = build_benchmark_navs(cfg, result.nav)
    ctx = EvalContext(
        config=cfg, prices=prices, forecast_panel=forecasts,
        backtest_result=result, benchmark_navs=bench_navs,
    )
    evals = run_all(ctx)
    return cfg, result, report, bench_navs, evals


# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------
cfg, force = _sidebar_config()

# The pipeline runs immediately on every render. Heavy work is cached via
# @st.cache_data keyed on the config + force flag, so subsequent renders with
# the same config are sub-second. First cold run on a fresh window can take
# 60-180s (vendor walk-forward) — the spinner shows progress.
with st.spinner(
    "Running pipeline… (first cold run on a fresh window: up to ~3 min; "
    "warm-cache hits: <5s)"
):
    t0 = time.perf_counter()
    cfg_out, result, report, bench_navs, evals = _run_pipeline(
        cfg.model_dump_json(), force
    )
    elapsed = time.perf_counter() - t0

# Cache the forecast panel and the symbol→asset_name map so the chart
# helpers can re-pivot without re-running the pipeline.
forecast_panel = build_forecasts(cfg_out).panel
symbol_to_asset_name = (
    forecast_panel.groupby("symbol")["asset_name"].first().to_dict()
)

# ── Top of dashboard: vendor-ported charts ───────────────────────────────
asset_regime_chart(forecast_panel, key_prefix="regime_top")
portfolio_composition_chart(
    weights=result.weights,
    nav=result.nav,
    symbol_to_asset_name=symbol_to_asset_name,
    risk_free_symbol=cfg_out.risk_free_asset,
    strategy_label=f"{cfg_out.allocator.upper()} / {cfg_out.forecast_method}",
    key_prefix="comp_top",
)

st.divider()

# Headline metrics
m = report.strategy_metrics
col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(7)
col_a.metric("CAGR",       f"{m['cagr']:+.2%}")
col_b.metric("Vol (ann.)", f"{m['ann_vol']:.2%}")
col_c.metric("Sharpe",     f"{m['sharpe']:+.2f}")
col_d.metric("Max DD",     f"{m['mdd']:+.2%}")
col_e.metric("Total tax",  f"${result.total_tax:,.0f}")
col_f.metric("Risk events", len(result.risk_events))
col_g.metric("Run time",   f"{elapsed:.1f}s")

# Equity curve vs benchmarks
fig = go.Figure()
fig.add_trace(go.Scatter(x=result.nav.index, y=result.nav.values,
                         mode="lines", name="Strategy", line=dict(width=3)))
for name, nav in bench_navs.items():
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, mode="lines",
                             name=name, line=dict(dash="dot")))
fig.update_layout(
    title="Equity curve vs benchmarks",
    xaxis_title="Date", yaxis_title="NAV",
    height=420, hovermode="x unified",
    template="plotly_dark",
)
st.plotly_chart(fig, width='stretch')

# Two-column: report + eval
left, right = st.columns([3, 2])
with left:
    st.subheader("Performance report")
    st.markdown(report.render_markdown())
with right:
    st.subheader("Eval gates")
    if any_critical_failed(evals):
        st.error("❌ Critical eval check failed — see report below.")
    else:
        st.success(f"✅ {sum(1 for r in evals if r.passed)}/{len(evals)} checks passed")
    st.markdown(render_markdown(evals))

# Risk events log
if not result.risk_events.empty:
    st.subheader("Risk monitor events")
    st.dataframe(result.risk_events, width='stretch')
