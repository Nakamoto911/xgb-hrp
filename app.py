"""Streamlit UI for the end-to-end pipeline.

Sidebar binds 1:1 to ``PipelineConfig`` (spec §17 acceptance criterion #5).
The page is laid out for *trading use*: the Asset Regime chart is the first
thing rendered (it only needs the forecast panel, the single most expensive
artifact), so it appears as soon as the cached forecasts are ready. The
backtest (equity, composition, metrics) runs after it; regime-quality
diagnostics (distinctiveness / stability / reactivity of the labels) live in
a second tab and the heavier performance report + eval gates in a third, so
they never delay the regime chart's first paint.

Freshness: the end date defaults to *today* and the window auto-rolls forward,
so connecting at any hour shows the latest available session (data is fetched
through ``end + 1 day`` since yfinance's ``end`` is exclusive). On-disk caches
are keyed by the window's end date, so the first load of each new day refreshes
to the latest data; use "Refresh latest data" to force an intraday refresh.

Launch with:    streamlit run app.py
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.charts import asset_regime_chart, portfolio_composition_chart
from pipeline.cockpit import build_cockpit_payload, render_cockpit
from pipeline.config import PipelineConfig
from pipeline.data import load_prices, load_risk_free
from pipeline.eval import EvalContext, any_critical_failed, render_markdown, run_all
from pipeline.executor import Executor
from pipeline.forecast import build_forecasts
from pipeline.performance import build_benchmark_navs, build_report
from pipeline.regime import build_regimes
from pipeline.regime_quality import render_regime_quality_tab

st.set_page_config(
    page_title="xgb-hrp pipeline", page_icon="📊", layout="wide",
)
st.title("📊 xgb-hrp — end-to-end quant pipeline")
st.caption("Per-asset JM regime → XGB forecast → HRP/EW allocation → drift-band executor with daily risk monitor.")

_TODAY = date.today().isoformat()


# -----------------------------------------------------------------------------
# Sidebar — bind every PipelineConfig field
# -----------------------------------------------------------------------------
def _sidebar_config() -> tuple[PipelineConfig, bool]:
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
                [
                    "ewma_smoothed", "prob_threshold", "regime_and_prob", "trend",
                    "last_day_regime", "ma200", "hybrid",
                ],
                index=0,
            )
            theta = st.slider("Bull-prob threshold θ", 0.30, 0.95, 0.40, step=0.01)
            trend_window = st.slider("Trend window (days)", 2, 30, 5)
            ma_window = st.slider("MA window (days, ma200 / hybrid)", 20, 300, 200)
            hybrid_bear_threshold = st.slider(
                "Hybrid bear threshold (smoothed p_bear, hybrid only)",
                0.05, 0.99, 0.80, step=0.01,
            )
        with st.expander("Allocator (Module 6)", expanded=False):
            # EW is the default: no covariance/linkage step, so the backtest is
            # fast and the dashboard returns quickly. Switch to HRP for the
            # full hierarchical-risk-parity allocation.
            allocator = st.selectbox(
                "Allocator", ["ew", "hrp", "momentum_30d", "min_vol_30d"], index=0
            )
            hrp_lookback = st.slider("HRP lookback (years)", 1, 10, 4)
            hrp_linkage = st.selectbox("HRP linkage", ["single", "complete", "ward"], index=0)
        with st.expander("Execution (Module 7)", expanded=False):
            freq = st.selectbox(
                "Rebalance frequency",
                ["daily", "weekly", "monthly", "quarterly", "semi-annually", "yearly"],
                index=3,
            )
            drift = st.slider("Drift threshold", 0.0, 0.10, 0.015, step=0.001)
            exec_policy = st.selectbox(
                "Execution policy",
                ["drift_band", "flip_only"],
                index=0,
                help="drift_band: the band triggers a full re-band to target. "
                     "flip_only (tax-aware): sell only on selection flips or to "
                     "trim back to target+band — the band caps overweight drift "
                     "instead of triggering gain realization.",
            )
            tc_bps = st.number_input("Transaction cost (bps)", 0.0, 50.0, 5.0)
            pfu_rate = st.slider("PFU rate", 0.0, 0.50, 0.314, step=0.001)
        with st.expander("Risk monitor (Module 8)", expanded=True):
            rm_enabled = st.checkbox("Enable risk monitor", value=True)
            rm_signal = st.selectbox(
                "Risk monitor signal",
                ["raw", "smoothed"],
                index=0,
                help="raw = unsmoothed daily P(bear) (whipsaw-prone); "
                     "smoothed = EWMA-smoothed P(bear), same signal the selection rules use.",
            )
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
            start = st.text_input("Start date (YYYY-MM-DD)", value="2007-01-01")
            # End date defaults to today and is meant to stay there: the window
            # auto-rolls forward so the dashboard always shows the latest
            # session. Blank is also treated as today.
            end = st.text_input("End date (YYYY-MM-DD; blank = today)", value=_TODAY)

        st.divider()
        cache_dir = Path(st.text_input("Cache dir", value="cache"))
        force = st.checkbox(
            "🔄 Refresh latest data",
            value=False,
            help="Force a re-fetch of market data + re-run, ignoring caches. "
                 "Use to pull intraday updates; otherwise caches refresh once per day.",
        )

    overrides = {
        "pool": pool,
        "jm_lookback_years": jm_lookback,
        "forecast_method": rule,
        "bull_prob_threshold": theta,
        "trend_window": trend_window,
        "ma_window": ma_window,
        "hybrid_bear_threshold": hybrid_bear_threshold,
        "allocator": allocator,
        "hrp_lookback_years": hrp_lookback,
        "hrp_linkage": hrp_linkage,
        "rebalance_frequency": freq,
        "drift_threshold": drift,
        "execution_policy": exec_policy,
        "transaction_cost_bps": tc_bps,
        "pfu_rate": pfu_rate,
        "risk_monitor_enabled": rm_enabled,
        "risk_monitor_signal": rm_signal,
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
# Cached stages (keyed on the config tuple + refresh flag).
#
# The forecast panel — the expensive per-asset walk-forward — is computed and
# cached on its own so the Asset Regime chart can render the moment it is ready,
# before the (comparatively cheap, especially under EW) backtest runs. ``ttl``
# bounds in-memory cache lifetime so a long-running server picks up the daily
# cache rollover and any externally pre-built (cron-warmed) parquet.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _load_forecasts(config_json: str, force_refresh: bool):
    cfg = PipelineConfig.model_validate_json(config_json)
    panel = build_forecasts(cfg, force_refresh=force_refresh).panel
    symbol_to_asset_name = panel.groupby("symbol")["asset_name"].first().to_dict()
    return cfg, panel, symbol_to_asset_name


# The two quality-tab loaders take ``force_refresh`` only as a cache key: by
# the time they run, ``_load_forecasts``/``_run_backtest`` (which execute
# earlier in the script) have already refreshed the underlying walk-forward
# and price parquets, so forcing again here would repeat the whole fit.
# ``force_project`` re-projects the regime panel from those fresh caches.
@st.cache_data(show_spinner=False, ttl=3600)
def _load_regimes(config_json: str, force_refresh: bool):
    cfg = PipelineConfig.model_validate_json(config_json)
    return build_regimes(cfg, force_refresh=False, force_project=force_refresh).panel


@st.cache_data(show_spinner=False, ttl=3600)
def _load_prices(config_json: str, force_refresh: bool):
    cfg = PipelineConfig.model_validate_json(config_json)
    return load_prices(cfg, force_refresh=False)


@st.cache_data(show_spinner=False, ttl=3600)
def _run_backtest(config_json: str, force_refresh: bool):
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
    return result, report, bench_navs, evals


# Cockpit payload reuses the already-warm walk-forward + price caches (built by
# the loaders above), so ``force_refresh`` here is a cache key only.
@st.cache_data(show_spinner=False, ttl=3600)
def _load_cockpit_payload(config_json: str, force_refresh: bool):
    cfg = PipelineConfig.model_validate_json(config_json)
    return build_cockpit_payload(cfg, force_refresh=False)


# -----------------------------------------------------------------------------
# Main panel
# -----------------------------------------------------------------------------
cfg, force = _sidebar_config()
config_json = cfg.model_dump_json()

# Step 1 — forecasts only. This is the heavy artifact the regime chart needs;
# everything else waits behind it.
with st.spinner("Computing per-asset regime forecasts… (first cold run on a "
                "fresh window: up to ~3 min; warm-cache hits: <5s)"):
    cfg_out, forecast_panel, symbol_to_asset_name = _load_forecasts(config_json, force)

data_through = pd.to_datetime(forecast_panel["date"]).max()
st.caption(
    f"Window {cfg_out.start_date} → {cfg_out.end_date or _TODAY} · "
    f"data through **{data_through:%Y-%m-%d}** · "
    f"{forecast_panel['symbol'].nunique()} assets · allocator **{cfg_out.allocator.upper()}**"
)

tab_regime, tab_quality, tab_perf, tab_cockpit = st.tabs(
    [
        "📈 Asset regime & portfolio",
        "🔬 Regime quality",
        "📊 Performance report & eval gates",
        "🛰️ Regime Cockpit",
    ]
)

# ── Tab 1: regime chart first, then the backtest-derived views ───────────────
with tab_regime:
    # Prices for the chart's MA200 / MA200 + ADX Signal choices. Cheap here:
    # ``_load_forecasts`` above has already warmed the price parquet this
    # run pulls from, so this cached read doesn't delay the chart's first
    # paint (the default "XGB (model)" signal doesn't even touch it).
    prices_chart = _load_prices(config_json, force)
    asset_regime_chart(forecast_panel, prices=prices_chart, key_prefix="regime_top")

    st.divider()

    with st.spinner("Running backtest…"):
        t0 = time.perf_counter()
        result, report, bench_navs, evals = _run_backtest(config_json, force)
        elapsed = time.perf_counter() - t0

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

    # Portfolio composition over time
    portfolio_composition_chart(
        weights=result.weights,
        nav=result.nav,
        symbol_to_asset_name=symbol_to_asset_name,
        risk_free_symbol=cfg_out.risk_free_asset,
        strategy_label=f"{cfg_out.allocator.upper()} / {cfg_out.forecast_method}",
        key_prefix="comp_top",
    )

    # Risk events log
    if not result.risk_events.empty:
        st.subheader("Risk monitor events")
        st.dataframe(result.risk_events, width='stretch')

# ── Tab 2: regime-quality diagnostics (forecasts + regimes + prices only) ────
with tab_quality:
    with st.spinner("Loading regime panel and prices…"):
        regime_panel = _load_regimes(config_json, force)
        prices_q = _load_prices(config_json, force)
    render_regime_quality_tab(
        forecast_panel, regime_panel, prices_q, cfg_out, key_prefix="rq"
    )

# ── Tab 3: heavier report + eval gates ───────────────────────────────────────
with tab_perf:
    st.subheader("Performance report")
    st.markdown(report.render_markdown())

    st.divider()

    st.subheader("Eval gates")
    if any_critical_failed(evals):
        st.error("❌ Critical eval check failed — see report below.")
    else:
        st.success(f"✅ {sum(1 for r in evals if r.passed)}/{len(evals)} checks passed")
    st.markdown(render_markdown(evals))

# ── Tab 4: embedded Regime Detection Cockpit (client-side SVG) ────────────────
with tab_cockpit:
    st.caption(
        "Audit view — the default timeline is the pipeline's committed production regimes "
        "(traded selection rule on the smoothed signal), loaded from the parquet/config for the "
        "sidebar pool & as-of date. The what-if overlay and chart controls are client-side "
        "(no rerun) and are always compared against production, never a silent replacement."
    )
    try:
        with st.spinner("Building cockpit payload…"):
            cockpit_payload = _load_cockpit_payload(config_json, force)
        render_cockpit(cockpit_payload)
    except Exception as exc:  # never crash the tab — surface and move on
        st.error(f"Could not build the Regime Cockpit: {exc}")
