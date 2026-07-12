"""Regime Detection Cockpit — production-audit payload + embedded renderer.

THE PURPOSE OF THIS TAB IS TO AUDIT PRODUCTION RESULTS. Everything shown by
default is what the pipeline actually produced and what the strategy actually
traded — computed in Python, never re-derived in JS.

Sources (all warm caches the other tabs already build):
  * :func:`pipeline._walk_forward.compute_signals` → per-asset walk-forward
    frame with ``Raw_Prob`` (raw p_bear), ``State_Prob`` (EWMA-smoothed p_bear —
    the signal the strategy acts on), ``JM_Target_State`` (JM label), the
    ``Feature_Avg_Ret_{w}`` features and ``Target_Return``/``RF_Rate``.
  * :func:`pipeline.forecast.build_forecasts` + :func:`pipeline.forecast.apply_rule`
    → the PRODUCTION per-asset traded regime. ``apply_rule`` with the configured
    ``forecast_method`` + ``bull_prob_threshold`` is the pipeline's own selection
    decision; not-selected = bear. This is authoritative "what the strategy
    traded", NOT a UI re-derivation. For ``ewma_smoothed`` it equals
    ``smoothed_p_bear > (1 - bull_prob_threshold)``.
  * :func:`pipeline.data.load_prices` → adjusted-close price (drawdown truth).

Per ticker the payload carries, all aligned to one ``dates`` axis:
    name, sym, dates[], price[],
    p_bear_smooth[]  — production signal (State_Prob)
    p_bear_raw[]     — for the audit overlay only (Raw_Prob)
    ret{5,10,21}[]   — Feature_Avg_Ret_{w}
    dsd{5,10,21}[]   — recomputed semi-deviation (Sortino denominator, see note)
    regimes[]        — PRODUCTION traded regime segments {start,end,state} (1=bear)
    jm[]             — JM regime_label segments {start,end,state} (reference)
    baselines{}      — reference trend baselines (ma200, dualma, tsmom, macd), same
                       segment shape as regimes; warmup days default to bull (long)
    rf[]             — daily risk-free rate (for the shootout's net long/flat NAV)

Plus a single shared ``PRODUCTION`` settings object (identical across tickers)
injected as its own template global for the read-only badge.

DOWNSIDE DEVIATION: the stored ``Sortino_{w}`` is the *ratio*; the cockpit's
downside panel wants its *denominator* — the semi-deviation of negative excess
returns. Recomputed with the SAME EWMA half-life the pipeline uses:
``sqrt((min(exc,0)**2).ewm(halflife=w).mean())`` where ``exc = Target_Return -
RF_Rate``. Verified to reproduce ``Feature_Avg_Ret_w`` from ``exc`` to ~1e-6 and
match the stored ``Feature_DD_log`` to ~1e-3; uniform across all assets
(bonds/gold have no ``DD_log`` but always carry ``Target_Return``/``RF_Rate``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline._walk_forward import compute_signals
from pipeline.config import PipelineConfig
from pipeline.data import load_prices
from pipeline.forecast import apply_rule, build_forecasts

logger = logging.getLogger(__name__)

WINDOWS: tuple[int, ...] = (5, 10, 21)
_MIN_DAYS = 120  # skip tickers with too little aligned history to plot meaningfully
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "regime_cockpit.html"

# forecast_method values whose per-asset decision is a pure threshold on the
# smoothed p_bear, so a single θ line can be drawn on the signal panel.
_THETA_METHODS = {"ewma_smoothed", "prob_threshold", "regime_and_prob"}


def _semi_deviation(exc: pd.Series, w: int) -> pd.Series:
    """Sortino denominator: EWMA semi-deviation of negative excess returns."""
    downside = np.minimum(exc, 0.0)
    return np.sqrt((downside**2).ewm(halflife=w).mean())


def _round(series: pd.Series, ndigits: int) -> list[float]:
    return [round(float(x), ndigits) for x in series.to_numpy()]


def _segments(label: np.ndarray) -> list[dict]:
    """Contiguous 0/1 runs → ``[{start, end, state}]`` (``end`` exclusive)."""
    n = len(label)
    if n == 0:
        return []
    segs: list[dict] = []
    a = 0
    for i in range(1, n):
        if label[i] != label[i - 1]:
            segs.append({"start": a, "end": i, "state": int(label[a])})
            a = i
    segs.append({"start": a, "end": n, "state": int(label[a])})
    return segs


def _baseline_labels(px: pd.Series) -> dict[str, pd.Series]:
    """Classic trend/regime baselines on a full (unaligned) price series, 1=bear.

    Warmup convention: rolling stats are NaN until they have enough history;
    the bear comparisons below evaluate to False on NaN, so warmup days default
    to 0 (bull/held) — "long until the signal has enough history".

    Note: this "ma200" reference baseline is a fixed canonical 200-day SMA and
    does NOT track ``config.ma_window`` — unlike the production ``ma200`` rule
    (``pipeline.forecast.rule_ma200``), which is configurable.
    """
    ma200 = px.rolling(200).mean()
    ma50 = px.rolling(50).mean()

    horizons = [(8, 24), (16, 48), (32, 96)]
    vol63 = px.rolling(63).std()
    zs = []
    for s, l in horizons:
        q = (px.ewm(halflife=s).mean() - px.ewm(halflife=l).mean()) / vol63
        zs.append(q / q.rolling(252).std())
    combined = sum(zs) / len(zs)

    return {
        "ma200": (px < ma200).astype(int),
        "dualma": (ma50 < ma200).astype(int),
        "tsmom": (px.pct_change(252) < 0).astype(int),
        "macd": (combined < 0).astype(int),
    }


def _production_settings(config: PipelineConfig, source: str) -> dict:
    """Read-only production regime settings for the on-chart PRODUCTION badge."""
    theta = (
        round(1.0 - config.bull_prob_threshold, 4)
        if config.forecast_method in _THETA_METHODS
        else None
    )
    return {
        "signal": "smoothed",  # smoothed p_bear = State_Prob
        "rule": config.forecast_method,
        "theta": theta,  # θ on smoothed p_bear; bear when above (None if rule isn't a p_bear threshold)
        "theta_clear": None,  # per-asset selection has no separate clear θ
        "dwell": None,  # per-asset selection has no dwell (dwell is portfolio-level)
        "bull_prob_threshold": config.bull_prob_threshold,
        "ma_window": config.ma_window if config.forecast_method in ("ma200", "hybrid") else None,
        "hybrid_bear_threshold": (
            config.hybrid_bear_threshold if config.forecast_method == "hybrid" else None
        ),
        "smoothing": "EWMA (paper per-asset half-life)",
        # portfolio-level risk monitor — context only, does not gate per-asset bands
        "portfolio": {
            "bear_prob_threshold": config.bear_prob_threshold,
            "universe_trigger": config.universe_pct_threshold,
            "universe_clear": config.universe_pct_clear_threshold,
            "dwell": config.risk_off_dwell_days,
        },
        "source": source,
        "config": f"{config.pool}@{config.end_date or 'today'} fv={config.feature_version}",
        "tc_bps": config.transaction_cost_bps,
        "pfu_rate": config.pfu_rate,
    }


def build_cockpit_payload(
    config: PipelineConfig,
    *,
    force_refresh: bool = False,
) -> dict:
    """Assemble the per-ticker production-audit payload.

    Returns ``{"assets": {ticker: {...}}, "production": {...}}``. Tickers with
    missing columns / no price / too-short history are skipped (never raised);
    only an entirely empty result raises.
    """
    signals = compute_signals(config, force_refresh=force_refresh)
    prices = load_prices(config, force_refresh=force_refresh)
    fc = build_forecasts(config, force_refresh=force_refresh)
    fpanel = fc.panel
    source = fc.path.name if fc.path is not None else "forecast panel (unsaved)"

    # PRODUCTION per-asset traded regime — the pipeline's own selection decision.
    # not-selected = bear. Authoritative; computed in Python.
    flags = apply_rule(
        fpanel,
        config.forecast_method,
        theta=config.bull_prob_threshold,
        trend_window=config.trend_window,
        # ffill mirrors the forward-filled panel the Executor actually trades
        # on (run_pipeline.py's prices_with_rf), so the price-aware rules
        # agree with production instead of the raw (gap-containing) panel.
        prices=prices.ffill(),
        ma_window=config.ma_window,
        hybrid_bear_threshold=config.hybrid_bear_threshold,
    )
    selected_by_symbol = {
        sym: g.set_index("date")["selected"] for sym, g in flags.groupby("symbol")
    }

    required = {"Raw_Prob", "State_Prob", "JM_Target_State", "Target_Return", "RF_Rate"} | {
        f"Feature_Avg_Ret_{w}" for w in WINDOWS
    }

    assets: dict[str, dict] = {}
    for symbol in prices.columns:  # canonical pool order
        df = signals.get(symbol)
        if df is None:
            continue
        missing = required - set(df.columns)
        if missing:
            logger.warning("Cockpit: %s missing %s — skipping", symbol, sorted(missing))
            continue
        sel = selected_by_symbol.get(symbol)
        if sel is None:
            logger.warning("Cockpit: %s has no production selection — skipping", symbol)
            continue

        px = prices[symbol]
        idx = df.index.intersection(px.index).intersection(sel.index)
        if len(idx) < _MIN_DAYS:
            logger.warning("Cockpit: %s only %d aligned days — skipping", symbol, len(idx))
            continue

        d = df.loc[idx]
        # semi-deviation on the full frame (proper EWMA warmup) then aligned
        exc_full = df["Target_Return"] - df["RF_Rate"]
        dsd_full = {w: _semi_deviation(exc_full, w) for w in WINDOWS}

        frame = pd.DataFrame(
            {
                "price": px.loc[idx].astype(float),
                "p_bear_smooth": d["State_Prob"].astype(float),
                "p_bear_raw": d["Raw_Prob"].astype(float),
                "prod_bear": (~sel.loc[idx].astype(bool)).astype(int),  # not-selected = bear
                "jm_bear": (d["JM_Target_State"].astype(int) == 1).astype(int),
                "rf": d["RF_Rate"].astype(float),
            },
            index=idx,
        )
        for w in WINDOWS:
            frame[f"ret{w}"] = d[f"Feature_Avg_Ret_{w}"].astype(float)
            frame[f"dsd{w}"] = dsd_full[w].loc[idx].astype(float)

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < _MIN_DAYS:
            logger.warning("Cockpit: %s too short after dropna (%d) — skipping", symbol, len(frame))
            continue

        base = _baseline_labels(px.dropna())
        baselines = {
            k: _segments(base[k].reindex(frame.index).fillna(0).astype(int).to_numpy())
            for k in ("ma200", "dualma", "tsmom", "macd")
        }

        assets[symbol] = {
            "name": df.attrs.get("asset_name", symbol),
            "sym": symbol,
            "dates": [ts.strftime("%Y-%m-%d") for ts in frame.index],
            "price": _round(frame["price"], 4),
            "p_bear_smooth": _round(frame["p_bear_smooth"], 5),
            "p_bear_raw": _round(frame["p_bear_raw"], 5),
            "ret": {str(w): _round(frame[f"ret{w}"], 6) for w in WINDOWS},
            "dsd": {str(w): _round(frame[f"dsd{w}"], 6) for w in WINDOWS},
            "regimes": _segments(frame["prod_bear"].to_numpy()),  # PRODUCTION traded
            "jm": _segments(frame["jm_bear"].to_numpy()),  # JM reference
            "baselines": baselines,
            "rf": _round(frame["rf"], 8),
        }

    if not assets:
        raise RuntimeError("Cockpit payload empty — no ticker produced aligned data.")
    return {"assets": assets, "production": _production_settings(config, source)}


def render_cockpit(payload: dict, *, height: int = 1120) -> None:
    """Substitute ``payload`` into the cockpit template and embed it.

    Injection points in the template: ``/*__ASSETS__*/`` (per-ticker data) and
    ``/*__PRODUCTION__*/`` (shared read-only production settings).
    """
    assets = payload.get("assets") if payload else None
    if not assets:
        st.info("No assets available for the Regime Cockpit in this pool/window.")
        return
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace(
        "/*__ASSETS__*/", json.dumps(assets, separators=(",", ":"))
    ).replace(
        "/*__PRODUCTION__*/", json.dumps(payload["production"], separators=(",", ":"))
    )
    components.html(html, height=height, scrolling=True)


__all__ = ["build_cockpit_payload", "render_cockpit"]
