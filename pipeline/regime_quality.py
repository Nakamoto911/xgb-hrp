"""Regime-quality diagnostics behind the "🔬 Regime quality" dashboard tab.

Grades the Asset Regime Chart on three questions:

* **Distinct?** — Bull and Bear days occupy separated regions of the
  (20d return, 20d vol) feature space. Measured with the closed-form
  2-Wasserstein distance between per-regime Gaussians (z-scored features,
  so the score is unit-free and comparable across assets) plus a KDE
  contour view.
* **Stable?** — regimes persist instead of flickering. Kaplan-Meier
  survival curves of spell durations, flicker rate, switches/year, and
  run-to-run label revisions (each dashboard run snapshots its labels so
  history rewrites become observable).
* **Tradable?** — the signal reacts to real drawdowns fast enough to pay.
  Drawdown events anchor crash/recovery detection lags, the price return
  given up between peak→detection and trough→re-entry, event-aligned
  P(Bear) trajectories, and the next-day return spread between
  Bull-labelled and Bear-labelled days.

Layout: the top half is pure metric functions (numpy/pandas only — no
streamlit, unit-tested, reusable by a future ``pipeline/eval`` regime
module). The bottom half is Streamlit/plotly render functions following
the :mod:`pipeline.charts` conventions: functions own their widgets, take
a ``key_prefix``, and return ``None``.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pipeline.charts import _ASSET_META, _RISK_ORDER, _RISK_RANK
from pipeline.config import PipelineConfig

_TRADING_DAYS = 252
_FLICKER_MAX_DAYS = 5
_EVENT_WINDOW = (-30, 60)
_MIN_REGIME_OBS = 40       # below this, per-regime covariance is unstable
_MIN_EVENTS_PER_OFFSET = 3  # mask trajectory offsets with fewer events

_BULL_COLOR = "rgb(34,139,34)"
_BEAR_COLOR = "rgb(200,30,30)"

# Selector label → internal source key. "smoothed" is what the executor
# trades on; "raw" is the unsmoothed XGB probability the heatmap shows;
# "jm" is the in-sample Viterbi label — an upper bound, not tradable.
_SIGNAL_SOURCES = {
    "Smoothed (traded signal)": "smoothed",
    "Raw P(Bear) > 0.5": "raw",
    "JM labels (in-sample upper bound)": "jm",
}

# Scoreboard pass cutoffs. Duration per SPEC §12.2; W2 ≥ 1 means the regime
# centroids sit ≥1σ apart in z-scored feature space.
_THRESHOLDS: dict[str, tuple[str, float]] = {
    "w2": (">", 1.0),
    "median_duration": (">", 20.0),
    "flicker_rate": ("<", 0.20),
    "spread_signal": (">", 0.0),
    "crash_lag_med": ("<", 10.0),
}


# =============================================================================
# Pure metric layer (no streamlit)
# =============================================================================
def extract_spells(labels: pd.Series) -> pd.DataFrame:
    """Run-length encode a date-indexed {'Bull','Bear'} series into spells.

    Returns ``[label, start, end, length, censored]``. A spell touching the
    first or last observation is ``censored``: its true duration is only
    bounded below by what we observed.
    """
    cols = ["label", "start", "end", "length", "censored"]
    lab = labels.dropna()
    if lab.empty:
        return pd.DataFrame(columns=cols)
    run_id = (lab != lab.shift()).cumsum()
    rows = [
        (grp.iloc[0], grp.index[0], grp.index[-1], len(grp))
        for _, grp in lab.groupby(run_id)
    ]
    spells = pd.DataFrame(rows, columns=cols[:4])
    spells["censored"] = False
    spells.iloc[0, spells.columns.get_loc("censored")] = True
    spells.iloc[-1, spells.columns.get_loc("censored")] = True
    return spells


def spell_stats(spells: pd.DataFrame, n_days: int) -> dict:
    """Summary stats for a spell table over ``n_days`` observations.

    Median duration uses complete (uncensored) spells when at least 3 exist,
    otherwise all spells — a short sample may hold a single censored spell.
    """
    if spells.empty or n_days <= 0:
        return {
            "median_duration": float("nan"),
            "flicker_rate": float("nan"),
            "switches_per_year": float("nan"),
            "n_spells": 0,
            "n_censored": 0,
        }
    complete = spells.loc[~spells["censored"], "length"]
    pool = complete if len(complete) >= 3 else spells["length"]
    return {
        "median_duration": float(pool.median()),
        "flicker_rate": float((spells["length"] <= _FLICKER_MAX_DAYS).mean()),
        "switches_per_year": float((len(spells) - 1) * _TRADING_DAYS / n_days),
        "n_spells": int(len(spells)),
        "n_censored": int(spells["censored"].sum()),
    }


def survival_curve(lengths: np.ndarray, censored: np.ndarray) -> pd.Series:
    """Kaplan-Meier product-limit estimate S(t) = P(spell survives past t).

    Censored spells stay in the at-risk set through their observed length,
    then drop out without an event. With no censoring this reduces to the
    empirical ``mean(lengths > t)``.
    """
    lengths = np.asarray(lengths, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if lengths.size == 0:
        return pd.Series(dtype=float)
    s = 1.0
    out: dict[int, float] = {}
    for t in range(1, int(lengths.max()) + 1):
        at_risk = int((lengths >= t).sum())
        events = int(((lengths == t) & ~censored).sum())
        if at_risk > 0:
            s *= 1.0 - events / at_risk
        out[t] = s
    return pd.Series(out, name="survival")


def w2_gaussian_2d(
    mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray
) -> float:
    """Closed-form 2-Wasserstein distance between two 2D Gaussians.

    For 2×2 SPD M, tr(√M) = √(tr M + 2√det M), so with
    M = √Σ₁ Σ₂ √Σ₁ (where tr M = tr Σ₁Σ₂, det M = det Σ₁ det Σ₂):

        W2² = |μ₁−μ₂|² + trΣ₁ + trΣ₂ − 2√(tr(Σ₁Σ₂) + 2√(detΣ₁ detΣ₂))

    No matrix square root needed.
    """
    mu1, mu2 = np.asarray(mu1, float), np.asarray(mu2, float)
    c1, c2 = np.asarray(cov1, float), np.asarray(cov2, float)
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    det_term = np.sqrt(max(np.linalg.det(c1), 0.0) * max(np.linalg.det(c2), 0.0))
    cross = np.sqrt(max(np.trace(c1 @ c2) + 2.0 * det_term, 0.0))
    w2_sq = mean_term + np.trace(c1) + np.trace(c2) - 2.0 * cross
    return float(np.sqrt(max(w2_sq, 0.0)))


def regime_feature_frame(
    prices: pd.Series, labels: pd.Series, window: int = 20
) -> pd.DataFrame:
    """Project an asset into (rolling return, realized vol) space per label.

    Returns ``[ret, vol, label]``: ``window``-day cumulative return and
    annualized realized vol, inner-joined with the labels, NaN rows dropped,
    then each dimension z-scored on the full sample so distances are
    unit-free and comparable across assets.
    """
    daily = prices.pct_change()
    feats = pd.DataFrame(
        {
            "ret": prices.pct_change(window),
            "vol": daily.rolling(window).std() * np.sqrt(_TRADING_DAYS),
        }
    )
    df = feats.join(labels.rename("label"), how="inner").dropna()
    if df.empty:
        return df
    for col in ("ret", "vol"):
        sd = df[col].std()
        df[col] = (df[col] - df[col].mean()) / (sd if sd > 0 else 1.0)
    return df


def regime_w2(feats: pd.DataFrame, min_obs: int = _MIN_REGIME_OBS) -> float:
    """W2 distance between the Bull and Bear Gaussians fitted on ``feats``.

    NaN when either regime has fewer than ``min_obs`` observations.
    """
    moments = {}
    for label in ("Bull", "Bear"):
        sub = feats.loc[feats["label"] == label, ["ret", "vol"]]
        if len(sub) < min_obs:
            return float("nan")
        moments[label] = (sub.mean().to_numpy(), np.cov(sub.to_numpy().T))
    return w2_gaussian_2d(*moments["Bull"], *moments["Bear"])


def next_day_spread(
    prices: pd.Series, bear_flag: pd.Series
) -> tuple[float, float, float]:
    """Annualized mean(r_{t+1} | Bull_t) − mean(r_{t+1} | Bear_t).

    ``bear_flag`` is day-t information (the executor's +1-day trade shift is
    what this conditioning mirrors). Returns ``(spread, bull_ann, bear_ann)``;
    NaN legs when a regime is never visited.
    """
    fwd = prices.pct_change().shift(-1)
    df = pd.DataFrame({"fwd": fwd}).join(bear_flag.rename("bear"), how="inner").dropna()
    bull = df.loc[~df["bear"].astype(bool), "fwd"]
    bear = df.loc[df["bear"].astype(bool), "fwd"]
    bull_ann = float(bull.mean() * _TRADING_DAYS) if len(bull) else float("nan")
    bear_ann = float(bear.mean() * _TRADING_DAYS) if len(bear) else float("nan")
    return bull_ann - bear_ann, bull_ann, bear_ann


def detect_drawdown_events(prices: pd.Series, threshold: float = 0.15) -> pd.DataFrame:
    """Find peak→trough drawdowns of at least ``threshold`` depth.

    Running-max segmentation: each underwater stretch between new highs is
    one candidate event. Returns ``[peak_date, trough_date, recovery_date,
    depth]`` where ``recovery_date`` is the first date the prior peak is
    regained (NaT if the sample ends underwater) and ``depth`` is negative.
    """
    cols = ["peak_date", "trough_date", "recovery_date", "depth"]
    px = prices.dropna()
    if px.empty or threshold <= 0:
        return pd.DataFrame(columns=cols)
    at_high = px >= px.cummax()
    groups = [grp for _, grp in px.groupby(at_high.cumsum())]
    events = []
    for i, grp in enumerate(groups):
        dd = grp / grp.iloc[0] - 1.0
        depth = float(dd.min())
        if depth <= -threshold:
            events.append(
                {
                    "peak_date": grp.index[0],
                    "trough_date": dd.idxmin(),
                    "recovery_date": (
                        groups[i + 1].index[0] if i + 1 < len(groups) else pd.NaT
                    ),
                    "depth": depth,
                }
            )
    return pd.DataFrame(events, columns=cols)


def signal_lags(
    events: pd.DataFrame, bear_flag: pd.Series, prices: pd.Series
) -> pd.DataFrame:
    """Grade the signal's reaction to each drawdown event.

    Adds per event:

    * ``crash_lag`` — trading days from peak until the signal first turns
      Bear, searched only within [peak, recovery-or-end] so a flip belonging
      to the *next* event can't count. 0 when already Bear at the peak
      (early warnings are not rewarded with negative lags). NaN = missed.
    * ``recovery_lag`` — trading days from trough until the signal turns
      back Bull, searched from max(trough, detection) so an event the signal
      never saw can't score a spurious instant re-entry. NaN = missed.
    * ``missed_exit_ret`` — price return peak → detection (the drawdown
      absorbed before the signal reacted).
    * ``missed_rebound_ret`` — price return trough → re-entry (the rebound
      foregone while still labelled Bear).
    * ``detect_date`` / ``reentry_date`` for the events table.
    """
    extra = [
        "crash_lag", "recovery_lag", "missed_exit_ret", "missed_rebound_ret",
        "detect_date", "reentry_date",
    ]
    df = pd.DataFrame({"px": prices, "bear": bear_flag}).dropna()
    if events.empty or df.empty:
        return events.assign(**{c: pd.Series(dtype=object) for c in extra})
    flags = df["bear"].astype(bool).to_numpy()
    px = df["px"].to_numpy()
    idx = df.index
    rows = []
    for ev in events.itertuples(index=False):
        peak_pos = int(idx.searchsorted(ev.peak_date))
        trough_pos = int(idx.searchsorted(ev.trough_date))
        end_pos = (
            int(idx.searchsorted(ev.recovery_date))
            if pd.notna(ev.recovery_date)
            else len(idx) - 1
        )
        end_pos = min(end_pos, len(idx) - 1)
        out = ev._asdict()
        out.update(dict.fromkeys(extra, np.nan))
        out["detect_date"], out["reentry_date"] = pd.NaT, pd.NaT
        if peak_pos < len(idx):
            hits = np.flatnonzero(flags[peak_pos : end_pos + 1])
            if hits.size:
                detect_pos = peak_pos + int(hits[0])
                out["crash_lag"] = float(hits[0])
                out["detect_date"] = idx[detect_pos]
                out["missed_exit_ret"] = float(px[detect_pos] / px[peak_pos] - 1.0)
                start = max(trough_pos, detect_pos)
                back = np.flatnonzero(~flags[start:])
                if back.size:
                    reentry_pos = start + int(back[0])
                    out["recovery_lag"] = float(reentry_pos - trough_pos)
                    out["reentry_date"] = idx[reentry_pos]
                    out["missed_rebound_ret"] = float(
                        px[reentry_pos] / px[trough_pos] - 1.0
                    )
        rows.append(out)
    return pd.DataFrame(rows)


def event_aligned_matrix(
    anchor_dates: Iterable[pd.Timestamp],
    series: pd.Series,
    window: tuple[int, int] = _EVENT_WINDOW,
) -> pd.DataFrame:
    """Align ``series`` around each anchor: rows = events, cols = offsets.

    Offsets are trading-day positions relative to the anchor (anchors snap
    to the first index position at/after the date). Cells where the window
    leaves the sample are NaN.
    """
    s = series.dropna()
    lo, hi = window
    offsets = np.arange(lo, hi + 1)
    values = s.to_numpy()
    rows, kept = [], []
    for anchor in anchor_dates:
        pos = int(s.index.searchsorted(pd.Timestamp(anchor)))
        if pos >= len(s):
            continue
        positions = pos + offsets
        valid = (positions >= 0) & (positions < len(s))
        row = np.full(offsets.shape, np.nan)
        row[valid] = values[positions[valid]]
        rows.append(row)
        kept.append(s.index[pos])
    return pd.DataFrame(rows, index=pd.Index(kept, name="anchor"), columns=offsets)


def trajectory_stats(
    mats: list[pd.DataFrame], min_events: int = _MIN_EVENTS_PER_OFFSET
) -> pd.DataFrame:
    """Pool event-aligned matrices into mean/IQR per offset.

    Offsets with fewer than ``min_events`` contributing events are masked
    so a single event can't masquerade as a band.
    """
    mats = [m for m in mats if not m.empty]
    if not mats:
        return pd.DataFrame(columns=["mean", "q25", "q75", "n"])
    pooled = pd.concat(mats, axis=0)
    n = pooled.notna().sum()
    out = pd.DataFrame(
        {
            "mean": pooled.mean(),
            "q25": pooled.quantile(0.25),
            "q75": pooled.quantile(0.75),
            "n": n,
        }
    )
    out.loc[n < min_events, ["mean", "q25", "q75"]] = np.nan
    return out


def bear_signal(
    forecast_sym: pd.DataFrame, jm_labels: pd.Series, source: str
) -> tuple[pd.Series, pd.Series]:
    """Resolve a signal source into (bear_flag, plotted probability).

    ``forecast_sym`` is one symbol's forecast-panel slice indexed by date;
    ``jm_labels`` the same symbol's regime-panel labels. The probability is
    what the event-aligned panels plot: P(Bear) for the probabilistic
    sources, the 0/1 label for ``jm``.
    """
    if source == "jm":
        flag = jm_labels == "Bear"
        return flag, flag.astype(float)
    if source == "raw":
        prob = forecast_sym["p_bear"]
        return prob > 0.5, prob
    if source == "smoothed":
        prob = 1.0 - forecast_sym["p_bull_smoothed"]
        return forecast_sym["regime_forecast"] == "Bear", prob
    raise ValueError(f"Unknown signal source: {source!r}")


@dataclass(frozen=True)
class AssetQuality:
    """All per-asset diagnostics, computed once and shared by every section."""

    symbol: str
    asset_name: str
    feats: pd.DataFrame    # z-scored [ret, vol, label]
    spells: pd.DataFrame
    events: pd.DataFrame   # drawdown events with lags + missed returns
    bear_flag: pd.Series
    prob: pd.Series        # series plotted in the event-aligned panels
    w2: float
    stats: dict
    spread_signal: float
    spread_jm: float


def _symbol_slice(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = panel.loc[panel["symbol"] == symbol].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date").sort_index()


def compute_quality(
    prices: pd.DataFrame,
    regime_panel: pd.DataFrame,
    forecast_panel: pd.DataFrame,
    *,
    dd_threshold: float,
    signal_source: str,
) -> list[AssetQuality]:
    """Per-asset quality diagnostics, risk-ordered (defensive first).

    Only symbols present in all three inputs are graded.
    """
    symbols = (
        set(forecast_panel["symbol"])
        & set(regime_panel["symbol"])
        & set(prices.columns)
    )
    name_of = forecast_panel.groupby("symbol")["asset_name"].first().to_dict()
    ordered = sorted(
        symbols, key=lambda s: _RISK_RANK.get(name_of.get(s, s), len(_RISK_ORDER))
    )
    out: list[AssetQuality] = []
    for symbol in ordered:
        forecast_sym = _symbol_slice(forecast_panel, symbol)
        jm_labels = _symbol_slice(regime_panel, symbol)["regime_label"]
        px = prices[symbol].dropna()
        bear_flag, prob = bear_signal(forecast_sym, jm_labels, signal_source)
        labels = pd.Series(
            np.where(bear_flag, "Bear", "Bull"), index=bear_flag.index
        )
        spells = extract_spells(labels)
        feats = regime_feature_frame(px, labels)
        events = signal_lags(
            detect_drawdown_events(px, dd_threshold), bear_flag, px
        )
        out.append(
            AssetQuality(
                symbol=symbol,
                asset_name=name_of.get(symbol, symbol),
                feats=feats,
                spells=spells,
                events=events,
                bear_flag=bear_flag,
                prob=prob,
                w2=regime_w2(feats),
                stats=spell_stats(spells, len(labels)),
                spread_signal=next_day_spread(px, bear_flag)[0],
                spread_jm=next_day_spread(px, jm_labels == "Bear")[0],
            )
        )
    return out


def build_scoreboard(qualities: list[AssetQuality]) -> pd.DataFrame:
    """One row per asset summarizing every pillar (NaN where undefined)."""
    rows = []
    for q in qualities:
        detected = q.events["crash_lag"].notna() if not q.events.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "asset_name": q.asset_name,
                "symbol": q.symbol,
                "w2": q.w2,
                "median_duration": q.stats["median_duration"],
                "flicker_rate": q.stats["flicker_rate"],
                "switches_per_year": q.stats["switches_per_year"],
                "spread_signal": q.spread_signal,
                "spread_jm": q.spread_jm,
                "crash_lag_med": float(q.events["crash_lag"].median()) if not q.events.empty else float("nan"),
                "recovery_lag_med": float(q.events["recovery_lag"].median()) if not q.events.empty else float("nan"),
                "n_events": int(len(q.events)),
                "n_detected": int(detected.sum()),
            }
        )
    return pd.DataFrame(rows)


def flip_comparison(forecast_panel: pd.DataFrame) -> pd.DataFrame:
    """Switches/year of the raw vs smoothed signal per asset.

    Shows what the EWMA smoothing buys in whipsaw reduction.
    """
    rows = []
    for symbol, grp in forecast_panel.groupby("symbol"):
        sym = grp.copy()
        sym["date"] = pd.to_datetime(sym["date"])
        sym = sym.set_index("date").sort_index()
        n_days = len(sym)
        raw = pd.Series(np.where(sym["p_bear"] > 0.5, "Bear", "Bull"), index=sym.index)
        smoothed = sym["regime_forecast"]
        rows.append(
            {
                "symbol": symbol,
                "asset_name": sym["asset_name"].iloc[0],
                "raw": spell_stats(extract_spells(raw), n_days)["switches_per_year"],
                "smoothed": spell_stats(extract_spells(smoothed), n_days)["switches_per_year"],
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Label vintages — run-to-run revision tracking
# -----------------------------------------------------------------------------
def append_vintage(
    regime_panel: pd.DataFrame, path: Path, run_date: str
) -> pd.DataFrame:
    """Snapshot today's labels into the vintages parquet; return all vintages.

    One snapshot per ``run_date`` — reruns on the same day are no-ops, so the
    first render of the day defines that day's vintage (an intraday force
    refresh won't overwrite it). Comparing consecutive vintages shows how
    much history each live re-fit rewrites — the out-of-sample stability
    measure a one-shot backtest can't provide.
    """
    snap = regime_panel[["symbol", "date", "regime_label"]].copy()
    snap.insert(0, "run_date", run_date)
    if path.exists():
        existing = pd.read_parquet(path)
        if (existing["run_date"] == run_date).any():
            return existing
        vintages = pd.concat([existing, snap], ignore_index=True)
    else:
        vintages = snap
    path.parent.mkdir(parents=True, exist_ok=True)
    vintages.to_parquet(path)
    return vintages


def revision_stats(vintages: pd.DataFrame) -> pd.DataFrame:
    """Label disagreement between consecutive vintages, per asset.

    Returns ``[run_date, prev_run_date, symbol, n_overlap, pct_revised]``
    over the (symbol, date) keys both vintages cover.
    """
    cols = ["run_date", "prev_run_date", "symbol", "n_overlap", "pct_revised"]
    runs = sorted(vintages["run_date"].unique())
    rows = []
    for prev, cur in zip(runs, runs[1:], strict=False):
        a = vintages.loc[vintages["run_date"] == prev].set_index(["symbol", "date"])["regime_label"]
        b = vintages.loc[vintages["run_date"] == cur].set_index(["symbol", "date"])["regime_label"]
        joined = a.to_frame("prev").join(b.to_frame("cur"), how="inner")
        if joined.empty:
            continue
        changed = (joined["prev"] != joined["cur"]).groupby(level="symbol")
        for symbol, grp in changed:
            rows.append(
                {
                    "run_date": cur,
                    "prev_run_date": prev,
                    "symbol": symbol,
                    "n_overlap": int(grp.size),
                    "pct_revised": float(grp.mean()),
                }
            )
    return pd.DataFrame(rows, columns=cols)


# =============================================================================
# Render layer (streamlit + plotly)
# =============================================================================
_GOOD_CSS = "background-color: rgba(27, 94, 32, 0.55)"
_BAD_CSS = "background-color: rgba(150, 30, 30, 0.55)"


def _threshold_css(value: float, op: str, cutoff: float) -> str:
    if pd.isna(value):
        return ""
    ok = value > cutoff if op == ">" else value < cutoff
    return _GOOD_CSS if ok else _BAD_CSS


def _render_scoreboard(scoreboard: pd.DataFrame, *, signal_label: str) -> None:
    st.subheader("Quality scoreboard")
    st.caption(
        f"All columns graded on **{signal_label}**, except *Spread (JM)* — the "
        "in-sample Viterbi labels, the ceiling a causal signal is chasing. "
        "Spreads are annualized next-day return differences (Bull-labelled "
        "minus Bear-labelled days, signal lagged one day as traded). Lags are "
        "median trading days to react to drawdown events."
    )
    display = scoreboard.rename(
        columns={
            "asset_name": "Asset",
            "symbol": "Symbol",
            "w2": "W2 separation",
            "median_duration": "Median dur (d)",
            "flicker_rate": "Flicker ≤5d",
            "switches_per_year": "Switches/yr",
            "spread_signal": "Spread (signal)",
            "spread_jm": "Spread (JM)",
            "crash_lag_med": "Crash lag (d)",
            "recovery_lag_med": "Recovery lag (d)",
            "n_events": "Events",
            "n_detected": "Detected",
        }
    )
    rename = {
        "w2": "W2 separation",
        "median_duration": "Median dur (d)",
        "flicker_rate": "Flicker ≤5d",
        "spread_signal": "Spread (signal)",
        "crash_lag_med": "Crash lag (d)",
    }
    styler = display.style.format(
        {
            "W2 separation": "{:.2f}",
            "Median dur (d)": "{:.0f}",
            "Flicker ≤5d": "{:.0%}",
            "Switches/yr": "{:.1f}",
            "Spread (signal)": "{:+.1%}",
            "Spread (JM)": "{:+.1%}",
            "Crash lag (d)": "{:.0f}",
            "Recovery lag (d)": "{:.0f}",
        },
        na_rep="—",
    )
    for key, (op, cutoff) in _THRESHOLDS.items():
        styler = styler.map(
            lambda v, op=op, cutoff=cutoff: _threshold_css(v, op, cutoff),
            subset=[rename[key]],
        )
    st.dataframe(styler, width='stretch', hide_index=True)
    st.caption(
        "Green/red vs cutoffs: W2 > 1σ · median duration > 20d (SPEC §12.2) · "
        "flicker < 20% · signal spread > 0 · crash lag < 10d. "
        "“—” = not computable in this window (e.g. no events)."
    )


def _distinctiveness_figure(feats: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for label, rgb in (("Bull", "34,139,34"), ("Bear", "200,30,30")):
        sub = feats.loc[feats["label"] == label]
        if len(sub) < 10:
            continue
        fig.add_trace(
            go.Histogram2dContour(
                x=sub["ret"],
                y=sub["vol"],
                name=label,
                ncontours=8,
                showscale=False,
                colorscale=[[0, f"rgba({rgb},0)"], [1, f"rgba({rgb},0.9)"]],
                contours=dict(coloring="lines"),
                line=dict(width=1.5),
                hoverinfo="skip",
            )
        )
        thin = sub.iloc[:: max(1, len(sub) // 600)]
        fig.add_trace(
            go.Scatter(
                x=thin["ret"],
                y=thin["vol"],
                mode="markers",
                name=f"{label} days",
                marker=dict(color=f"rgba({rgb},0.35)", size=4),
                hovertemplate="ret %{x:.2f}σ · vol %{y:.2f}σ<extra>" + label + "</extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="20d return (z-score)",
        yaxis_title="20d realized vol (z-score)",
        height=430,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _w2_bar_figure(scoreboard: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=scoreboard["asset_name"],
            y=scoreboard["w2"],
            marker_color=[
                _ASSET_META.get(n, ("#CCCCCC", n))[0] for n in scoreboard["asset_name"]
            ],
            hovertemplate="%{x}: W2 = %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="white",
        annotation_text="1σ separation", annotation_font_color="white",
    )
    fig.update_layout(
        title="W2 separation per asset",
        yaxis_title="W2 (z-scored feature space)",
        height=430,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _render_distinctiveness(
    qualities: list[AssetQuality], scoreboard: pd.DataFrame, scope: str
) -> None:
    st.subheader("1 · Distinctiveness — are Bull and Bear different markets?")
    st.caption(
        "Each day plotted in (20d return, 20d vol) space, colored by its "
        "label. Two separated islands = the labels carve out genuinely "
        "different market physics; one blended cloud = arbitrary distinctions."
    )
    if scope == "__all__":
        feats = pd.concat([q.feats for q in qualities], ignore_index=True)
        title = "All assets (pooled, per-asset z-scores)"
    else:
        q = next(q for q in qualities if q.symbol == scope)
        feats, title = q.feats, f"{q.asset_name} ({q.symbol})"
    left, right = st.columns([3, 2])
    if feats.empty:
        left.info("Not enough overlapping price/label history for the feature plot.")
    else:
        left.plotly_chart(_distinctiveness_figure(feats, title), width='stretch')
    right.plotly_chart(_w2_bar_figure(scoreboard), width='stretch')


def _survival_figure(spells: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for label, color in (("Bull", _BULL_COLOR), ("Bear", _BEAR_COLOR)):
        sub = spells.loc[spells["label"] == label]
        if sub.empty:
            continue
        surv = survival_curve(sub["length"].to_numpy(), sub["censored"].to_numpy())
        fig.add_trace(
            go.Scatter(
                x=surv.index,
                y=surv.values,
                name=f"{label} ({len(sub)} spells)",
                line=dict(color=color, width=2, shape="hv"),
                hovertemplate="day %{x}: %{y:.0%} survive<extra>" + label + "</extra>",
            )
        )
    fig.add_vline(
        x=_FLICKER_MAX_DAYS, line_dash="dot", line_color="gray",
        annotation_text="flicker zone", annotation_font_color="gray",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Consecutive trading days in regime",
        yaxis_title="P(spell survives)",
        yaxis=dict(range=[0, 1.02], tickformat=".0%"),
        height=400,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _flip_figure(flips: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=flips["asset_name"], y=flips["raw"], name="Raw P(Bear) > 0.5",
            marker_color="rgba(255,215,0,0.85)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=flips["asset_name"], y=flips["smoothed"], name="Smoothed (traded)",
            marker_color="rgba(100,181,246,0.9)",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Signal flips per year — what the EWMA smoothing buys",
        yaxis_title="Switches / year",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _render_revisions(vintages: pd.DataFrame) -> None:
    st.markdown("**Run-to-run label revisions**")
    n_runs = vintages["run_date"].nunique()
    if n_runs < 2:
        st.info(
            "Each dashboard run snapshots its regime labels (one vintage per "
            f"day; {n_runs} so far). Once a second vintage exists, this panel "
            "shows how much history each live re-fit rewrites — labels that "
            "keep changing after the fact can't be trusted in real time."
        )
        return
    stats = revision_stats(vintages)
    if stats.empty:
        st.info("No overlapping (symbol, date) coverage between vintages yet.")
        return
    latest = stats.loc[stats["run_date"] == stats["run_date"].max()]
    weighted = (latest["pct_revised"] * latest["n_overlap"]).sum() / latest["n_overlap"].sum()
    st.metric(
        f"History revised in latest run ({latest['run_date'].iloc[0]})",
        f"{weighted:.2%}",
        help="Share of previously published labels that changed vs the prior vintage.",
    )
    pivot = stats.pivot_table(
        index="symbol", columns="run_date", values="pct_revised"
    ).sort_index()
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, "rgb(34,139,34)"], [0.5, "rgb(255,215,0)"], [1, "rgb(200,30,30)"]],
            zmin=0, zmax=max(0.10, float(np.nanmax(pivot.to_numpy()))),
            colorbar=dict(title="% revised", tickformat=".1%"),
            hovertemplate="%{y} · run %{x}: %{z:.2%} revised<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(300, 28 * len(pivot) + 80),
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')


def _render_stability(
    qualities: list[AssetQuality],
    forecast_panel: pd.DataFrame,
    vintages: pd.DataFrame,
    scope: str,
) -> None:
    st.subheader("2 · Stability — do regimes persist or flicker?")
    st.caption(
        "Survival curves should decay slowly: a cliff inside the flicker zone "
        "means the signal whipsaws and the backtest bleeds turnover."
    )
    if scope == "__all__":
        spells = pd.concat([q.spells for q in qualities], ignore_index=True)
        title = "Regime survival — all assets pooled"
        n_censored = int(spells["censored"].sum()) if not spells.empty else 0
    else:
        q = next(q for q in qualities if q.symbol == scope)
        spells = q.spells
        title = f"Regime survival — {q.asset_name} ({q.symbol})"
        n_censored = q.stats["n_censored"]
    left, right = st.columns(2)
    if spells.empty:
        left.info("No spells in this window.")
    else:
        left.plotly_chart(_survival_figure(spells, title), width='stretch')
        left.caption(
            f"{n_censored} spell(s) touch a sample edge (censored) — they count "
            "as at-risk for their observed length, Kaplan-Meier style."
        )
    right.plotly_chart(_flip_figure(flip_comparison(forecast_panel)), width='stretch')
    st.divider()
    _render_revisions(vintages)


def _trajectory_panel_figure(
    peak_stats: pd.DataFrame, trough_stats: pd.DataFrame, prob_label: str
) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Crash reactivity — peaks aligned at T=0",
            "Recovery reactivity — troughs aligned at T=0",
        ),
        shared_yaxes=True,
    )
    band_color = "rgba(100,181,246,0.25)"
    for col, stats in ((1, peak_stats), (2, trough_stats)):
        if stats.empty:
            continue
        x = list(stats.index)
        fig.add_trace(
            go.Scatter(x=x, y=stats["q75"], line=dict(width=0),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(x=x, y=stats["q25"], line=dict(width=0), fill="tonexty",
                       fillcolor=band_color, name="IQR", showlegend=(col == 1),
                       hoverinfo="skip"),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=stats["mean"], name="Mean", showlegend=(col == 1),
                line=dict(color="rgb(100,181,246)", width=2.5),
                customdata=stats["n"],
                hovertemplate="T%{x:+d}: %{y:.0%} (%{customdata} events)<extra></extra>",
            ),
            row=1, col=col,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=col)
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=1, col=col)
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title_text=prob_label, row=1, col=1)
    fig.update_xaxes(title_text="Trading days from peak", row=1, col=1)
    fig.update_xaxes(title_text="Trading days from trough", row=1, col=2)
    fig.update_layout(
        height=420,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _render_reactivity(
    qualities: list[AssetQuality], scope: str, dd_threshold: float, prob_label: str
) -> None:
    st.subheader("3 · Reactivity — does the signal catch crashes and recoveries?")
    st.caption(
        "Anchored on real drawdowns. Good shape: a near-vertical climb right "
        "after T=0 on the crash panel (every day of lag is drawdown absorbed) "
        "and a controlled decay over a few weeks on the recovery panel "
        "(instant drops there usually mean whipsaw, not skill)."
    )
    in_scope = qualities if scope == "__all__" else [
        q for q in qualities if q.symbol == scope
    ]
    events_frames = [
        q.events.assign(asset_name=q.asset_name, symbol=q.symbol)
        for q in in_scope if not q.events.empty
    ]
    if not events_frames:
        st.info(
            f"No drawdown events ≥ {dd_threshold:.0%} in this window — lower "
            "the threshold or widen the start date."
        )
        return
    peak_stats = trajectory_stats(
        [event_aligned_matrix(q.events["peak_date"], q.prob) for q in in_scope if not q.events.empty]
    )
    trough_stats = trajectory_stats(
        [event_aligned_matrix(q.events["trough_date"], q.prob) for q in in_scope if not q.events.empty]
    )
    if peak_stats.empty and trough_stats.empty:
        st.info("Too few events near the sample interior for aligned trajectories.")
    else:
        st.plotly_chart(
            _trajectory_panel_figure(peak_stats, trough_stats, prob_label),
            width='stretch',
        )
        st.caption(
            "Mean signal across events with interquartile band; offsets with "
            f"fewer than {_MIN_EVENTS_PER_OFFSET} contributing events are masked."
        )

    events = pd.concat(events_frames, ignore_index=True).sort_values("depth")
    display = events[
        [
            "asset_name", "symbol", "peak_date", "trough_date", "depth",
            "crash_lag", "detect_date", "missed_exit_ret",
            "recovery_lag", "reentry_date", "missed_rebound_ret",
        ]
    ].rename(
        columns={
            "asset_name": "Asset", "symbol": "Symbol", "peak_date": "Peak",
            "trough_date": "Trough", "depth": "Depth",
            "crash_lag": "Crash lag (d)", "detect_date": "Detected",
            "missed_exit_ret": "Missed exit",
            "recovery_lag": "Recovery lag (d)", "reentry_date": "Re-entered",
            "missed_rebound_ret": "Missed rebound",
        }
    )
    st.markdown(f"**Drawdown events ≥ {dd_threshold:.0%}** ({len(display)})")
    st.dataframe(
        display.style.format(
            {
                "Peak": "{:%Y-%m-%d}", "Trough": "{:%Y-%m-%d}",
                "Detected": "{:%Y-%m-%d}", "Re-entered": "{:%Y-%m-%d}",
                "Depth": "{:.1%}", "Missed exit": "{:.1%}",
                "Missed rebound": "{:.1%}",
                "Crash lag (d)": "{:.0f}", "Recovery lag (d)": "{:.0f}",
            },
            na_rep="—",
        ),
        width='stretch',
        hide_index=True,
    )
    st.caption(
        "*Missed exit* = price return from peak until the signal turned Bear "
        "(drawdown absorbed before reacting). *Missed rebound* = return from "
        "trough until it turned Bull again (recovery foregone). “—” = the "
        "signal never reacted within the event."
    )


def render_regime_quality_tab(
    forecast_panel: pd.DataFrame,
    regime_panel: pd.DataFrame,
    prices: pd.DataFrame,
    config: PipelineConfig,
    *,
    key_prefix: str = "rq",
) -> None:
    """Render the full Regime quality tab.

    ``forecast_panel`` / ``regime_panel`` are the long panels from
    :func:`pipeline.forecast.build_forecasts` and
    :func:`pipeline.regime.build_regimes`; ``prices`` the wide Adj-Close
    frame from :func:`pipeline.data.load_prices`.
    """
    st.subheader("Regime quality")
    st.caption(
        "Is the Asset Regime Chart accurate, stable, and tradable? "
        "Distinctiveness · stability · reactivity, graded per asset."
    )
    if (
        forecast_panel is None or forecast_panel.empty
        or regime_panel is None or regime_panel.empty
        or prices is None or prices.empty
    ):
        st.info("Forecast panel, regime panel and prices are required — run the pipeline first.")
        return

    dates = pd.to_datetime(forecast_panel["date"])
    span_years = (dates.max() - dates.min()).days / 365.25
    if span_years < 5:
        st.warning(
            f"Only {span_years:.1f} years in the current window — reactivity "
            "and revision statistics need full market cycles. Widen the start "
            "date (e.g. 2007-01-01) in the sidebar for reliable numbers."
        )

    ctrl_src, ctrl_dd, ctrl_asset = st.columns([2, 2, 2])
    source_label = ctrl_src.selectbox(
        "Signal to grade",
        list(_SIGNAL_SOURCES),
        index=0,
        key=f"{key_prefix}_source",
        help="The smoothed signal is what the executor trades. JM labels are "
             "fit in-sample — the ceiling, not a tradable signal.",
    )
    signal_source = _SIGNAL_SOURCES[source_label]
    dd_threshold = ctrl_dd.slider(
        "Drawdown event threshold",
        min_value=5, max_value=40, value=15, step=1,
        format="%d%%",
        key=f"{key_prefix}_dd",
        help="Peak-to-trough move that counts as a reactivity event.",
    ) / 100.0

    qualities = compute_quality(
        prices, regime_panel, forecast_panel,
        dd_threshold=dd_threshold, signal_source=signal_source,
    )
    if not qualities:
        st.info("No symbols overlap between prices, regimes and forecasts.")
        return
    asset_options = {"All assets (pooled)": "__all__"} | {
        f"{q.asset_name} ({q.symbol})": q.symbol for q in qualities
    }
    scope_label = ctrl_asset.selectbox(
        "Asset detail",
        list(asset_options),
        index=0,
        key=f"{key_prefix}_asset",
    )
    scope = asset_options[scope_label]

    scoreboard = build_scoreboard(qualities)
    _render_scoreboard(scoreboard, signal_label=source_label)
    st.divider()
    _render_distinctiveness(qualities, scoreboard, scope)
    st.divider()

    vintage_path = (
        Path(config.cache_dir)
        / f"regime_vintages_{config.pool}_{config.feature_version}.parquet"
    )
    vintages = append_vintage(regime_panel, vintage_path, date.today().isoformat())
    _render_stability(qualities, forecast_panel, vintages, scope)
    st.divider()

    prob_label = "Fraction in Bear" if signal_source == "jm" else "P(Bear)"
    _render_reactivity(qualities, scope, dd_threshold, prob_label)


__all__ = [
    "AssetQuality",
    "append_vintage",
    "bear_signal",
    "build_scoreboard",
    "compute_quality",
    "detect_drawdown_events",
    "event_aligned_matrix",
    "extract_spells",
    "flip_comparison",
    "next_day_spread",
    "regime_feature_frame",
    "regime_w2",
    "render_regime_quality_tab",
    "revision_stats",
    "signal_lags",
    "spell_stats",
    "survival_curve",
    "trajectory_stats",
    "w2_gaussian_2d",
]
