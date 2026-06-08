"""Streamlit chart helpers ported from vendor/xgboost/portfolio_construction.py.

Two charts:

* :func:`asset_regime_chart` — daily P(bear) heatmap per asset (risk-ordered
  rows from defensive → risk-on), with Resolution and Time-range pickers.
  Mirrors the chart in vendor lines 382-500.

* :func:`portfolio_composition_chart` — stacked-bar of MVO weights per
  period with the cumulative wealth line overlaid on a secondary y-axis.
  Mirrors the chart in vendor lines 786-935.

Both functions own their own Streamlit widgets (selectbox / radio) and
return ``None``; the calling page only needs to invoke them in order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# Defensive → risk-on order used by the regime heatmap (vendor convention).
_RISK_ORDER = [
    "Treasury", "AggBond", "Corporate", "HighYield",
    "LargeCap", "MidCap", "SmallCap",
    "EAFE", "EM",
    "REIT", "Commodity", "Gold",
]
_RISK_RANK = {n: i for i, n in enumerate(_RISK_ORDER)}

# Asset-class colour palette (vendor _ASSET_META).
_ASSET_META: dict[str, tuple[str, str]] = {
    "LargeCap":  ("#0D47A1", "LargeCap"),
    "MidCap":    ("#1976D2", "MidCap"),
    "SmallCap":  ("#64B5F6", "SmallCap"),
    "EAFE":      ("#1B5E20", "EAFE"),
    "EM":        ("#66BB6A", "EM"),
    "AggBond":   ("#B71C1C", "AggBond"),
    "Treasury":  ("#E53935", "Treasury"),
    "Corporate": ("#EF9A9A", "Corporate"),
    "HighYield": ("#FFCDD2", "HighYield"),
    "REIT":      ("#6A1B9A", "REIT"),
    "Commodity": ("#E65100", "Commodity"),
    "Gold":      ("#FDD835", "Gold"),
    "Cash":      ("#000000", "Cash"),
}

# Bottom-of-stack → top draw order.
_DRAW_ORDER = [
    "LargeCap", "MidCap", "SmallCap",
    "EAFE", "EM",
    "AggBond", "Treasury", "Corporate", "HighYield",
    "REIT",
    "Commodity", "Gold",
    "Cash",
]

_FREQ_MAP = {"Daily": None, "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
_RANGE_OFFSETS = {
    "1M": pd.DateOffset(months=1),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}


# -----------------------------------------------------------------------------
# Asset regime heatmap
# -----------------------------------------------------------------------------
def asset_regime_chart(
    forecast_panel: pd.DataFrame, *, key_prefix: str = "regime"
) -> None:
    """Render the Asset Regime heatmap (P(bear) per asset over time).

    ``forecast_panel`` is the long-form output of
    :func:`pipeline.forecast.build_forecasts` with columns
    ``[symbol, asset_name, date, p_bear, ...]``.
    """
    st.subheader("Asset Regime Chart")

    if forecast_panel is None or forecast_panel.empty:
        st.info("No forecast panel — run the pipeline first.")
        return

    ctrl_res, ctrl_range, _ = st.columns([1, 2, 2])
    resolution = ctrl_res.selectbox(
        "Resolution",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        index=0,
        key=f"{key_prefix}_res",
    )
    range_label = ctrl_range.radio(
        "Time range",
        ["1M", "6M", "1Y", "5Y", "ALL"],
        index=2,
        horizontal=True,
        key=f"{key_prefix}_range",
    )

    # Pivot to date × asset_name (P(bear)).
    pivot = (
        forecast_panel.pivot_table(
            index="date", columns="asset_name", values="p_bear", aggfunc="mean"
        )
        .sort_index()
    )
    # Resample to the chosen resolution.
    resample_rule = _FREQ_MAP[resolution]
    if resample_rule is not None:
        pivot = pivot.resample(resample_rule).mean()

    # Clip to time range.
    end_ts = pivot.index.max()
    start_ts = (
        end_ts - _RANGE_OFFSETS[range_label] if range_label != "ALL" else pivot.index.min()
    )
    pivot = pivot.loc[start_ts:end_ts]

    if pivot.empty:
        st.info(f"No data in the {range_label} window.")
        return

    # Risk-ordered y-axis (defensive at the top).
    assets = sorted(pivot.columns, key=lambda n: _RISK_RANK.get(n, len(_RISK_ORDER)))
    pivot = pivot[assets]

    # Build (symbol) label suffix from the forecast panel itself.
    name_to_symbol = (
        forecast_panel.groupby("asset_name")["symbol"].first().to_dict()
    )
    y_labels = [f"{a} ({name_to_symbol.get(a, a)})" for a in assets]

    if resolution == "Quarterly":
        x_labels = [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in pivot.index]
    else:
        date_fmt = {"Daily": "%Y-%m-%d", "Weekly": "%Y-%m-%d", "Monthly": "%Y-%m"}[resolution]
        x_labels = [d.strftime(date_fmt) for d in pivot.index]

    z_matrix = [
        [None if pd.isna(v) else float(v) for v in pivot[a].to_numpy()]
        for a in assets
    ]

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "rgb(34,139,34)"],
            [0.5, "rgb(255,215,0)"],
            [1.0, "rgb(200,30,30)"],
        ],
        zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b><br>%{x}<br>P(Bear): %{z:.0%}<extra></extra>",
        colorbar=dict(
            title="P(Bear)",
            tickformat=".0%",
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            len=0.6,
        ),
        xgap=1, ygap=2,
    ))
    row_height_px = 36
    fig.update_layout(
        height=max(300, row_height_px * len(assets) + 80),
        template="plotly_dark",
        plot_bgcolor="rgb(55,55,55)",
        margin=dict(l=10, r=10, t=10, b=60),
        xaxis=dict(side="bottom", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )
    st.plotly_chart(fig, width='stretch')


# -----------------------------------------------------------------------------
# Portfolio composition stacked-bar + cumulative wealth
# -----------------------------------------------------------------------------
def portfolio_composition_chart(
    weights: pd.DataFrame,
    nav: pd.Series,
    *,
    symbol_to_asset_name: dict[str, str],
    risk_free_symbol: str,
    strategy_label: str = "Strategy",
    key_prefix: str = "comp",
) -> None:
    """Stacked-bar of per-period MVO weights + cumulative-wealth overlay.

    ``weights`` is the date × symbol weight matrix from BacktestResult.
    ``nav`` is the daily NAV series (used to compute cumulative wealth).
    ``symbol_to_asset_name`` maps yfinance symbols to vendor class names
    (LargeCap / AggBond / …) so the colour palette and draw order kick in.
    The risk-free leg is relabelled "Cash" before plotting.
    """
    st.subheader("Portfolio composition over time")
    st.caption(
        "Stacked bar chart of allocator weights per period. Assets grouped "
        "by type (same-type assets share a colour family). The risk-free "
        f"leg ({risk_free_symbol}) is shown as 'Cash'."
    )

    if weights is None or weights.empty:
        st.info("No weights matrix — run the pipeline first.")
        return

    ctrl_res, ctrl_range, _ = st.columns([1, 2, 2])
    freq_label = ctrl_res.selectbox(
        "Resolution",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        index=2,
        key=f"{key_prefix}_freq",
    )
    range_label = ctrl_range.radio(
        "Time range",
        ["1M", "6M", "1Y", "5Y", "ALL"],
        index=4,
        horizontal=True,
        key=f"{key_prefix}_range",
    )

    # Translate columns: symbol → asset_name (with RF leg → "Cash").
    rename = {
        s: ("Cash" if s == risk_free_symbol else symbol_to_asset_name.get(s, s))
        for s in weights.columns
    }
    w = weights.rename(columns=rename).copy()
    # Collapse possible duplicate columns (if two symbols map to the same name).
    w = w.T.groupby(level=0).sum().T

    # Time-range filter.
    end_ts = w.index.max()
    start_ts = (
        end_ts - _RANGE_OFFSETS[range_label] if range_label != "ALL" else w.index.min()
    )
    w = w[w.index >= start_ts]

    # Resample.
    if freq_label == "Daily":
        w_rs = w.copy()
    else:
        w_rs = w.resample({"Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}[freq_label]).mean()
    # Any uninvested fraction (slippage from drift, post-tax shortfall) → augment "Cash".
    residual = (1.0 - w_rs.sum(axis=1)).clip(lower=0.0)
    w_rs["Cash"] = w_rs.get("Cash", pd.Series(0.0, index=w_rs.index)) + residual

    # Re-order columns to the canonical bottom→top stack.
    cols_present = [c for c in _DRAW_ORDER if c in w_rs.columns]
    extras = [c for c in w_rs.columns if c not in _DRAW_ORDER and c != "Cash"]
    w_rs = w_rs[cols_present + extras]

    # Hover text: only show non-trivial weights, sorted desc.
    hover_texts = []
    fmt = "%Y-%m-%d" if freq_label == "Daily" else "%Y-%m"
    for date, row in w_rs.iterrows():
        nonzero = row[row > 0.0005].sort_values(ascending=False)
        lines = [f"<b>{pd.Timestamp(date).strftime(fmt)}</b>"]
        for col, val in nonzero.items():
            lines.append(f" {col}: {val:.1%}")
        hover_texts.append("<br>".join(lines))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in w_rs.columns:
        color, label = _ASSET_META.get(col, ("#CCCCCC", col))
        fig.add_trace(
            go.Bar(
                x=w_rs.index,
                y=w_rs[col].round(4),
                name=label,
                marker_color=color,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    # Cumulative wealth on secondary axis (rebase to 1.0).
    wealth = nav / nav.iloc[0]
    wealth = wealth[wealth.index >= start_ts]
    if freq_label != "Daily":
        wealth = wealth.resample(
            {"Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}[freq_label]
        ).last()
    fig.add_trace(
        go.Scatter(
            x=wealth.index,
            y=wealth.values,
            name="Wealth",
            line=dict(color="white", width=2),
            hovertemplate="%{x|%Y-%m}: %{y:.2f}x<extra>Wealth</extra>",
        ),
        secondary_y=True,
    )

    # Invisible per-period tooltip carrier (so hover shows the sorted weights).
    fig.add_trace(
        go.Scatter(
            x=w_rs.index,
            y=[0.5] * len(w_rs),
            mode="markers",
            marker=dict(opacity=0, size=18),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
            showlegend=False,
            name="",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        barmode="stack",
        title=(
            f"Portfolio weights & cumulative wealth — {strategy_label} "
            f"({freq_label.lower()}, {range_label})"
        ),
        xaxis_title="Date",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        yaxis2_title="Cumulative wealth",
        yaxis2=dict(showgrid=False),
        height=520,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch')


__all__ = ["asset_regime_chart", "portfolio_composition_chart"]
