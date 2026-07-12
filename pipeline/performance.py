"""Performance metrics + benchmark NAVs + sub-period decomposition + report.

Module 9 of SPEC.md (§11).

Metrics computed per the xgboost-repo conventions:
    cum_return, ann_vol, sharpe, sortino, mdd, calmar, turnover, tax_drag, hit_rate

Benchmarks:
    S&P B&H        — total-return index, no transaction costs or taxes
    Vanguard 60/40 — single ETF (VSMGX or EUR-equivalent), no costs either

Sub-period decomposition (spec §11.3):
    GFC 2007-2009, Recovery 2010-2014, Late Cycle 2015-2019, COVID 2020,
    Post-COVID 2021-onwards. Sub-periods are clipped to the available NAV range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.data import load_benchmarks
from pipeline.executor import BacktestResult

# Spec §11.3 sub-period boundaries (start_inclusive, end_inclusive).
SUB_PERIODS: dict[str, tuple[str, str]] = {
    "GFC":         ("2007-01-01", "2009-12-31"),
    "Recovery":    ("2010-01-01", "2014-12-31"),
    "Late_Cycle":  ("2015-01-01", "2019-12-31"),
    "COVID":       ("2020-01-01", "2020-12-31"),
    "Post_COVID":  ("2021-01-01", "2099-12-31"),
}

_TRADING_DAYS = 252


# -----------------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return a / b if b not in (0.0, 0) and np.isfinite(b) else 0.0


def compute_metrics(
    nav: pd.Series,
    *,
    total_tc: float = 0.0,
    total_tax: float = 0.0,
    turnover_annual: float | None = None,
) -> dict[str, float]:
    """Compute standard performance metrics on a daily-NAV series."""
    if nav.empty or len(nav) < 2:
        return {k: 0.0 for k in (
            "n_days", "cum_return", "cagr", "ann_vol", "sharpe", "sortino",
            "mdd", "calmar", "hit_rate", "turnover_annual",
            "tax_drag", "tc_drag",
        )}
    nav = nav.astype(float).dropna()
    ret = nav.pct_change().dropna()
    n_days = len(nav)
    years = n_days / _TRADING_DAYS

    cum_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = (1.0 + cum_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(_TRADING_DAYS))
    downside = ret[ret < 0].std(ddof=1)
    sharpe = _safe_div(float(ret.mean()) * _TRADING_DAYS, ann_vol)
    sortino = _safe_div(
        float(ret.mean()) * _TRADING_DAYS,
        float(downside) * np.sqrt(_TRADING_DAYS) if not np.isnan(downside) else 0.0,
    )
    running_max = nav.cummax()
    mdd = float((nav / running_max - 1.0).min())
    calmar = _safe_div(cagr, abs(mdd))

    # Monthly hit rate.
    monthly = nav.resample("BME").last()
    monthly_ret = monthly.pct_change().dropna()
    hit_rate = float((monthly_ret > 0).mean()) if not monthly_ret.empty else 0.0

    start_nav = float(nav.iloc[0])
    return {
        "n_days": n_days,
        "cum_return": cum_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "turnover_annual": float(turnover_annual) if turnover_annual is not None else 0.0,
        "tax_drag": _safe_div(total_tax, start_nav),
        "tc_drag": _safe_div(total_tc, start_nav),
    }


def compute_turnover_annual(trades: pd.DataFrame, nav: pd.Series) -> float:
    """Annual one-way dollar-weighted turnover ≈ Σ |trade_value| / mean_nav / years."""
    if trades.empty or nav.empty:
        return 0.0
    trades = trades.copy()
    trades["notional"] = trades["units"] * trades["price"]
    total = float(trades["notional"].abs().sum())
    mean_nav = float(nav.mean())
    years = max(1.0, len(nav) / _TRADING_DAYS)
    return _safe_div(total, mean_nav) / years


# -----------------------------------------------------------------------------
# Sub-period decomposition (§11.3)
# -----------------------------------------------------------------------------
def sub_period_metrics(
    nav: pd.Series, **metrics_kwargs
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, (start, end) in SUB_PERIODS.items():
        window = nav.loc[start:end]
        if len(window) < 5:
            continue
        # Rebase to 1.0 at window start for cum_return semantics.
        rebased = window / window.iloc[0]
        out[name] = compute_metrics(rebased, **metrics_kwargs)
    return out


# -----------------------------------------------------------------------------
# Benchmark NAVs
# -----------------------------------------------------------------------------
def build_buy_and_hold_nav(
    prices: pd.DataFrame, ticker: str, initial_capital: float = 100_000.0
) -> pd.Series:
    if ticker not in prices.columns:
        raise KeyError(f"Benchmark ticker {ticker!r} missing from prices panel.")
    p = prices[ticker].dropna()
    if p.empty:
        return pd.Series(dtype=float, name=ticker)
    units = initial_capital / float(p.iloc[0])
    return (p * units).rename(ticker)


def build_benchmark_navs(
    config: PipelineConfig,
    nav: pd.Series,
    *,
    initial_capital: float = 100_000.0,
    force_refresh: bool = False,
) -> dict[str, pd.Series]:
    """Load + align benchmark NAVs to the strategy NAV's date range."""
    bench_prices = load_benchmarks(config, force_refresh=force_refresh)
    if bench_prices.empty or nav.empty:
        return {}
    bench_prices = bench_prices.reindex(nav.index).ffill().bfill()
    out: dict[str, pd.Series] = {}
    if config.benchmark_bh and config.benchmark_bh in bench_prices.columns:
        out["S&P_BH"] = build_buy_and_hold_nav(bench_prices, config.benchmark_bh, initial_capital)
    if config.benchmark_6040 and config.benchmark_6040 in bench_prices.columns:
        out["60_40"] = build_buy_and_hold_nav(bench_prices, config.benchmark_6040, initial_capital)
    return out


# -----------------------------------------------------------------------------
# Report bundle + renderer
# -----------------------------------------------------------------------------
@dataclass
class PerformanceReport:
    config: PipelineConfig
    strategy_metrics: dict[str, float]
    strategy_sub_periods: dict[str, dict[str, float]]
    benchmark_metrics: dict[str, dict[str, float]]
    benchmark_sub_periods: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    risk_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    tax_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    def render_markdown(self) -> str:
        lines: list[str] = []
        cfg = self.config
        lines.append(f"# Pipeline Report — pool=`{cfg.pool}`")
        lines.append("")
        lines.append(f"- Window: `{cfg.start_date}` → `{cfg.end_date or 'today'}`")
        lines.append(f"- Allocator: `{cfg.allocator}` | rebalance `{cfg.rebalance_frequency}` | drift `{cfg.drift_threshold}` | policy `{cfg.execution_policy}`")
        lines.append(f"- Forecast rule: `{cfg.forecast_method}` θ={cfg.bull_prob_threshold}")
        lines.append(
            f"- Risk monitor: enabled={cfg.risk_monitor_enabled} "
            f"(bear_th={cfg.bear_prob_threshold}, trigger@{cfg.universe_pct_threshold:.0%}, "
            f"clear@{cfg.universe_pct_clear_threshold:.0%}, dwell={cfg.risk_off_dwell_days}d, "
            f"reenter={cfg.reenter_mode})"
        )
        lines.append("")

        # Headline metrics table (strategy + benchmarks side-by-side).
        cols = ["sharpe", "sortino", "cagr", "ann_vol", "mdd", "calmar", "hit_rate", "tc_drag", "tax_drag"]
        rows = [("Strategy", self.strategy_metrics)]
        for name, m in self.benchmark_metrics.items():
            rows.append((name, m))
        lines.append("## Headline metrics")
        lines.append("")
        lines.append("| Series | " + " | ".join(cols) + " |")
        lines.append("|---|" + "|".join("---" for _ in cols) + "|")
        for label, m in rows:
            row_vals = []
            for c in cols:
                v = m.get(c, 0.0)
                if c in ("cagr", "ann_vol", "mdd", "hit_rate", "tc_drag", "tax_drag"):
                    row_vals.append(f"{v:+.2%}")
                else:
                    row_vals.append(f"{v:+.2f}")
            lines.append(f"| {label} | " + " | ".join(row_vals) + " |")
        lines.append("")

        # Sub-period decomposition.
        if self.strategy_sub_periods:
            lines.append("## Sub-period Sharpe")
            lines.append("")
            sub_cols = list(self.strategy_sub_periods)
            head = ["Series"] + sub_cols
            lines.append("| " + " | ".join(head) + " |")
            lines.append("|" + "|".join("---" for _ in head) + "|")
            def _fmt_row(label: str, subs: dict[str, dict[str, float]]) -> str:
                vals = [f"{subs.get(p, {}).get('sharpe', float('nan')):+.2f}" if p in subs else "—" for p in sub_cols]
                return f"| {label} | " + " | ".join(vals) + " |"
            lines.append(_fmt_row("Strategy", self.strategy_sub_periods))
            for name, subs in self.benchmark_sub_periods.items():
                lines.append(_fmt_row(name, subs))
            lines.append("")

        # Risk events log.
        if not self.risk_events.empty:
            lines.append(f"## Risk monitor events ({len(self.risk_events)} transitions)")
            lines.append("")
            lines.append("| Date | Transition | State after | Universe bear % |")
            lines.append("|---|---|---|---|")
            for _, r in self.risk_events.iterrows():
                lines.append(
                    f"| {pd.Timestamp(r['date']).date()} | {r['transition']} | "
                    f"{r['state_after']} | {r['universe_bear_pct']:.0%} |"
                )
            lines.append("")

        # Tax history.
        if not self.tax_history.empty:
            total_tax = float(self.tax_history["tax_due"].sum())
            outstanding = float(self.tax_history.iloc[-1]["carryforward_outstanding"])
            lines.append(f"## Tax accruals (PFU {cfg.pfu_rate:.1%}, AVCO basis, {cfg.loss_carryforward_years}y carryforward)")
            lines.append("")
            lines.append(f"- Total tax paid: **${total_tax:,.0f}**")
            lines.append(f"- Outstanding loss carryforward: ${outstanding:,.0f}")
            lines.append("")
            lines.append("| Year | Net realized | Carryforward applied | Tax due | Carryforward outstanding |")
            lines.append("|---|---|---|---|---|")
            for _, r in self.tax_history.iterrows():
                lines.append(
                    f"| {int(r['year'])} | ${r['net_realized']:,.0f} | "
                    f"${r['applied_carryforward']:,.0f} | ${r['tax_due']:,.0f} | "
                    f"${r['carryforward_outstanding']:,.0f} |"
                )
            lines.append("")

        return "\n".join(lines)


def build_report(
    config: PipelineConfig,
    result: BacktestResult,
    *,
    prices_with_rf: pd.DataFrame | None = None,
    initial_capital: float = 100_000.0,
    force_refresh: bool = False,
) -> PerformanceReport:
    """Compose a full PerformanceReport from a BacktestResult."""
    turnover = compute_turnover_annual(result.trades, result.nav)
    strategy_metrics = compute_metrics(
        result.nav,
        total_tc=result.total_tc,
        total_tax=result.total_tax,
        turnover_annual=turnover,
    )
    strategy_sub = sub_period_metrics(result.nav)

    benchmark_navs = build_benchmark_navs(
        config, result.nav, initial_capital=initial_capital, force_refresh=force_refresh
    )
    benchmark_metrics = {
        name: compute_metrics(nav) for name, nav in benchmark_navs.items() if not nav.empty
    }
    benchmark_subs = {
        name: sub_period_metrics(nav) for name, nav in benchmark_navs.items() if not nav.empty
    }
    return PerformanceReport(
        config=config,
        strategy_metrics=strategy_metrics,
        strategy_sub_periods=strategy_sub,
        benchmark_metrics=benchmark_metrics,
        benchmark_sub_periods=benchmark_subs,
        risk_events=result.risk_events,
        tax_history=result.tax_history,
    )


def write_report(report: PerformanceReport, out_dir: Path) -> Path:
    end = report.config.end_date or date.today().isoformat()
    path = out_dir / f"report_{report.config.pool}_{report.config.start_date}_{end}_{report.config.feature_version}.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(report.render_markdown())
    return path


__all__ = [
    "PerformanceReport",
    "SUB_PERIODS",
    "build_buy_and_hold_nav",
    "build_benchmark_navs",
    "build_report",
    "compute_metrics",
    "compute_turnover_annual",
    "sub_period_metrics",
    "write_report",
]
