"""Selection-rule comparison harness — one command, one combined table.

Runs the **full portfolio backtest** (allocator + executor + PFU tax + risk
monitor) for several `forecast_method` / `bull_prob_threshold` variants
against the *same* cached prices / risk-free / forecast panel, and emits a
single Markdown comparison table: variant rows (net of tax + transaction
costs) plus the two gross benchmark rows (S&P B&H, 60/40).

This complements the per-asset "detector shootout" in the Regime Cockpit
(see MEMORY.md, 2026-07-07 entry) with the portfolio-level verdict: does
`ma200` / `hybrid` still beat the production `ewma_smoothed` rule once the
risk monitor, HRP allocator, drift-band execution and PFU tax all interact
with the selection rule, not just a per-asset long/flat arena?

Never touches the network. Every loader hits a warm Parquet cache written by
``run_pipeline.py``; a missing cache file fails fast with the exact path and
the command to populate it, e.g.::

    python run_pipeline.py --phase all --pool etf --start-date 2007-01-01

Usage::

    python scripts/compare_rules.py --pool etf --start-date 2007-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.data import _cache_path, load_prices, load_risk_free
from pipeline.executor import BacktestResult, Executor
from pipeline.forecast import build_forecasts, forecast_panel_cache_path
from pipeline.performance import PerformanceReport, build_report

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Variant definitions — (name, PipelineConfig field overrides). Row order below
# is the table row order.
# -----------------------------------------------------------------------------
Variant = tuple[str, dict[str, object]]

VARIANTS: list[Variant] = [
    # Pre-change production default (SPEC.md original bull_prob_threshold).
    ("production_theta060", {"forecast_method": "ewma_smoothed", "bull_prob_threshold": 0.60}),
    # New default / MEMORY.md "quick win" (2026-07-07 entry, step 1).
    ("production_theta040", {"forecast_method": "ewma_smoothed", "bull_prob_threshold": 0.40}),
    # Pure price-trend rule (MEMORY.md step 2 — "decisive test").
    ("ma200", {"forecast_method": "ma200"}),
    # Price-trend + smoothed-p_bear backstop (defaults: ma_window=200,
    # hybrid_bear_threshold=0.80).
    ("hybrid", {"forecast_method": "hybrid"}),
]

_METRIC_COLS = [
    "cagr", "sharpe", "sortino", "mdd", "calmar", "ann_vol",
    "turnover_annual", "tax_drag", "tc_drag", "trades_per_year", "n_risk_events",
]
_PCT_COLS = {"cagr", "ann_vol", "mdd", "turnover_annual", "tax_drag", "tc_drag"}
_COL_LABELS = {
    "cagr": "CAGR", "sharpe": "Sharpe", "sortino": "Sortino", "mdd": "MDD",
    "calmar": "Calmar", "ann_vol": "Ann. Vol", "turnover_annual": "Ann. Turnover",
    "tax_drag": "Tax Drag", "tc_drag": "TC Drag", "trades_per_year": "Trades/yr",
    "n_risk_events": "Risk-off events",
}


@dataclass
class VariantResult:
    """One variant's raw backtest + report output, plus harness-computed extras."""

    name: str
    config: PipelineConfig
    result: BacktestResult
    report: PerformanceReport
    n_trades: int
    trades_per_year: float
    n_risk_events: int


# -----------------------------------------------------------------------------
# Warm-cache guard — never fetch; fail fast naming the exact missing path.
# -----------------------------------------------------------------------------
def _require_warm_caches(cfg: PipelineConfig) -> None:
    """Raise ``FileNotFoundError`` listing every missing cache file + the fix.

    Market data is not available in this environment (network blocked), so
    this harness must never attempt a fetch — it only ever reads pre-warmed
    Parquet caches written by ``run_pipeline.py``.
    """
    end = cfg.end_date or date.today().isoformat()
    checks = [
        ("prices", _cache_path(cfg.cache_dir, cfg.pool, cfg.start_date, end, cfg.feature_version)),
        ("risk-free", _cache_path(
            cfg.cache_dir, f"{cfg.pool}_rf", cfg.start_date, end, cfg.feature_version
        )),
        ("benchmarks", _cache_path(
            cfg.cache_dir, f"{cfg.pool}_bench", cfg.start_date, end, cfg.feature_version
        )),
        ("forecast panel", forecast_panel_cache_path(cfg)),
    ]
    missing = [(label, path) for label, path in checks if not path.exists()]
    if not missing:
        return
    lines = [
        "compare_rules: missing warm cache file(s) — this harness never hits the "
        "network. Missing:",
    ]
    for label, path in missing:
        lines.append(f"  [{label}] {path}")
    lines.append("")
    lines.append(
        "Populate all caches first with:\n"
        f"    python run_pipeline.py --phase all --pool {cfg.pool} "
        f"--start-date {cfg.start_date} --end-date {end} --cache-dir {cfg.cache_dir}"
    )
    raise FileNotFoundError("\n".join(lines))


# -----------------------------------------------------------------------------
# Per-variant run
# -----------------------------------------------------------------------------
def _run_variant(
    name: str,
    config: PipelineConfig,
    prices_with_rf: pd.DataFrame,
    forecast_panel: pd.DataFrame,
) -> VariantResult:
    t0 = time.perf_counter()
    result = Executor(
        config=config, prices=prices_with_rf, forecast_panel=forecast_panel,
        initial_capital=100_000.0,
    ).run()
    report = build_report(config, result, prices_with_rf=prices_with_rf)
    n_days = report.strategy_metrics.get("n_days", 0) or 0
    years = n_days / 252.0
    n_trades = len(result.trades)
    trades_per_year = (n_trades / years) if years > 0 else 0.0
    n_risk_events = len(result.risk_events) if result.risk_events is not None else 0
    dt = time.perf_counter() - t0
    logger.info(
        "[%s] nav_final=$%.0f trades=%d risk_events=%d elapsed=%.2fs",
        name, result.nav.iloc[-1] if not result.nav.empty else float("nan"),
        n_trades, n_risk_events, dt,
    )
    return VariantResult(
        name=name, config=config, result=result, report=report,
        n_trades=n_trades, trades_per_year=trades_per_year, n_risk_events=n_risk_events,
    )


# -----------------------------------------------------------------------------
# Table + markdown rendering
# -----------------------------------------------------------------------------
def _build_table(variant_results: list[VariantResult]) -> pd.DataFrame:
    """Combined comparison table: one row per variant + one per benchmark.

    Benchmarks are pulled from the *first* variant's report only — every
    variant shares the same warm price/benchmark cache, so recomputing them
    per variant would be pure waste (build_report does recompute internally
    on each call, which is a small accepted waste; we still only take the
    benchmark rows once here).
    """
    rows: dict[str, dict[str, float]] = {}
    for vr in variant_results:
        m = vr.report.strategy_metrics
        rows[vr.name] = {
            "cagr": m.get("cagr", float("nan")),
            "sharpe": m.get("sharpe", float("nan")),
            "sortino": m.get("sortino", float("nan")),
            "mdd": m.get("mdd", float("nan")),
            "calmar": m.get("calmar", float("nan")),
            "ann_vol": m.get("ann_vol", float("nan")),
            "turnover_annual": m.get("turnover_annual", float("nan")),
            "tax_drag": m.get("tax_drag", float("nan")),
            "tc_drag": m.get("tc_drag", float("nan")),
            "trades_per_year": vr.trades_per_year,
            "n_risk_events": float(vr.n_risk_events),
        }
    if variant_results:
        for bench_name, bm in variant_results[0].report.benchmark_metrics.items():
            rows[bench_name] = {
                "cagr": bm.get("cagr", float("nan")),
                "sharpe": bm.get("sharpe", float("nan")),
                "sortino": bm.get("sortino", float("nan")),
                "mdd": bm.get("mdd", float("nan")),
                "calmar": bm.get("calmar", float("nan")),
                "ann_vol": bm.get("ann_vol", float("nan")),
                # Not applicable to a static gross benchmark — 0, not NaN, so
                # the table stays fully finite/renderable.
                "turnover_annual": 0.0,
                "tax_drag": bm.get("tax_drag", 0.0),
                "tc_drag": bm.get("tc_drag", 0.0),
                "trades_per_year": 0.0,
                "n_risk_events": 0.0,
            }
    return pd.DataFrame.from_dict(rows, orient="index", columns=_METRIC_COLS)


def _fmt_cell(col: str, v: float) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"  # em dash
    if col in _PCT_COLS:
        return f"{v:+.2%}"
    if col == "n_risk_events":
        return f"{int(round(v))}"
    if col == "trades_per_year":
        return f"{v:.1f}"
    return f"{v:+.2f}"  # sharpe, sortino, calmar


def _render_markdown_table(table: pd.DataFrame) -> str:
    header = ["Series"] + [_COL_LABELS[c] for c in table.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for label, row in table.iterrows():
        vals = [_fmt_cell(c, row[c]) for c in table.columns]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _render_header(config_base: PipelineConfig, variant_results: list[VariantResult]) -> str:
    end = config_base.end_date or date.today().isoformat()
    lines = [
        "# Selection-rule comparison",
        "",
        f"- Pool: `{config_base.pool}`  |  Window: `{config_base.start_date}` → `{end}`",
        f"- Allocator: `{config_base.allocator}`  |  Rebalance frequency: "
        f"`{config_base.rebalance_frequency}`  |  Drift threshold: "
        f"`{config_base.drift_threshold}`  |  Execution policy: "
        f"`{config_base.execution_policy}`",
        f"- Risk monitor: **{'ENABLED' if config_base.risk_monitor_enabled else 'DISABLED'}** "
        f"(signal=`{config_base.risk_monitor_signal}`, "
        f"bear_prob_threshold=`{config_base.bear_prob_threshold}`, "
        f"universe_pct_threshold=`{config_base.universe_pct_threshold}`, "
        f"universe_pct_clear_threshold=`{config_base.universe_pct_clear_threshold}`, "
        f"risk_off_dwell_days=`{config_base.risk_off_dwell_days}`)",
        "- Strategy rows are **NET** of PFU tax (rate="
        f"{config_base.pfu_rate:.1%}) and transaction costs "
        f"({config_base.transaction_cost_bps:.1f} bps); benchmark rows "
        "(`S&P_BH`, `60_40`) are **GROSS** (no tax/costs).",
        "",
        "Variants:",
    ]
    for vr in variant_results:
        cfg = vr.config
        lines.append(
            f"  - `{vr.name}`: forecast_method=`{cfg.forecast_method}` "
            f"bull_prob_threshold={cfg.bull_prob_threshold} "
            f"ma_window={cfg.ma_window} hybrid_bear_threshold={cfg.hybrid_bear_threshold}"
        )
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Public entry point — importable, no subprocess required.
# -----------------------------------------------------------------------------
def compare(
    config_base: PipelineConfig,
    variants: list[Variant] | None = None,
    *,
    write: bool = True,
    print_table: bool = True,
) -> pd.DataFrame:
    """Run every variant against ``config_base``'s window on shared warm caches.

    Loads prices / risk-free / forecast panel **once** — they are identical
    across every variant, since only the selection rule (``forecast_method``
    and its knobs) changes — then runs the full backtest (allocator +
    executor + PFU tax + risk monitor) per variant via :class:`Executor`.

    Returns a combined comparison table: one row per variant (net of tax +
    transaction costs) plus one row per benchmark (gross). As a side effect
    (unless ``write=False``), also renders the table + a header block as
    Markdown, prints it (unless ``print_table=False``), and writes it to
    ``{cache_dir}/compare_rules_{pool}_{start}_{end}.md``.

    Raises ``FileNotFoundError`` — naming the exact missing cache file(s) and
    the ``run_pipeline.py`` invocation that creates them — if any required
    cache is cold. Never touches the network itself.
    """
    if variants is None:
        variants = VARIANTS
    _require_warm_caches(config_base)

    logger.info("Loading warm caches (prices, risk-free, forecast panel)…")
    prices = load_prices(config_base)
    rf_series = load_risk_free(config_base)
    forecast_panel = build_forecasts(config_base).panel
    prices_with_rf = (
        pd.concat([prices, rf_series.rename(config_base.risk_free_asset)], axis=1)
        .sort_index()
        .ffill()
    )

    variant_results: list[VariantResult] = []
    for name, overrides in variants:
        cfg = config_base.model_copy(update=overrides)
        logger.info("Running variant %r (forecast_method=%s)…", name, cfg.forecast_method)
        variant_results.append(_run_variant(name, cfg, prices_with_rf, forecast_panel))

    table = _build_table(variant_results)
    md = _render_header(config_base, variant_results) + "\n" + _render_markdown_table(table)

    if print_table:
        print(md)
    if write:
        end = config_base.end_date or date.today().isoformat()
        # Non-default policy gets its own file so a flip_only run never
        # clobbers the drift_band baseline table for the same window.
        policy_tag = (
            "" if config_base.execution_policy == "drift_band"
            else f"_{config_base.execution_policy}"
        )
        out_path = (
            config_base.cache_dir
            / f"compare_rules_{config_base.pool}_{config_base.start_date}_{end}{policy_tag}.md"
        )
        config_base.cache_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        logger.info("Wrote %s", out_path)

    return table


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_rules",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pool", choices=["etf", "mutual_fund", "european"], default="etf")
    p.add_argument("--start-date", default="2007-01-01")
    p.add_argument("--end-date", default=None, help="Default: today (PipelineConfig semantics).")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument(
        "--no-risk-monitor", action="store_true",
        help="Run every variant with risk_monitor_enabled=False.",
    )
    p.add_argument(
        "--allocator", choices=["hrp", "ew", "momentum_30d", "min_vol_30d"], default="hrp",
    )
    p.add_argument(
        "--rebalance-frequency",
        choices=["daily", "weekly", "monthly", "quarterly", "semi-annually", "yearly"],
        default=None,
        help="Default: PipelineConfig's default (quarterly).",
    )
    p.add_argument("--drift-threshold", type=float, default=None)
    p.add_argument(
        "--execution-policy", choices=["drift_band", "flip_only"], default=None,
        help="drift_band: band triggers a full re-band to target (SPEC §9.2). "
             "flip_only: tax-aware — sell only on selection flips or to trim "
             "back to target+band; band is a cap, not a trigger (SPEC §16).",
    )
    p.add_argument(
        "--risk-monitor-signal", choices=["raw", "smoothed"], default=None,
        help="Module 8 input signal: raw P(bear) (default) or EWMA-smoothed.",
    )
    p.add_argument("--bear-prob-threshold", type=float, default=None)
    p.add_argument("--universe-pct-threshold", type=float, default=None)
    p.add_argument("--universe-pct-clear-threshold", type=float, default=None)
    p.add_argument("--risk-off-dwell-days", type=int, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    overrides: dict = {
        "pool": args.pool,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "cache_dir": args.cache_dir,
        "allocator": args.allocator,
        "risk_monitor_enabled": not args.no_risk_monitor,
    }
    if args.rebalance_frequency is not None:
        overrides["rebalance_frequency"] = args.rebalance_frequency
    optional_overrides = {
        "drift_threshold": args.drift_threshold,
        "execution_policy": args.execution_policy,
        "risk_monitor_signal": args.risk_monitor_signal,
        "bear_prob_threshold": args.bear_prob_threshold,
        "universe_pct_threshold": args.universe_pct_threshold,
        "universe_pct_clear_threshold": args.universe_pct_clear_threshold,
        "risk_off_dwell_days": args.risk_off_dwell_days,
    }
    overrides.update({k: v for k, v in optional_overrides.items() if v is not None})
    return PipelineConfig.model_validate(overrides)


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else (
        logging.INFO if args.verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config_base = _config_from_args(args)
    try:
        compare(config_base)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
