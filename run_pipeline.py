"""CLI entry point — Phase 1 + Phase 2 commands.

Subcommands implemented so far:

    data      : load + clean + cache prices for the configured pool
    regime    : per-asset JM regime panel (uses cached walk-forward when present)
    forecast  : per-asset XGB forecast panel (shares walk-forward cache with regime)
    select    : apply the configured selection rule + emit per-date selected lists
    allocate  : run the configured allocator at end-of-window and print weights
    all       : run the full Phase-1 + Phase-2 sequence

Later phases (executor, risk monitor, performance + eval framework) per SPEC.md §17.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline.allocator import allocate
from pipeline.config import PipelineConfig
from pipeline.data import load_prices, load_risk_free
from pipeline.eval import (
    EvalContext,
    any_critical_failed,
    write_eval_report,
)
from pipeline.eval import (
    run_all as run_all_evals,
)
from pipeline.executor import Executor
from pipeline.forecast import PRICE_RULES, build_forecasts
from pipeline.performance import build_benchmark_navs, build_report, write_report
from pipeline.regime import build_regimes
from pipeline.selector import select

PHASES = ("data", "regime", "forecast", "select", "allocate", "backtest", "report", "all")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_pipeline", description=__doc__)
    p.add_argument("--phase", choices=PHASES, default="all")
    p.add_argument("--pool", choices=["etf", "mutual_fund", "european"], default="etf")
    p.add_argument("--start-date", default="2007-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--allocator", choices=["hrp", "ew", "momentum_30d", "min_vol_30d"], default=None
    )
    p.add_argument(
        "--forecast-method",
        choices=[
            "prob_threshold", "regime_and_prob", "ewma_smoothed", "trend",
            "last_day_regime", "ma200", "hybrid",
        ],
        default=None,
    )
    p.add_argument("--bull-prob-threshold", type=float, default=None)
    p.add_argument("--ma-window", type=int, default=None)
    p.add_argument("--hybrid-bear-threshold", type=float, default=None)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument(
        "--gate",
        action="store_true",
        help="After --phase report (or all), exit non-zero if any critical eval check fails.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--config", type=Path, default=None,
                   help="Optional YAML config; CLI flags override its values.")
    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    base: dict = {}
    if args.config is not None:
        base.update(PipelineConfig.from_yaml(args.config).model_dump())
    overrides = {
        "pool": args.pool,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "cache_dir": args.cache_dir,
        "n_jobs": args.n_jobs,
        "allocator": args.allocator,
        "forecast_method": args.forecast_method,
        "bull_prob_threshold": args.bull_prob_threshold,
        "ma_window": args.ma_window,
        "hybrid_bear_threshold": args.hybrid_bear_threshold,
    }
    base.update({k: v for k, v in overrides.items() if v is not None})
    return PipelineConfig.model_validate(base)


def _phase_data(cfg: PipelineConfig, force_refresh: bool) -> pd.DataFrame:
    t0 = time.perf_counter()
    prices = load_prices(cfg, force_refresh=force_refresh)
    dt = time.perf_counter() - t0
    print(
        f"[data] prices shape={prices.shape} "
        f"first={prices.index.min().date()} last={prices.index.max().date()} "
        f"elapsed={dt:.2f}s"
    )
    return prices


def _phase_regime(cfg: PipelineConfig, force_refresh: bool) -> pd.DataFrame:
    t0 = time.perf_counter()
    out = build_regimes(cfg, force_refresh=force_refresh)
    dt = time.perf_counter() - t0
    print(
        f"[regime] panel shape={out.panel.shape} "
        f"symbols={out.panel['symbol'].nunique()} path={out.path} elapsed={dt:.2f}s"
    )
    return out.panel


def _phase_forecast(cfg: PipelineConfig, force_refresh: bool) -> pd.DataFrame:
    t0 = time.perf_counter()
    out = build_forecasts(cfg, force_refresh=force_refresh)
    dt = time.perf_counter() - t0
    print(
        f"[forecast] panel shape={out.panel.shape} "
        f"symbols={out.panel['symbol'].nunique()} path={out.path} elapsed={dt:.2f}s"
    )
    return out.panel


def _phase_select(
    cfg: PipelineConfig, forecast_panel: pd.DataFrame, prices: pd.DataFrame | None = None
):
    t0 = time.perf_counter()
    out = select(forecast_panel, cfg, prices=prices)
    dt = time.perf_counter() - t0
    sel_counts = pd.Series({d: len(s) for d, s in out.by_date.items()})
    print(
        f"[select] rule={cfg.forecast_method} theta={cfg.bull_prob_threshold} "
        f"dates={len(out.by_date)} avg_selected={sel_counts.mean():.2f} "
        f"empty_dates={(sel_counts == 0).sum()} elapsed={dt:.2f}s"
    )
    return out


def _phase_allocate(
    cfg: PipelineConfig,
    prices: pd.DataFrame,
    selected_today: list[str],
) -> pd.Series:
    """Demo: allocate using the last lookback window of returns."""
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    if not selected_today:
        print("[allocate] selection empty — would route to risk-free.")
        return pd.Series(dtype=float)
    t0 = time.perf_counter()
    w = allocate(returns, selected_today, cfg)
    dt = time.perf_counter() - t0
    print(
        f"[allocate] allocator={cfg.allocator} selected={len(selected_today)} "
        f"elapsed={dt:.2f}s"
    )
    if not w.empty:
        for s, x in w.sort_values(ascending=False).items():
            print(f"           {s:6s}  {x:.4f}")
    return w


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else (
        logging.INFO if args.verbose == 1 else logging.DEBUG
    )
    logging.basicConfig(
        level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    cfg = _config_from_args(args)
    print(
        f"pool={cfg.pool} window=[{cfg.start_date}, {cfg.end_date or 'today'}] "
        f"cache_dir={cfg.cache_dir} n_jobs={cfg.n_jobs} "
        f"allocator={cfg.allocator} rule={cfg.forecast_method}"
    )

    prices: pd.DataFrame | None = None
    forecast_panel: pd.DataFrame | None = None
    sel_out = None

    if args.phase in ("data", "all"):
        prices = _phase_data(cfg, args.force_refresh)
    if args.phase in ("regime", "all"):
        _phase_regime(cfg, args.force_refresh)
    if args.phase in ("forecast", "select", "allocate", "all"):
        forecast_panel = _phase_forecast(cfg, args.force_refresh)
    if args.phase in ("select", "allocate", "all"):
        assert forecast_panel is not None
        if prices is None and cfg.forecast_method in PRICE_RULES:
            prices = load_prices(cfg)  # warm-cache hit; needed by the price-aware rules
        sel_out = _phase_select(cfg, forecast_panel, prices)
    if args.phase in ("allocate", "all"):
        if prices is None:
            prices = load_prices(cfg)  # warm-cache hit
        last_date = max(sel_out.by_date) if sel_out and sel_out.by_date else None
        selected_today = sel_out.by_date.get(last_date, []) if last_date else []
        _phase_allocate(cfg, prices, selected_today)
    backtest_result = None
    prices_with_rf = None
    if args.phase in ("backtest", "report", "all"):
        if prices is None:
            prices = load_prices(cfg)
        if forecast_panel is None:
            forecast_panel = _phase_forecast(cfg, args.force_refresh)
        rf_series = load_risk_free(cfg, force_refresh=args.force_refresh)
        prices_with_rf = pd.concat(
            [prices, rf_series.rename(cfg.risk_free_asset)], axis=1
        ).sort_index()
        prices_with_rf = prices_with_rf.ffill()  # pad weekends/holiday mismatches
        t0 = time.perf_counter()
        backtest_result = Executor(
            config=cfg, prices=prices_with_rf, forecast_panel=forecast_panel,
            initial_capital=100_000.0,
        ).run()
        result = backtest_result
        dt = time.perf_counter() - t0
        out_dir = cfg.cache_dir
        end_str = cfg.end_date or pd.Timestamp.today().date().isoformat()
        result.nav.to_frame("nav").to_parquet(
            out_dir / f"nav_{cfg.pool}_{cfg.start_date}_{end_str}_{cfg.feature_version}.parquet"
        )
        if not result.trades.empty:
            result.trades.to_parquet(
                out_dir / f"trades_{cfg.pool}_{cfg.start_date}_{end_str}_{cfg.feature_version}.parquet"
            )
        if not result.risk_events.empty:
            result.risk_events.to_parquet(
                out_dir / f"risk_events_{cfg.pool}_{cfg.start_date}_{end_str}_{cfg.feature_version}.parquet"
            )
        nav = result.nav
        cum_return = nav.iloc[-1] / nav.iloc[0] - 1.0
        daily_ret = nav.pct_change().dropna()
        vol_ann = float(daily_ret.std() * (252 ** 0.5))
        sharpe = float(daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0.0
        mdd = float((nav / nav.cummax() - 1.0).min())
        n_trades = len(result.trades)
        n_risk_events = len(result.risk_events)
        print(
            f"[backtest] nav_final=${nav.iloc[-1]:,.0f} cum_return={cum_return:+.2%} "
            f"vol_ann={vol_ann:.2%} sharpe={sharpe:.2f} mdd={mdd:.2%}"
        )
        print(
            f"           trades={n_trades} total_tc=${result.total_tc:,.0f} "
            f"total_tax=${result.total_tax:,.0f} risk_events={n_risk_events} "
            f"elapsed={dt:.2f}s"
        )

    if args.phase in ("report", "all"):
        if backtest_result is None:
            print("[report] no backtest_result available; run --phase backtest first.")
            return 1
        t0 = time.perf_counter()
        report = build_report(
            cfg, backtest_result,
            prices_with_rf=prices_with_rf,
            force_refresh=args.force_refresh,
        )
        report_path = write_report(report, cfg.cache_dir)
        # Eval run.
        ctx = EvalContext(
            config=cfg,
            prices=prices,
            forecast_panel=forecast_panel,
            backtest_result=backtest_result,
            benchmark_navs=build_benchmark_navs(
                cfg, backtest_result.nav, force_refresh=False
            ) if backtest_result is not None else {},
        )
        results = run_all_evals(ctx)
        end_str = cfg.end_date or pd.Timestamp.today().date().isoformat()
        eval_path = write_eval_report(
            results, cfg.cache_dir,
            pool=cfg.pool, start=cfg.start_date, end=end_str, version=cfg.feature_version,
        )
        n_pass = sum(1 for r in results if r.passed)
        n_fail = sum(1 for r in results if not r.passed)
        n_crit = sum(1 for r in results if (not r.passed) and r.critical)
        dt = time.perf_counter() - t0
        print(
            f"[report] report={report_path} eval={eval_path} "
            f"checks={n_pass}pass/{n_fail}fail (crit_fail={n_crit}) elapsed={dt:.2f}s"
        )
        if args.gate and any_critical_failed(results):
            print(f"[gate] critical eval failure ({n_crit}) — exiting non-zero")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
