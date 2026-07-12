"""Grid-sweep driver around ``compare_rules.compare`` — allocator × execution ×
risk-monitor sensitivity, one combined tidy table + Markdown report.

Runs the **full portfolio backtest** (allocator + executor + PFU tax + risk
monitor) for every point in a hard-coded grid of

    allocator            × rebalance_frequency × drift_threshold × risk_monitor

against warm caches, reusing :func:`compare_rules.compare` for the 4
selection-rule variants (production_theta060, production_theta040, ma200,
hybrid) plus the two gross benchmarks (S&P_BH, 60_40) at each grid point.
Configs run in parallel via ``ProcessPoolExecutor``; a failure in one config
is logged and skipped rather than aborting the sweep.

Never touches the network — every config reads the same warm Parquet caches
``compare_rules.compare`` requires; a cold cache raises ``FileNotFoundError``
naming the exact missing path (see ``compare_rules._require_warm_caches``).

Usage::

    python scripts/sweep_rules.py --pool etf --start-date 2007-01-01
    python scripts/sweep_rules.py --dry-run          # list the 24 configs, exit
    python scripts/sweep_rules.py --limit 2           # smoke test
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# Make both this script's directory (for `from compare_rules import ...`) and
# the repo root (for `from pipeline...` regardless of caller cwd) importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compare_rules import _COL_LABELS, _METRIC_COLS, _fmt_cell, compare  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Grid definitions
# -----------------------------------------------------------------------------
# A grid point is a plain, picklable dict of overrides — deliberately not a
# class, so ProcessPoolExecutor's spawn-based workers can build the
# PipelineConfig themselves without needing anything more than this module's
# top-level state.
GridPoint = dict[str, object]

ALLOCATORS: list[str] = ["ew", "hrp"]
REBALANCE_FREQUENCIES: list[str] = ["monthly", "quarterly"]
DRIFT_THRESHOLDS: list[float] = [0.015, 0.05, 0.10]

# Named risk-monitor settings — "rare_fire" widens the hysteresis band
# (bear_prob_threshold 0.70→0.85, universe_pct_threshold 0.40→0.60, smoothed
# signal) so the monitor only trips on genuinely broad, persistent bear
# readings instead of every daily XGB jitter.
RISK_MONITOR_SETTINGS: dict[str, dict[str, object]] = {
    "off": {"risk_monitor_enabled": False},
    "rare_fire": {
        "risk_monitor_enabled": True,
        "risk_monitor_signal": "smoothed",
        "bear_prob_threshold": 0.85,
        "universe_pct_threshold": 0.60,
    },
}
RISK_MONITOR_ORDER: list[str] = ["off", "rare_fire"]

_BENCHMARK_NAMES: tuple[str, ...] = ("S&P_BH", "60_40")
_TIDY_COLS: list[str] = [
    "allocator", "rebalance_frequency", "drift_threshold", "risk_monitor", "variant",
    *_METRIC_COLS,
]


def _build_grid() -> list[GridPoint]:
    """24 configs: allocator × rebalance_frequency × drift_threshold × risk_monitor.

    Deterministic ``itertools.product`` order (allocator outermost, risk
    monitor innermost) — matches the order the fields are listed above.
    """
    return [
        {
            "allocator": a,
            "rebalance_frequency": f,
            "drift_threshold": d,
            "risk_monitor": m,
        }
        for a, f, d, m in itertools.product(
            ALLOCATORS, REBALANCE_FREQUENCIES, DRIFT_THRESHOLDS, RISK_MONITOR_ORDER
        )
    ]


def _label(point: GridPoint) -> str:
    return (
        f"allocator={point['allocator']}, freq={point['rebalance_frequency']}, "
        f"drift={point['drift_threshold']}, monitor={point['risk_monitor']}"
    )


# -----------------------------------------------------------------------------
# Worker — module top-level, plain picklable arguments (macOS spawn safe).
# -----------------------------------------------------------------------------
def _worker_init(log_level: int) -> None:
    """Configure logging inside each worker process (spawn doesn't inherit it)."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _run_grid_point(
    point: GridPoint,
    base_kwargs: dict[str, object],
) -> tuple[GridPoint, list[dict[str, object]]]:
    """Build the PipelineConfig for one grid point and run ``compare`` on it.

    Runs in a separate process — takes only picklable plain data
    (``point``/``base_kwargs`` dicts) and reconstructs everything else
    (config, imports) locally. Returns ``(point, records)`` where ``records``
    is ``compare()``'s result table as a list of row-dicts (one per variant +
    one per benchmark), keeping the pickled payload simple.
    """
    overrides: dict[str, object] = dict(base_kwargs)
    overrides["allocator"] = point["allocator"]
    overrides["rebalance_frequency"] = point["rebalance_frequency"]
    overrides["drift_threshold"] = point["drift_threshold"]
    overrides.update(RISK_MONITOR_SETTINGS[point["risk_monitor"]])  # type: ignore[index]

    cfg = PipelineConfig.model_validate(overrides)
    table = compare(cfg, write=False, print_table=False)
    records = table.reset_index().rename(columns={"index": "variant"}).to_dict("records")
    return point, records


# -----------------------------------------------------------------------------
# Sweep driver — importable, no subprocess required.
# -----------------------------------------------------------------------------
def run_sweep(
    grid: list[GridPoint],
    *,
    pool: str,
    start_date: str,
    end_date: str | None,
    cache_dir: Path,
    jobs: int,
    log_level: int = logging.INFO,
) -> tuple[pd.DataFrame, list[GridPoint]]:
    """Run every grid point in parallel and aggregate into one tidy DataFrame.

    Returns ``(tidy, failed)``: ``tidy`` has columns ``_TIDY_COLS`` with the
    two benchmark rows once at the top (allocator/rebalance_frequency/
    risk_monitor = "benchmark", drift_threshold = NaN) followed by one row
    per (grid point × variant) in grid order. ``failed`` lists the grid
    points whose config raised (already logged) so the caller can report a
    nonzero exit code.
    """
    base_kwargs: dict[str, object] = {
        "pool": pool, "start_date": start_date, "end_date": end_date, "cache_dir": cache_dir,
    }
    n = len(grid)
    t0 = time.perf_counter()
    records_by_idx: dict[int, list[dict[str, object]]] = {}
    failed: list[GridPoint] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs, initializer=_worker_init, initargs=(log_level,)
    ) as executor:
        future_to_idx = {
            executor.submit(_run_grid_point, point, base_kwargs): idx
            for idx, point in enumerate(grid)
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            point = grid[idx]
            done += 1
            elapsed = time.perf_counter() - t0
            try:
                _, records = future.result()
            except Exception:
                logger.exception("config failed: %s", _label(point))
                failed.append(point)
                continue
            records_by_idx[idx] = records
            logger.info(
                "config %d/%d done (allocator=%s, freq=%s, drift=%s, monitor=%s) elapsed=%.1fs",
                done, n, point["allocator"], point["rebalance_frequency"],
                point["drift_threshold"], point["risk_monitor"], elapsed,
            )

    benchmark_rows: dict[str, dict[str, object]] = {}
    grid_rows: list[dict[str, object]] = []
    for idx, point in enumerate(grid):
        records = records_by_idx.get(idx)
        if records is None:
            continue
        for rec in records:
            variant = rec["variant"]
            metrics = {c: rec[c] for c in _METRIC_COLS}
            if variant in _BENCHMARK_NAMES:
                if variant not in benchmark_rows:
                    benchmark_rows[variant] = {
                        "allocator": "benchmark",
                        "rebalance_frequency": "benchmark",
                        "drift_threshold": float("nan"),
                        "risk_monitor": "benchmark",
                        "variant": variant,
                        **metrics,
                    }
                continue
            grid_rows.append({
                "allocator": point["allocator"],
                "rebalance_frequency": point["rebalance_frequency"],
                "drift_threshold": point["drift_threshold"],
                "risk_monitor": point["risk_monitor"],
                "variant": variant,
                **metrics,
            })

    bench_list = [benchmark_rows[name] for name in _BENCHMARK_NAMES if name in benchmark_rows]
    tidy = pd.DataFrame(bench_list + grid_rows, columns=_TIDY_COLS)
    return tidy, failed


# -----------------------------------------------------------------------------
# Markdown rendering — label columns rendered locally; metric columns reuse
# compare_rules._fmt_cell so the two reports stay visually consistent.
# -----------------------------------------------------------------------------
_LABEL_COLS: list[str] = ["allocator", "rebalance_frequency", "drift_threshold", "risk_monitor", "variant"]
_LABEL_HEADERS: dict[str, str] = {
    "allocator": "Allocator", "rebalance_frequency": "Rebalance",
    "drift_threshold": "Drift", "risk_monitor": "Monitor", "variant": "Series",
}


def _fmt_label_cell(col: str, v: object) -> str:
    if col == "drift_threshold":
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"  # em dash
        return f"{float(v):g}"
    return str(v)


def _render_sweep_table(df: pd.DataFrame) -> str:
    header = [_LABEL_HEADERS[c] for c in _LABEL_COLS] + [_COL_LABELS[c] for c in _METRIC_COLS]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for _, row in df.iterrows():
        vals = [_fmt_label_cell(c, row[c]) for c in _LABEL_COLS]
        vals += [_fmt_cell(c, row[c]) for c in _METRIC_COLS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _render_sweep_markdown(
    tidy: pd.DataFrame,
    *,
    pool: str,
    start_date: str,
    end: str,
    n_configs: int,
    failed: list[GridPoint],
) -> str:
    ref = PipelineConfig()  # only used for the (unswept) tax/cost defaults below
    lines = [
        "# Rule / allocator / execution grid sweep",
        "",
        f"- Pool: `{pool}`  |  Window: `{start_date}` → `{end}`",
        f"- Grid ({n_configs} configs): allocator=`{ALLOCATORS}` × "
        f"rebalance_frequency=`{REBALANCE_FREQUENCIES}` × "
        f"drift_threshold=`{DRIFT_THRESHOLDS}` × risk_monitor=`{RISK_MONITOR_ORDER}`",
        "- Risk monitor settings:",
        "  - `off`: risk_monitor_enabled=False",
        "  - `rare_fire`: risk_monitor_enabled=True, risk_monitor_signal=`smoothed`, "
        "bear_prob_threshold=`0.85`, universe_pct_threshold=`0.60`",
        f"- Strategy rows are **NET** of PFU tax (rate={ref.pfu_rate:.1%}) and transaction "
        f"costs ({ref.transaction_cost_bps:.1f} bps); benchmark rows (`S&P_BH`, `60_40`) are "
        "**GROSS** (no tax/costs) and listed once — they don't vary by "
        "allocator/rebalance/drift/monitor.",
    ]
    if failed:
        lines.append(
            f"- **{len(failed)} config(s) failed** and are omitted below: "
            + "; ".join(_label(p) for p in failed)
        )
    lines.append("")

    sorted_tidy = tidy.sort_values("cagr", ascending=False, na_position="last")
    lines.append(_render_sweep_table(sorted_tidy))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _default_jobs() -> int:
    cpu = os.cpu_count() or 4
    return max(1, min(6, cpu - 2))


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep_rules",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pool", choices=["etf", "mutual_fund", "european"], default="etf")
    p.add_argument("--start-date", default="2007-01-01")
    p.add_argument("--end-date", default=None, help="Default: today (PipelineConfig semantics).")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"))
    p.add_argument(
        "--jobs", type=int, default=_default_jobs(),
        help="Parallel worker processes (default: min(6, cpu_count - 2), floored at 1).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N grid configs (smoke test).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the grid and exit without running anything.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def _print_dry_run(grid: list[GridPoint]) -> None:
    print(f"Sweep grid ({len(grid)} configs):")
    for i, point in enumerate(grid, start=1):
        print(f"  {i:2d}. {_label(point)}")


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    # Long-running batch tool: INFO by default so progress is always visible.
    level = logging.INFO if args.verbose == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    grid = _build_grid()
    if args.limit is not None:
        grid = grid[: max(args.limit, 0)]

    if args.dry_run:
        _print_dry_run(grid)
        return 0

    jobs = max(1, args.jobs)
    end = args.end_date or date.today().isoformat()
    logger.info(
        "Starting sweep: pool=%s window=%s→%s configs=%d jobs=%d cache_dir=%s",
        args.pool, args.start_date, end, len(grid), jobs, args.cache_dir,
    )

    try:
        tidy, failed = run_sweep(
            grid,
            pool=args.pool, start_date=args.start_date, end_date=args.end_date,
            cache_dir=args.cache_dir, jobs=jobs, log_level=level,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.cache_dir / f"sweep_rules_{args.pool}_{args.start_date}_{end}.csv"
    md_path = args.cache_dir / f"sweep_rules_{args.pool}_{args.start_date}_{end}.md"

    tidy.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    md = _render_sweep_markdown(
        tidy, pool=args.pool, start_date=args.start_date, end=end,
        n_configs=len(grid), failed=failed,
    )
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)

    print(md)

    if failed:
        logger.error("%d/%d configs failed: %s", len(failed), len(grid), [_label(p) for p in failed])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
