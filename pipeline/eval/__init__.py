"""Per-step and end-to-end eval registry (Section 12 of SPEC.md).

Usage::

    from pipeline.eval import EvalContext, run_all, render_markdown, any_critical_failed

    ctx = EvalContext(config=cfg, prices=prices, ..., backtest_result=res)
    results = run_all(ctx)
    print(render_markdown(results))
    sys.exit(1 if any_critical_failed(results) else 0)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.eval._base import EvalCheck, EvalContext, EvalResult
from pipeline.eval.allocator_eval import CHECKS as _ALLOC
from pipeline.eval.data_eval import CHECKS as _DATA
from pipeline.eval.e2e_eval import CHECKS as _E2E
from pipeline.eval.executor_eval import CHECKS as _EXEC
from pipeline.eval.risk_eval import CHECKS as _RISK

REGISTRY: list[EvalCheck] = [*_DATA, *_ALLOC, *_EXEC, *_RISK, *_E2E]


def run_all(ctx: EvalContext) -> list[EvalResult]:
    return [check.run(ctx) for check in REGISTRY]


def any_critical_failed(results: list[EvalResult]) -> bool:
    return any((not r.passed) and r.critical for r in results)


def render_markdown(results: list[EvalResult]) -> str:
    lines: list[str] = ["# Eval report", ""]
    by_cat: dict[str, list[EvalResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    n_crit = sum(1 for r in results if (not r.passed) and r.critical)
    lines.append(f"Summary: **{n_pass} pass / {n_fail} fail** ({n_crit} critical fail)")
    lines.append("")
    for category in sorted(by_cat):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Check | Status | Metric | Critical | Message |")
        lines.append("|---|---|---|---|---|")
        for r in by_cat[category]:
            status = "✅ pass" if r.passed else "❌ FAIL"
            metric = f"{r.metric:.4f}" if isinstance(r.metric, float) and not pd.isna(r.metric) else ""
            crit = "yes" if r.critical else ""
            lines.append(f"| {r.name} | {status} | {metric} | {crit} | {r.message} |")
        lines.append("")
    return "\n".join(lines)


def write_eval_report(results: list[EvalResult], out_dir: Path,
                      *, pool: str, start: str, end: str, version: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_report_{pool}_{start}_{end}_{version}.md"
    path.write_text(render_markdown(results))
    return path


__all__ = [
    "EvalCheck",
    "EvalContext",
    "EvalResult",
    "REGISTRY",
    "any_critical_failed",
    "render_markdown",
    "run_all",
    "write_eval_report",
]
