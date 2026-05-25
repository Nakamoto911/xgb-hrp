"""Eval primitives: EvalCheck protocol, EvalResult, EvalContext.

Each per-category module (data_eval, allocator_eval, …) exposes a
``CHECKS`` list of ``EvalCheck`` instances. The top-level registry in
``pipeline.eval`` runs them against a populated :class:`EvalContext`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from pipeline.config import PipelineConfig


@dataclass
class EvalContext:
    """Bundle of inputs passed to every check.

    Fields are optional so individual phases can run subsets of checks
    (e.g. a data-layer audit doesn't need a BacktestResult).
    """
    config: PipelineConfig
    prices: Optional[pd.DataFrame] = None
    regime_panel: Optional[pd.DataFrame] = None
    forecast_panel: Optional[pd.DataFrame] = None
    backtest_result: Optional[Any] = None      # pipeline.executor.BacktestResult
    benchmark_navs: dict[str, pd.Series] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    name: str
    category: str
    passed: bool
    metric: float = float("nan")
    message: str = ""
    critical: bool = False


@dataclass
class EvalCheck:
    name: str
    category: str
    fn: Callable[[EvalContext], EvalResult]
    critical: bool = False

    def run(self, ctx: EvalContext) -> EvalResult:
        try:
            r = self.fn(ctx)
            # Allow checks to omit category/name; fill from registration.
            r.name = r.name or self.name
            r.category = r.category or self.category
            r.critical = self.critical
            return r
        except Exception as e:
            return EvalResult(
                name=self.name, category=self.category, passed=False,
                message=f"check raised {type(e).__name__}: {e}", critical=self.critical,
            )


def passed(name: str, *, metric: float = float("nan"), message: str = "") -> EvalResult:
    return EvalResult(name=name, category="", passed=True, metric=metric, message=message)


def failed(name: str, *, metric: float = float("nan"), message: str = "") -> EvalResult:
    return EvalResult(name=name, category="", passed=False, metric=metric, message=message)
