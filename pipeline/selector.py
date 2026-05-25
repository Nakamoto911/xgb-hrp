"""Asset selector — Module 5 of SPEC.md.

Applies the configured selection rule to the per-asset forecast panel
emitted by :mod:`pipeline.forecast` and returns, per rebalance date, the
list of symbols to allocate to. Edge cases:

* Empty selection → ``selected = []``. The caller routes the rebalance
  to the pool's risk-free asset.
* Singleton selection → returned as-is. HRP, EW, momentum and min-vol
  all collapse to a 1.0 weight gracefully on a one-asset universe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.forecast import apply_rule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorOutput:
    """Bundle of per-date selection results.

    Attributes
    ----------
    flags : pd.DataFrame
        Long-form ``[date, symbol, selected]`` boolean table.
    by_date : dict[pd.Timestamp, list[str]]
        Convenience view: date → ordered list of selected symbols.
    diffs : pd.DataFrame
        Per-rebalance churn ``[date, added, removed, kept]`` for eval/turnover analytics.
    """

    flags: pd.DataFrame
    by_date: dict[pd.Timestamp, list[str]]
    diffs: pd.DataFrame


def _selected_lists(flags: pd.DataFrame) -> dict[pd.Timestamp, list[str]]:
    out: dict[pd.Timestamp, list[str]] = {}
    for d, sub in flags.groupby("date"):
        out[pd.Timestamp(d)] = sub.loc[sub["selected"], "symbol"].sort_values().tolist()
    return out


def _selection_diffs(by_date: dict[pd.Timestamp, list[str]]) -> pd.DataFrame:
    rows = []
    prev: set[str] = set()
    for d in sorted(by_date):
        cur = set(by_date[d])
        rows.append(
            {
                "date": d,
                "n_selected": len(cur),
                "added": sorted(cur - prev),
                "removed": sorted(prev - cur),
                "kept": sorted(cur & prev),
            }
        )
        prev = cur
    return pd.DataFrame(rows)


def select(
    forecast_panel: pd.DataFrame,
    config: PipelineConfig,
    *,
    on_dates: Optional[pd.DatetimeIndex] = None,
) -> SelectorOutput:
    """Run the configured selection rule on a forecast panel.

    ``on_dates`` restricts the output to a specific set of rebalance dates
    (e.g. drift-band rebalances). When ``None`` every date in the panel
    receives a selection.
    """
    flags = apply_rule(
        forecast_panel,
        config.forecast_method,
        theta=config.bull_prob_threshold,
        trend_window=config.trend_window,
    )
    if on_dates is not None:
        flags = flags[flags["date"].isin(on_dates)]

    by_date = _selected_lists(flags)
    diffs = _selection_diffs(by_date)

    empty_dates = [d for d, syms in by_date.items() if not syms]
    if empty_dates:
        logger.info(
            "Selection empty on %d/%d dates — caller should route to risk-free.",
            len(empty_dates), len(by_date),
        )
    return SelectorOutput(flags=flags, by_date=by_date, diffs=diffs)


__all__ = ["SelectorOutput", "select"]
