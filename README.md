# xgb-hrp

End-to-end quant investment pipeline that fuses a per-asset Statistical Jump
Model + XGBoost regime forecaster with a Hierarchical Risk Parity allocator,
runs through a drift-band executor with French PFU / AVCO tax accounting,
and rides on top of a daily Risk Monitor that can force a risk-off override
between scheduled rebalances.

See [SPEC.md](SPEC.md) for the full v1.1 design. This README is the operator manual.

---

## What's in the box

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐
│ data (Module 2) │→ │ regime + forecast│→ │ select + alloc  │→ │ executor +   │
│  per-pool prices│  │  per-asset       │  │  HRP / EW / mom │  │  AVCO ledger │
│  + parquet cache│  │  JM walk-forward │  │  / min-vol      │  │  + risk      │
│  + FRED macros  │  │  + XGB forecast  │  │                 │  │  monitor     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └──────┬───────┘
                                                                      ▼
                                       ┌──────────────────────────────────────┐
                                       │ performance + benchmarks + eval (§12)│
                                       │ markdown report + parquet artifacts  │
                                       └──────────────────────────────────────┘
```

- 3 asset pools: `etf` (12 US ETFs), `mutual_fund` (12 US MFs), `european` (20 UCITS).
- 5 user-selectable forecast rules, 4 allocators, 6 rebalance frequencies.
- True AVCO cost-basis ledger with vintage-aged 10-year loss carryforward.
- Daily Risk Monitor with hysteresis (separate trigger/clear thresholds + dwell).
- 15 eval checks across 5 categories; `--gate` exits non-zero on critical fail.

---

## Quick start

```bash
# One-time: clone with submodules (vendor/hrp + vendor/xgboost).
git clone --recurse-submodules <repo-url> xgb-hrp
cd xgb-hrp

# Create venv (Python ≥ 3.11) and install in editable mode.
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

# Smoke run on the ETF pool (cold ≈ 60s, warm ≈ 2s).
python run_pipeline.py --phase all --pool etf \
       --start-date 2023-01-01 --end-date 2024-06-30

# Full 19-year validation with gate enforcement.
python run_pipeline.py --phase all --pool etf \
       --start-date 2007-01-01 --gate

# Launch the Streamlit UI.
streamlit run app.py
```

After a run, look in `cache/`:

| File | Contents |
|---|---|
| `prices_{pool}_{start}_{end}_v1.parquet`     | Cleaned price panel |
| `wf_{pool}_{symbol}_…parquet`                | Per-asset JM+XGB walk-forward (shared by Modules 3 & 4) |
| `regimes_{pool}_…parquet`                    | Per-asset JM regime labels |
| `forecasts_{pool}_…parquet`                  | Per-asset XGB forecast panel |
| `nav_{pool}_…parquet`                        | Daily NAV |
| `trades_{pool}_…parquet`                     | Full trade ledger with realized gains |
| `risk_events_{pool}_…parquet`                | Risk monitor state transitions |
| `report_{pool}_…md`                          | Headline metrics + sub-period table + tax accruals |
| `eval_report_{pool}_…md`                     | Per-check pass/fail with metrics |

---

## CLI reference

```
python run_pipeline.py [OPTIONS]

  --phase  {data, regime, forecast, select, allocate, backtest, report, all}
  --pool   {etf, mutual_fund, european}
  --start-date, --end-date         ISO dates (end defaults to today)
  --cache-dir PATH                 default: ./cache
  --n-jobs INT                     joblib outer parallelism, default -1
  --allocator {hrp, ew, momentum_30d, min_vol_30d}
  --forecast-method {prob_threshold, regime_and_prob, ewma_smoothed,
                     trend, last_day_regime, ma200, hybrid}
  --bull-prob-threshold FLOAT
  --ma-window INT                  SMA window (days) for ma200 / hybrid, default 200
  --hybrid-bear-threshold FLOAT    smoothed p_bear gate for hybrid, default 0.80
  --config PATH                    optional YAML; CLI flags override
  --force-refresh                  rebuild every cache from scratch
  --gate                           exit non-zero on critical eval fail
  -v[v]                            INFO / DEBUG logging
```

Each `--phase` can be run on its own — later phases reuse cached artifacts
from earlier ones. Useful patterns:

```bash
# Tune the selection rule without re-running JM/XGB:
python run_pipeline.py --phase report --pool etf --start-date 2007-01-01 \
       --forecast-method prob_threshold --bull-prob-threshold 0.55

# Allocator ablation (cached signals, fresh executor + report):
for alloc in hrp ew momentum_30d min_vol_30d; do
  python run_pipeline.py --phase report --pool etf --start-date 2007-01-01 \
         --allocator "$alloc"
done
```

---

## Configuration

Every run is driven by [`pipeline.config.PipelineConfig`](pipeline/config.py)
— a pydantic-v2 model that mirrors SPEC.md §14 verbatim. The Streamlit UI
binds 1:1 to it; the CLI takes a YAML override file via `--config`.

```python
from pipeline.config import PipelineConfig

cfg = PipelineConfig(
    pool="etf",
    forecast_method="ewma_smoothed",
    bull_prob_threshold=0.60,
    allocator="hrp",
    rebalance_frequency="quarterly",
    risk_monitor_enabled=True,
    bear_prob_threshold=0.70,
    universe_pct_threshold=0.40,
    universe_pct_clear_threshold=0.25,
    reenter_mode="immediate_fresh",  # immediate re-allocation on clearance
    start_date="2007-01-01",
)
```

A `model_validator` enforces `universe_pct_clear_threshold <
universe_pct_threshold` (the hysteresis sanity rule). Per-pool defaults
(risk-free leg, B&H and 60/40 benchmark tickers) resolve automatically.

---

## Eval framework + `--gate`

Each phase has its own checks ([pipeline/eval/](pipeline/eval/)). The registry
runs them against a populated `EvalContext` and produces
`eval_report_{pool}_…md`.

| Category | Critical checks | Informational checks |
|---|---|---|
| `data`         | `missing_data`, `calendar_alignment` | `outliers`, `stale_streak` |
| `allocator`    | `weight_sum`, `non_negativity`       | `concentration` (HHI) |
| `executor`     | `cost_accounting`, `avco_recompute`  | `risk_off_latency` |
| `risk_monitor` | —                                    | `hysteresis` |
| `e2e`          | `causality_audit`                    | `beats_bh`, `beats_6040`, `reproducibility` |

`avco_recompute` re-walks the trade ledger through an independent
`AVCOLedger` and asserts the running `avg_cost` matches the live ledger to
1e-6. This is the spec §12.6 check that protects against tax-drag
regressions.

`--gate` exits with code 2 if any **critical** check fails. Wire it into
pre-commit / CI before publishing a benchmark.

---

## Testing

```bash
pytest                            # 75 tests, ~1s wall time
pytest tests/test_ledger.py -v    # focused: AVCO + TaxBook properties
pytest -k risk_monitor            # focused: trigger / hysteresis / dwell
```

Coverage spans every public module: config validators, data layer cache,
forecast rules, allocator dispatch (each of 4 allocators on synthetic
returns), executor smoke (with and without the risk monitor — verifies the
ablation), eval registry, and the markdown report renderer.

---

## Repository layout

```
xgb-hrp/
├── pipeline/
│   ├── config.py          # PipelineConfig — single source of truth
│   ├── data.py            # parquet cache + parallel yfinance + FRED
│   ├── _walk_forward.py   # shared per-asset JM+XGB worker (loky)
│   ├── _pools.py          # European pool asset spec table
│   ├── regime.py          # Module 3 view (JM regime labels)
│   ├── forecast.py        # Module 4 view + 5 selection rules
│   ├── selector.py        # Module 5 — per-date selection + diffs
│   ├── allocator.py       # Module 6 — HRP / EW / momentum / min-vol
│   ├── executor.py        # Module 7 — AVCO ledger + drift-band loop
│   ├── risk_monitor.py    # Module 8 — daily trigger + hysteresis
│   ├── performance.py     # Module 9 — metrics + sub-periods + report
│   ├── eval/              # §12 eval framework
│   └── _vendor.py         # sys.path shim for vendor submodules
├── vendor/
│   ├── hrp/               # github.com/Nakamoto911/hrp     (submodule)
│   └── xgboost/           # github.com/Nakamoto911/xgboost (submodule)
├── tests/                 # 75 pytest cases
├── app.py                 # Streamlit UI
├── run_pipeline.py        # CLI
├── SPEC.md                # v1.1 design spec
└── README.md              # this file
```

---

## Performance characteristics

Measured on a 10-core M-series Mac, ETF pool (12 assets):

| Operation                           | Cold     | Warm   |
|-------------------------------------|----------|--------|
| Data fetch (12 ETFs, 18 months)     | ~0.7s    | <0.1s  |
| Data fetch (12 ETFs, 19 years)      | ~10s     | <0.1s  |
| JM+XGB walk-forward (18 months)     | ~82s     | <0.2s  |
| JM+XGB walk-forward (19 years)      | ~176s    | <0.2s  |
| Selector + allocator (19 years)     | n/a      | <0.5s  |
| Executor + risk monitor (19 years)  | n/a      | ~2.5s  |
| Report + eval                       | n/a      | <0.5s  |
| **Full pipeline (19 years, warm)**  | —        | **~4s** |

The vendor walk-forward dominates cold latency. Warm-cache hits (re-running
with a tweaked selection rule or allocator) are dominated by the executor
loop and complete in ~5s on 19 years of daily data.

---

## v1 acceptance criteria (per SPEC.md §17)

| # | Criterion | Status |
|---|---|---|
| 1 | All 3 pools cold < 90s on 8-core | ❌ 176s on ETF (vendor walk-forward limit) |
| 2 | All eval gates pass on ETF default config | ✅ 0 critical fail on 19-year window |
| 3 | Beats S&P B&H on Sharpe AND Max DD | ⚠️ MDD wins (58% reduction), Sharpe loses on 19-year bull market |
| 4 | Risk Monitor MDD reduction ≥ 20% | ✅ 58% achieved |
| 5 | Streamlit UI binding 1:1 to PipelineConfig | ✅ |
| 6 | CLI emits MD report + parquet artifacts | ✅ |
| 7 | `--gate` exits non-zero on critical fail | ✅ both directions verified |

The two open items are not code defects — see SPEC.md §16 "Open Questions"
for the strategy-tuning and latency-optimization work that would close
them.

---

## Vendor submodules

This repo depends on two upstream projects pinned as git submodules:

- `vendor/hrp/`     — github.com/Nakamoto911/hrp     (HRP allocator + PFU backtest)
- `vendor/xgboost/` — github.com/Nakamoto911/xgboost (JM model + XGBoost forecaster + paper feature math)

The orchestration layer in `pipeline/` is the only new code. We intentionally
don't fork: bumping the submodule pins picks up upstream improvements
without merge conflicts. See `pipeline/_vendor.py` for the sys.path shim
that resolves their script-style imports.

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

---

## License

TBD. The vendored submodules have their own licenses — see their respective
LICENSE files.
