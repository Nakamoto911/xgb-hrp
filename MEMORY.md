# Project research log — xgb-hrp

Durable findings and decisions that aren't recoverable from the code or git
history. Newest first. Keep entries short; link to the scripts/commits that
produced them.

---

## 2026-07-12 — Step-2 decisive test run (user, local): portfolio layer is now the binding constraint

**Setup.** `scripts/compare_rules.py`, ETF pool, 2007-01-01 → 2026-07-12, HRP,
quarterly, net of PFU 31.4% + 5 bps. Benchmarks gross: S&P B&H 8.99% CAGR /
−56.8% MDD; 60/40 6.57% / −41.1%.

**Risk monitor ON (production config):**

| Variant | Net CAGR | MDD | Ann.Vol | Turnover | Tax drag | Risk-off events |
|---|---|---|---|---|---|---|
| production θ=0.60 | 1.26% | −20.7% | 3.5% | 637%/yr | 13.8% | 51 |
| production θ=0.40 | 1.61% | −15.9% | 3.0% | 574%/yr | 17.8% | 51 |
| ma200 | 1.62% | −13.2% | 2.9% | 574%/yr | 17.9% | 51 |
| hybrid | 1.43% | −13.1% | 2.9% | 607%/yr | 15.7% | 51 |

**Risk monitor OFF (ablation):**

| Variant | Net CAGR | MDD | Ann.Vol | Turnover | Tax drag |
|---|---|---|---|---|---|
| production θ=0.60 | −0.90% | −56.6% | 10.1% | 346%/yr | 2.4% |
| production θ=0.40 | −0.48% | −57.9% | 9.8% | 275%/yr | 6.5% |
| ma200 | +1.85% | −22.4% | 6.4% | 283%/yr | 14.3% |
| hybrid | +2.71% | −24.7% | 6.7% | 352%/yr | 30.9% |

**Reads (in order of importance):**

1. **Production selection contributes nothing without the monitor** — negative
   CAGR *and* full B&H-scale −57% MDD. All of production's crash protection
   came from the daily risk monitor, none from the quarterly p_bear selection.
   Selection-rule verdict is final at every level: trend ≥ vol-regime.
2. **The monitor whipsaws:** 51 full liquidate+re-enter cycles ≈ 2.6/yr (vs ~6
   real crash episodes since 2007), each realizing all gains at 31.4%. With it
   ON, portfolio vol is 2.9–3.5% and CAGR ≈ T-bill yield → the book is parked
   in BIL most of the time. Root cause: monitor breadth reads RAW `p_bear`
   (`Raw_Prob`), which flips daily; hysteresis+dwell can't absorb it.
3. **HRP tilts the book into bonds** (inverse-variance → AGG/SPTL/SPBO/HYG/GLD
   heavy; same failure mode documented in vendor/hrp memory.md for the European
   pool): monitor-off vol 6.4–10% vs B&H 19.7%. A large share of the CAGR gap
   vs the per-asset EW shootout (8.8% MA200 daily) is allocator tilt, not
   signal quality.
4. **Churn/taxes:** ~280–350%/yr turnover at quarterly cadence — selection
   flips at HRP weights + 1.5% drift band re-trading every anchor. Hybrid has
   the best gross alpha (highest CAGR despite a 30.9% cumulative tax drag —
   it sells winners on p_bear spikes); a tax-aware policy (wider drift band,
   rebalance-on-flip-only, soft risk-off per SPEC §16) is where the next
   points of CAGR are.
5. Sharpe caveat: performance.py uses raw returns (no rf subtraction), so
   cash-parked variants get flattered Sharpes; read CAGR/MDD instead.

**Next sweep (knobs shipped this commit):** `--allocator ew` (kill the bond
tilt; closest to the per-asset arena), `--rebalance-frequency monthly` (trend
rules like faster anchors; drift band keeps trades rare), `--drift-threshold
0.05`, and rare-fire monitor: `--risk-monitor-signal smoothed
--bear-prob-threshold 0.85 --universe-pct-threshold 0.60`.

---

## 2026-07-12 — Paper-fidelity audit: why production loses to MA200; ma200/hybrid rules shipped

**Question.** Production (JM+XGB per arXiv 2406.09578) loses to MA200 per-asset net
(2026-07-07 entry). Implementation bug, mis-application of the paper, or wrong tool
for a taxed low-frequency account?

**Findings (code + paper audit — refcard: `vendor/xgboost/refcard.md`):**

1. **The implementation is faithful; the expectation was not.** The vendor repo was
   validated hard against the paper (JM matches the authors' `jumpmodels` library
   with 100% state agreement; XGB uses paper-default hyperparameters; features and
   per-asset smoothing halflives per Tables 2–3; residual gaps documented there:
   Yahoo-vs-Bloomberg data ≈ −0.14 Sharpe, λ-grid sensitivity). **The paper never
   benchmarks against MA200 or any trend rule** — only B&H, JM-only and EWMA-μ.
   "Better results" in the paper means better than buy-and-hold, gross of taxes,
   at daily rebalancing (portfolio turnover 2.1–11.7×/yr, refcard Table 6). Losing
   to MA200 does not contradict the paper — the authors never ran that comparison.
2. **The paper's alpha lives at an operating point we can't trade.** Headline
   Sharpe 1.12 = MinVar-MVO with regime-conditioned μ, daily rebalancing, 5 bps,
   no taxes. Under PFU 31.4% per realized gain + ~quarterly cadence, fast regime
   switching is exactly what gets destroyed. The paper's own Table 7: JM-XGB
   next-day return-forecast correlation is 2.4% — the signal's content is
   vol-regime classification (risk control), not direction. Matches the vendor's
   own experiment log ("wins in crises, loses in bull markets"; multi-asset
   replication 7/11 wins vs B&H against the paper's claimed 11/12).
3. **xgb-hrp deviates from the paper's portfolio construction by design**
   (selection + HRP + quarterly drift-band executor + taxes, instead of daily MVO
   with regime-μ) — sensible, but "paper results" were never the right yardstick
   for this pipeline either.
4. **Dead tuning knobs.** PipelineConfig declares `jm_lookback_years`,
   `jm_n_states`, `jm_max_iter`, `jm_tol`, `jm_n_init`, `xgb_max_depth`,
   `xgb_learning_rate`, `xgb_n_estimators`, `xgb_smoothing_halflife`, but
   `pipeline/_walk_forward.py` forwards only `jm_lambda_grid` + transaction cost
   to the vendor call — the rest silently do nothing. Wire through or remove.
5. **Benchmark nit:** etf-pool `benchmark_bh` is `^GSPC` (price index,
   ex-dividends) vs a total-return strategy NAV — `beats_bh` flattered ~2pp/yr.
   Prefer `^SP500TR`.
6. Executor grants every rule same-close execution (signal at t's close, trade at
   t's close, returns accrue from t+1) — apples-to-apples across rules.

**Shipped (this commit):**
- `ma200` and `hybrid` (bear = price < SMA(`ma_window`) OR smoothed p_bear >
  `hybrid_bear_threshold`; defaults 200 / 0.80) as `forecast_method` options;
  price panel threaded selector→apply_rule→executor/CLI/UI/cockpit; new config
  fields `ma_window`, `hybrid_bear_threshold`; SMA warm-up defaults to bull
  (cockpit convention). Step 1 quick win: `bull_prob_threshold` default
  0.60 → 0.40. Tests: 124 passing.
- `scripts/compare_rules.py` — one command → the step-2 decisive table
  (net CAGR / Sharpe / Sortino / MDD / Calmar / turnover / tax+tc drag /
  trades/yr / risk-off events) for production θ=0.60, θ=0.40, ma200, hybrid
  + gross B&H and 60/40 rows; `--no-risk-monitor` ablation flag.

**Step-2 decisive run still pending** — the remote session's network policy blocks
all market-data hosts. Run locally (or allow `query1.finance.yahoo.com`,
`query2.finance.yahoo.com`, `fc.yahoo.com`, `fred.stlouisfed.org` in the
environment's egress settings):

```bash
python run_pipeline.py --phase all --pool etf --start-date 2007-01-01
python scripts/compare_rules.py --pool etf --start-date 2007-01-01
```

Hypotheses to check against the table: quarterly gating will cut MA200's
per-asset edge (the 2026-07-07 shootout assumed daily flips), while the daily
risk monitor (XGB p_bear breadth) still covers crash exits between rebalances —
so `hybrid`/`ma200` + monitor should keep most of the MDD win at ≤ ~quarterly
cadence; verify the trades/yr column against the once-per-quarter goal.

**Step-3 root-cause fix (designed, not implemented — vendor-repo work):** the XGB
target is the JM vol-regime label, so the model is trained to answer the wrong
question for a long/flat trader. In Nakamoto911/xgboost: add price-location
features (distance-to-MA200, 12-month momentum) to the XGB feature set
(`run_period_forecast`, main.py:463-500) and/or switch the training target to a
directional/trend label. Optional after hybrid; the 324-config grid result says
no amount of threshold tuning fixes it.

---

## 2026-07-07 — Production regime rule loses to MA200 (per-asset, net)

**Setup.** ETF pool, 2007-01-01 → 2026-07-06, warm caches. Each detector
simulated as a *tradable long/flat strategy* per asset: hold the asset when the
detector says bull, earn the daily risk-free rate when it says bear, charge
`transaction_cost_bps` (5 bps) per flip and `pfu_rate` (31.4%) tax on realized
gains at each exit. This is the "Detector shootout" table in the Regime Cockpit
(commit `fa897d6`). Analysis scripts (session scratchpad, not in repo):
`prod_vs_ma200.py`, `hybrid_test.py`.

**Result — equal-weight net CAGR across the 12 assets:**

| Detector | Net CAGR (EW) | Sortino | Avg MDD | Beats MA200 |
|---|---|---|---|---|
| Production (θ=0.40 on smoothed p_bear) | 3.7% | 0.36 | −33% | 0/12 |
| Best-tuned production family (θ=0.60, min-len 10) | 4.3% | 0.45 | −34% | **0/12** |
| Buy & hold | 5.2% | 0.48 | −48% | 0/12 |
| Hybrid: `price < MA200 OR p_bear > 0.80` | 7.6% | 0.92 | **−16.4%** | 1/12 |
| **MA200 alone** | **8.8%** | **1.06** | −17.8% | — |

Production loses to MA200 on **every one of the 12 assets**. A 324-config grid
over the whole production rule family (θ × θ_clear × dwell × min-length) never
beats MA200 on a single asset — so the gap is **structural, not a tuning
problem**.

**Why.** The p_bear signal classifies *volatility regimes*; a long/flat
strategy is paid for *price direction*. Double-sided losses vs MA200 on IVV:
- 338 days out while MA200 was in → +124% cumulative upside missed. Recurs
  almost every year (sits out vol-heavy rebounds: 41 days into the 2012
  recovery, 32 into the 2020 rebound, big misses again in 2023).
- 181 days in while MA200 was out → −49% absorbed (Dec-2018, Aug-2011,
  Jan-2016, Dec-2022 selloffs the trend rule had already left).
- Only **14%** of IVV bear calls saw price actually fall.

Exit *speed* is fine (production 15d median crash lag ≈ MA200's 13d) — the
problem is being defensive during up-markets, not being slow to crashes.

**Next steps (agreed, in order):**
1. Quick win (does **not** close the gap): sidebar bull-prob threshold
   0.60 → 0.40 (θ_bear 0.40 → 0.60). +0.6pp EW net CAGR, stable in both
   history halves.
2. **Decisive test:** add `ma200` and the hybrid rule (`price < MA200 OR
   p_bear > 0.80`) as `forecast_method` options in `pipeline/forecast.py`
   (`apply_rule`), then run the **full portfolio backtest** (allocator +
   executor + taxes). The per-asset verdict may change at the portfolio level,
   where p_bear also drives the risk monitor and allocation.
3. **Root-cause fix:** the training target. JM labels reward vol-regime
   separation, not tradable timing. Add price-location features (distance to
   200d MA, 12-month momentum) to the XGB, or move to a directional label.

**Caveats.** Single historical sample; MA200 window is the canonical default,
not tuned; a per-asset long/flat arena inherently favors price-level rules; the
cockpit's MA200 "bear win rate" (97%) is partly a convention artifact (the
switch-day return is assigned to the new regime, which flatters price-triggered
rules).
