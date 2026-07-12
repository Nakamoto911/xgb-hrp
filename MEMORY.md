# Project research log — xgb-hrp

Durable findings and decisions that aren't recoverable from the code or git
history. Newest first. Keep entries short; link to the scripts/commits that
produced them.

---

## 2026-07-12 — Execution-grid sweep: EW + wide drift band lifts hybrid to ~60/40 net; hash-order determinism bug found & fixed

**Setup.** New `scripts/sweep_rules.py` (parallel driver around `compare_rules.compare`),
ETF pool, 2007-01-01 → 2026-07-12, net of PFU 31.4% + 5 bps. Grid: allocator {ew, hrp} ×
freq {monthly, quarterly} × drift {0.015, 0.05, 0.10} × monitor {off, rare-fire
(smoothed, θ_bear=0.85, breadth=0.60)} = 24 configs × 4 rules; then targeted probes on
the winning axis (quarterly/semi-annually/yearly × drift 0.10–0.30, EW, monitor off).
Full tables: `cache/sweep_rules_etf_*.{csv,md}`. Target: net CAGR ≥ 6.57% (60/40 gross)
with MDD ≤ −25% and ≤ ~quarterly trading.

**Determinism bug (found by this sweep, fixed this commit).** Backtest results depended
on `PYTHONHASHSEED`: `Executor` built its trade plan by iterating a `set` of symbols,
and buys are cash-constrained, so hash order decided which symbols got short-changed
when cash was tight. Same config gave hybrid net CAGR 5.32–6.64% (MDD −26.5% to −32.4%)
across runs. Fix: `sorted(universe)` in the plan loop (`pipeline/executor.py`);
two unseeded runs now byte-identical at two configs; 125 tests pass. **All earlier
portfolio-level point values (incl. today's step-2 decisive tables below) are single
draws under this bug — orderings stand (gaps are multi-pp), point values ±~0.6pp.**
Corollary worth exploiting later: fill order moves CAGR by >1pp → the sequential
cash-constrained fill itself is a lever (pro-rata scaling would remove the artifact).

**Sweep verdicts (deterministic re-run, every ordering unanimous across the grid):**

1. **hybrid > ma200 ≫ production at all 24 configs** — selection-rule verdict now
   confirmed at portfolio level for every allocator/cadence/band/monitor combination.
   Production keeps −57% MDD everywhere without the old whipsaw monitor: it provides
   no crash protection at any operating point.
2. **EW > HRP everywhere** (~+1.4pp CAGR on like configs) — kills the bond-tilt drag
   documented 2026-07-12; HRP's only virtue is lower vol (~6.5% vs ~9%).
3. **Rare-fire monitor is strictly harmful**: 8 liquidation events cost 1.2–3.9pp CAGR
   and *worsen* MDD at the wide-band configs (trend selection already exits crashes;
   the monitor only adds tax realizations). At quarterly/0.20: hybrid 6.64% → 2.71%,
   MDD −26.5% → −30.7%. Monitor OFF is the right default for trend rules.
4. **Wider drift band ≫ everything else** — the tax-realization channel dominates:
   within the grid, best config = ew/quarterly/0.10/off/hybrid 4.44% (−25.4% MDD,
   10.3 trades/yr). Tax-free ablation (pfu=0, same config): hybrid 6.02% → taxes cost
   ~1.6pp even at drift 0.10.

**Probe results on the winning axis (EW, monitor off, deterministic):**

| Freq | Drift | Rule | Net CAGR | MDD | Sharpe | Turnover | Tax drag | Trades/yr |
|---|---|---|---|---|---|---|---|---|
| quarterly | 0.20 | hybrid | **+6.64%** | −26.5% | 0.76 | 50%/yr | 19.6% | **2.4** |
| semi-ann. | 0.20 | hybrid | +6.01% | −27.8% | 0.74 | 42%/yr | 31.1% | 1.9 |
| quarterly | 0.25 | ma200 | +5.34% | −25.3% | 0.76 | 40%/yr | **8.5%** | **1.3** |
| quarterly | 0.25 | hybrid | +5.33% | −25.4% | 0.71 | 76%/yr | 40.2% | 2.6 |
| quarterly | 0.15 | hybrid | +4.97% | −26.3% | 0.67 | 113%/yr | 56.0% | 4.4 |
| quarterly | 0.30 | ma200 | +4.64% | **−21.0%** | 0.82 | 58%/yr | 11.5% | 1.5 |
| yearly | 0.20 | hybrid | +3.46% | **−14.2%** | 0.73 | 47%/yr | 18.1% | 1.3 |

(yearly anchors break ma200 — crash exits wait for drift events: −51% MDD.)

**Verdict vs target: close, not closed.** quarterly/0.20/hybrid hits CAGR (6.64% ≥
6.57%) and cadence (2.4 trades/yr) but misses MDD by 1.5pp (−26.5% vs −25%); nothing
satisfies all three. Caveats: the drift ridge is peaked, not a plateau (0.15/0.25
neighbors sit 1.3–1.7pp lower — anchor-alignment path dependence, single historical
sample); and the pre-fix spread shows ±0.6pp of pure fill-order sensitivity at this
operating point. Treat "hybrid ≈ 5–6.6% net, MDD ≈ −25%" as the honest read.
ma200 @ quarterly/0.25 is the low-tax simple alternative (5.34%, 8.5% tax drag,
1.3 trades/yr).

**Subperiod split (2007–2016 / 2017–2026, NAV sliced from the full-window run —
same session, deterministic):**

| Variant | H1 CAGR | H1 MDD | H2 CAGR | H2 MDD |
|---|---|---|---|---|
| hybrid drift 0.15 | +5.39% | −17.8% | +4.65% | −26.3% |
| hybrid drift 0.20 | +4.93% | −26.5% | +8.44% | −26.0% |
| hybrid drift 0.25 | +5.79% | −21.9% | +5.19% | −25.4% |
| ma200 drift 0.25 | +2.95% | −13.7% | +7.88% | −25.3% |
| 60/40 gross (VSMGX) | +4.69% | −41.1% | +8.54% | −22.4% |

Reads: (a) hybrid tracks the gross 60/40 CAGR within ±0.3pp *net of tax* in **both**
halves (beats it in H1) — consistent, not one-era luck; (b) but the drift-ridge
location flips (H1 best: 0.25; H2 best: 0.20), so the full-window 6.64% peak at 0.20
is partly parameter luck — the robust statement is "hybrid ≈ 5–6.5% net across drift
0.15–0.25"; (c) the durable edge over 60/40 is drawdown control in H1-type regimes
(−26% vs −41% in 2008) plus the tax-policy headroom, not raw CAGR; in H2 60/40 had
the shallower MDD; (d) ma200 is far more era-dependent (2.95% H1) — hybrid's p_bear
backstop earns its keep exactly there.

**Next steps (in order):**
1. Tax-aware execution per SPEC §16 (rebalance-on-flip-only, never sell winners to
   re-band): tax drag at the peak is still 19.6% cumulative, and the pfu=0 ablation
   says ~+1.3–1.8pp CAGR headroom → would clear 60/40 with margin.
2. Pro-rata (not sequential) cash-constrained fills — removes the >1pp fill-order
   artifact the determinism bug exposed.
3. Step-3 root-cause fix (vendor XGB directional target) unchanged — hybrid's edge
   over ma200 (+1.3pp at the peak, +2.4pp in 2007–2016) is the p_bear backstop
   earning its keep, so a better-trained signal compounds it.
4. Pick production drift by robustness, not the peak: 0.20–0.25 band, decide after
   the tax-aware executor lands (it changes the turnover/tax geometry).

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
