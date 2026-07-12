# End-to-End Quant Investment Pipeline — Specification
**Status:** Draft v1.1
**Author:** Paul (spec drafted with Claude)
**Source repos:**
- `xgboost/` (JM-XGB regime forecasting) — https://github.com/Nakamoto911/xgboost
- `hrp/` (HRP portfolio engine) — https://github.com/Nakamoto911/hrp
**Changelog:**
- **v1.1** — Fixed re-entry drag (Section 10.2): default `reenter_mode` switched from `next_rebalance` to `immediate_fresh` to capture V-shape recoveries. Made AVCO cost-basis accounting explicit and mandatory (Section 9.3.1) — French PFU / CTO law requires AVCO; FIFO is not a valid alternative for the target jurisdiction. Added re-entry alpha capture eval check.
- **v1.0** — Initial draft.
---
## 1. Purpose and Scope
This document specifies an end-to-end quantitative investment pipeline that fuses the two existing engines into a single, daily-running system. The pipeline:
1. Selects an asset universe (one of three pre-defined pools).
2. Builds per-asset regime histories using the Statistical Jump Model (JM) from the `xgboost` repo.
3. Forecasts each asset's next-period regime + probability using an XGBoost classifier (one model per asset). The forecast method is user-selectable.
4. Filters the universe to assets meeting a minimum bull-regime probability threshold.
5. Allocates capital across the filtered set using HRP by default (with EW / momentum / min-vol alternatives).
6. Executes trades using drift-band rebalancing from the `hrp` repo, with the rebalance frequency user-configurable (quarterly default).
7. Computes performance and compares against benchmarks (S&P B&H and a Vanguard 60/40 lifestyle ETF).
8. Runs an independent **Risk Monitor** daily that can force a risk-off (100% risk-free asset) between scheduled rebalances when the universe's aggregate bear probability crosses a threshold.
An evaluation module is specified for every pipeline step and for the workflow end-to-end (Section 10).
The pipeline must be optimized for speed at every layer (Section 11).
**What this spec is not:** It is not a re-implementation plan for JM or HRP internals — both are inherited from the existing repos with their parameter defaults preserved. It is a specification for the new orchestration layer, the per-asset forecast pipeline, the risk monitor, and the evaluation framework.
---
## 2. High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                       USER (Streamlit UI)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Asset Pool Selector  → {ETF, Mutual Fund, EUR Investable}    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Data Layer (cached, parallel fetch per asset)                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Regime Chart Builder (per-asset JM, parallel)                │
│    Output: regime label series + JM params per asset            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Forecast Engine (per-asset XGBoost, parallel)                │
│    Output: (P_bull_t+1, P_bear_t+1, smoothed_prob, regime)      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
┌──────────────────────┐         ┌────────────────────────────────┐
│ 5a. Asset Selector   │         │ 8. RISK MONITOR (daily, async) │
│  (bull prob filter)  │         │  Universe-level bear scan      │
└──────────┬───────────┘         │  Triggers risk-off override    │
           │                     └────────────────┬───────────────┘
           ▼                                      │
┌──────────────────────────┐                      │
│ 6. Allocator             │                      │
│  HRP / EW / Mom / MinVol │                      │
└──────────┬───────────────┘                      │
           ▼                                      │
┌──────────────────────────┐                      │
│ 7. Drift-Band Executor   │◀─────────────────────┘
│  (rebalance freq + risk-off override)           │
└──────────┬──────────────────────────────────────┘
           ▼
┌──────────────────────────┐
│ 9. Performance & Bench   │  (S&P B&H, Vanguard 60/40)
└──────────────────────────┘
(Eval module hooks into every box — Section 10)
```
---
## 3. Module 1 — Asset Pool Selector
### 3.1 Pools (matching `hrp/` repo)
| Pool key      | Source        | Universe size | Benchmark (B&H)                | Risk-free leg          |
|---------------|---------------|---------------|--------------------------------|------------------------|
| `etf`         | yfinance      | US ETF list   | `^GSPC` (S&P 500 TR)           | `BIL` (SHV/BIL/SHY)    |
| `mutual_fund` | yfinance      | US MF list    | `^SP500TR`                     | `VBMFX` short-dur sub  |
| `european`    | yfinance Xetra| EUR investable| `^STOXX` or pool-local index   | EUR T-bill ETF         |
Exact ticker lists are inherited from `hrp/hrp_engine/data.py` and `xgboost/misc_scripts/asset_lists.md`. New pools can be added by appending a section to a config file — same convention as `asset_lists.md`.
### 3.2 Risk-free destination (per user decision)
Single risk-free asset, configurable per pool. Defaults: `BIL` for ETF/MF pools, EUR equivalent for European pool.
### 3.3 Benchmarks
Two benchmarks are run alongside every strategy backtest:
- **S&P Buy & Hold:** `^GSPC` total-return, no rebalancing, no costs.
- **Vanguard 60/40 Lifestyle:** `VSMGX` (Vanguard LifeStrategy Moderate Growth, 60% equity / 40% bond), or the equivalent UCITS/EUR variant for the European pool.
Both are loaded through the same data layer and subjected to the same cost model (zero for B&H, none for Vanguard since it's a single ETF).
---
## 4. Module 2 — Data Layer
Inherited from `hrp/hrp_engine/data.py` and `xgboost/main.py` caching logic. Three additions for this pipeline:
1. **Unified cache root.** One `cache/` directory shared across the pipeline. Cache keys: `{pool}_{ticker}_{start}_{end}_{feature_version}.parquet`. Parquet replaces pickle for cross-process safety and ~3× smaller files.
2. **Parallel fetch.** `asyncio.gather` over all tickers in the pool (yfinance HTTP is I/O-bound). Fall back to ThreadPoolExecutor where async APIs are unavailable.
3. **Auxiliary series.** FRED series (`DGS2`, `DGS10`, `^VIX`, `^IRX`) loaded once per pipeline run, shared across all per-asset models — these are macro features, not per-asset features.
**Data cleaning** (from `hrp` repo): flat startup segment trimming + 3-day rolling median filter for bad ticks. Applied identically here.
**Feature engineering** (from `xgboost/main.py`):
- Return features (per asset): `DD_log_5/21`, `Avg_Ret_5/10/21`, `Sortino_5/10/21`. EWMA-based with per-asset-class half-life table (`PAPER_EWMA_HL`).
- Macro features (shared): `Yield_2Y_EWMA_diff`, `Yield_Slope_EWMA_10`, `Yield_Slope_EWMA_diff_21`, `VIX_EWMA_log_diff`, `Stock_Bond_Corr` (252-day rolling).
---
## 5. Module 3 — Per-Asset Regime Chart Builder
One JM model per asset (per user decision, matches `xgboost/misc_scripts/benchmark_assets.py`).
### 5.1 JM configuration (inherited)
| Param                 | Value                                                |
|-----------------------|------------------------------------------------------|
| States                | 2 (Bull = low-vol, Bear = high-vol)                  |
| Optimization          | K-means++ init `n_init=10` → Viterbi                |
| `max_iter` / `tol`    | 1000 / 1e-8                                          |
| Jump penalty λ        | Tuned per chunk on 5y rolling validation (Sharpe)   |
| λ grid                | `[4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0]`|
| EWMA half-life        | `PAPER_EWMA_HL` table (per asset class)             |
| Online prediction     | `predict_online` (forward-only Viterbi, causal)     |
### 5.2 Output
Per asset, for the OOS window:
- `regime_label[t]` ∈ {Bull, Bear}
- `jm_means[t]` (in-sample regime means, used downstream by some allocators)
- `lambda[chunk]` (selected jump penalty per 6-month chunk)
Stored as a single `regimes_{pool}_{date}.parquet` keyed by (ticker, date).
### 5.3 Parallelization
JM fitting is independent across assets. Use `joblib.Parallel(n_jobs=-1, backend="loky")`. Empirically (from `benchmark_assets.py`): ~0.6s/asset/chunk on M-series. 12 assets × ~34 chunks × 0.6s = ~245s single-threaded → ~30s on 8 cores.
---
## 6. Module 4 — Per-Asset Regime Forecast Engine
One XGBoost classifier per asset, trained on (features → next-day regime label from JM). Configuration inherited from `xgboost` repo paper defaults.
### 6.1 XGBoost configuration (inherited)
| Param          | Value                                              |
|----------------|----------------------------------------------------|
| Lookback       | 11 years rolling                                   |
| `max_depth`    | 6                                                  |
| `learning_rate`| 0.3                                                |
| `n_estimators` | 100                                                |
| Regularization | None (paper defaults)                              |
| Features       | JM one-hot regime + return features + macro features|
### 6.2 Forecast outputs (per asset, per day)
- `P_bull[t+1]`, `P_bear[t+1]` — raw next-day class probabilities
- `P_bull_smoothed[t+1]` — EWMA-smoothed probability (paper-style)
- `regime_forecast[t+1]` — argmax label, with paper-style +1-day shift to avoid look-ahead
### 6.3 User-selectable forecast methods (Asset Selection Rule)
Per user decision: the user picks the rule applied in Module 5. All rules run from the same XGBoost outputs above; the rule is a thin selector layer.
| Rule key             | Definition                                                    | Threshold default |
|----------------------|---------------------------------------------------------------|-------------------|
| `prob_threshold`     | `P_bull[t+1] ≥ θ`                                             | shared `bull_prob_threshold`, default 0.40 |
| `regime_and_prob`    | `regime_forecast = Bull AND P_bull ≥ θ`                       | shared `bull_prob_threshold`, default 0.40 |
| `ewma_smoothed`      | `P_bull_smoothed[t+1] ≥ θ` (matches `xgboost` repo signal)    | shared `bull_prob_threshold`, default 0.40 |
| `trend`              | `P_bull` rising over last N days (slope > 0, OLS on window)   | N = 5, slope > 0  |
| `last_day_regime`    | Simple: `regime_forecast[t+1] == Bull`                        | n/a               |
| `ma200`               | Price-only: `price ≥ SMA(price, N)`                          | N = 200           |
| `hybrid`              | `price ≥ SMA(price, N) AND P_bear_smoothed ≤ θ_bear` (i.e. bear when price < MA200 OR smoothed p_bear > θ_bear) | N = 200, θ_bear = 0.80 |
All rules share the EWMA smoothing pipeline (half-life from `xgboost` config) so they're commensurable. `ma200` and `hybrid` are price-aware — they additionally require the pool's price panel, not just the forecast panel.
### 6.4 Parallelization
Per-asset XGBoost training is embarrassingly parallel. `joblib.Parallel(n_jobs=-1)`. XGBoost itself uses internal threads — explicitly set `n_jobs=1` *inside* each model and parallelize at the outer (per-asset) level to avoid thread oversubscription.
---
## 7. Module 5 — Asset Selector
Takes the per-asset forecast output from Module 4 and emits the **selected list** for the upcoming rebalance period.
### 7.1 Logic
```python
def select_assets(forecasts: dict[str, ForecastRow], rule: SelectionRule, theta: float) -> list[str]:
    return [t for t, f in forecasts.items() if rule.passes(f, theta)]
```
### 7.2 Edge cases
- **Empty selection:** if 0 assets pass, route 100% to risk-free. Log the event.
- **Singleton selection:** allocator degrades to 100% in the single asset (HRP, EW, momentum, min-vol all collapse to 1.0 weight gracefully).
- **Selection diff:** compute Δ(selected) vs previous rebalance for eval & turnover analytics.
---
## 8. Module 6 — Portfolio Allocator
Default: **HRP** from `hrp/hrp_engine/hrp.py`, with all `hrp` repo features preserved (RMT denoising, tree-based bisection, correlation-to-distance mapping).
### 8.1 Allocator menu
| Allocator      | Description                                                              | Source                       |
|----------------|--------------------------------------------------------------------------|------------------------------|
| `hrp` (default)| Hierarchical Risk Parity, RMT-denoised cov, tree bisection               | `hrp/hrp_engine/hrp.py`      |
| `ew`           | Equal weight across selected assets                                      | trivial                      |
| `momentum_30d` | Weights proportional to trailing 30-day total return (negative clipped) | new — see 8.2                |
| `min_vol_30d`  | Weights proportional to `1/σ_30d` (inverse trailing 30-day vol)         | new — see 8.2                |
### 8.2 Momentum & min-vol specifications
**Momentum 30d:**
- `r_i = total return over last 30 trading days`
- `r_i_clipped = max(r_i, 0)` — assets with negative momentum get 0 weight
- `w_i = r_i_clipped / Σ r_j_clipped`
- If all clipped to 0 → fall back to EW on the selected set.
**Min-vol 30d:**
- `σ_i = std of daily log returns over last 30 trading days`
- `w_i = (1/σ_i) / Σ (1/σ_j)`
- σ floored at 1e-6 to prevent blowup.
Both use the same data already on disk — no extra fetches.
### 8.3 HRP configuration (inherited from `hrp` repo defaults)
- Lookback: 4 years
- Linkage: `single` (configurable: `single`, `complete`, `ward`)
- Bisection: `tree` (configurable: `tree`, `index`)
- Denoising: Marchenko-Pastur, eigenvalues below `λ_max = σ²(1 + √(N/T))²` set to mean, trace preserved
---
## 9. Module 7 — Drift-Band Executor & Rebalance Loop
Inherited from `hrp/hrp_engine/backtest.py`.
### 9.1 Rebalance frequency
User-configurable. Default: **quarterly**. Options: `daily`, `weekly`, `monthly`, `quarterly`, `semi-annually`, `yearly`.
### 9.2 Drift-band logic
At each rebalance date:
1. Allocator produces target weights `w_target` over the selected set.
2. Current portfolio weights `w_current` are measured (after market drift since last execution).
3. For each asset: if `|w_target_i - w_current_i| < drift_threshold`, do not trade. Else trade.
4. Drift threshold default: **0.015 (1.5%)**, inherited from `hrp` repo.
### 9.3 Costs
| Cost              | Default                            | Source              |
|-------------------|------------------------------------|---------------------|
| Transaction cost  | 5 bps on buy/sell notional        | `hrp` repo          |
| Capital gains tax | 31.4% PFU flat, AVCO cost basis with loss carryforward | `hrp` repo          |
| Slippage          | None modeled (assume MOC execution)| n/a                 |
#### 9.3.1 Cost-basis accounting method — AVCO (mandatory)
The executor maintains an internal ledger tracking the **average purchase price per asset (AVCO — Average Cost / Prix Moyen Pondéré d'Acquisition)**. This is not a modeling choice: French tax law requires AVCO for securities held in a CTO (compte-titres ordinaire) under the PFU regime. FIFO is not a valid alternative for the target jurisdiction.
**Ledger semantics, per asset:**
- `units_held[t]` — total units currently owned
- `avg_cost[t]` — weighted average purchase price across all open units
**On a buy of `Δu` units at price `p`:**
```
new_units    = units_held + Δu
new_avg_cost = (units_held * avg_cost + Δu * p) / new_units
units_held   = new_units
avg_cost     = new_avg_cost
# Transaction costs are added to cost basis (French convention: frais d'acquisition
# included in PMP), not expensed separately for tax purposes.
```
**On a sell of `Δu` units at price `p`:**
```
realized_gain_per_unit = p - avg_cost
realized_gain          = Δu * realized_gain_per_unit       # gross, before TC
units_held             = units_held - Δu
# avg_cost is UNCHANGED on a partial sale (this is the defining property of AVCO)
# A full sale (units_held → 0) resets avg_cost = NaN until next buy.
```
**Tax computation at year-end (or at the end of the backtest window for terminal accounting):**
```
net_realized_gain   = Σ realized_gains_year - Σ realized_losses_year
                      + carryforward_loss_balance
tax_due             = max(0, net_realized_gain) * pfu_rate     # 0.314 default
new_carryforward    = min(0, net_realized_gain)                # negative number,
                                                                # 10-year French rule
                                                                # (configurable horizon)
```
**Loss carryforward horizon.** French law caps the carryforward at 10 years for capital losses on a CTO. The executor enforces this by aging losses by vintage year. Default carryforward window is 10 years; tunable for jurisdictions with different rules.
**Mark-to-market vs realized.** Unrealized P&L is tracked for NAV but never enters the tax computation. Only realized events (sells, including risk-off liquidations and drift-band rebalances) crystallize taxable gains.
**Why this matters for the eval module.** The executor eval check `cost_basis_tracking_matches_independent_recompute` (Section 12.6) re-walks the entire trade ledger from scratch using an independent AVCO implementation and asserts the running `avg_cost` series matches within 1e-6. This catches off-by-one and order-of-operations bugs in the live ledger, which would otherwise silently corrupt tax drag metrics.
### 9.4 Risk-off override (interaction with Module 8)
The executor accepts a `force_risk_off` flag from the Risk Monitor. When raised:
1. **Immediately** (not waiting for next scheduled rebalance) liquidate all positions.
2. Move 100% to the pool's risk-free asset.
3. Apply transaction costs and tax accounting normally.
4. Resume scheduled rebalances; risk-off persists until the Risk Monitor clears it (Section 10).
---
## 10. Module 8 — Risk Monitor (daily)
This is the **new** module, not present in either source repo. Runs **every trading day**, independent of the rebalance frequency.
### 10.1 Trigger logic (per user decision)
```
trigger_risk_off = (count(assets where P_bear[t] > bear_prob_threshold) / N) > universe_pct_threshold
```
Defaults:
- `bear_prob_threshold = 0.70` (an asset is in "high bear" if `P_bear` > 0.70)
- `universe_pct_threshold = 0.40` (trigger if >40% of universe is in high bear)
Both thresholds are tunable.
### 10.2 Clearance logic (symmetric, hysteresis)
To avoid flip-flopping in/out of risk-off, clearance uses a **lower** threshold and a **dwell time**:
```
clear_risk_off = (count(assets where P_bear[t] > bear_prob_threshold) / N) < universe_pct_clear_threshold
                 AND consecutive_clear_days ≥ dwell_days
```
Defaults:
- `universe_pct_clear_threshold = 0.25` (clear when <25% of universe in high bear)
- `dwell_days = 5` (must hold below clear threshold for 5 consecutive days)
#### 10.2.1 Re-entry (default: immediate, with fresh allocation)
On clearance, the strategy re-enters **immediately** — *not* at the next scheduled rebalance. Waiting for the next rebalance with a quarterly cycle would forfeit 6–10 weeks of recovery, which is the exact pathology a fast risk-off trigger is supposed to avoid. The `dwell_days` hysteresis above is the whipsaw guard; making re-entry symmetric with the slow rebalance cadence would defeat the purpose of having a fast monitor in the first place.
The re-entry sub-mode controls *how* the new allocation is computed:
| Re-entry mode (`reenter_mode`) | Behavior on clearance day                                          | When to use                                      |
|--------------------------------|--------------------------------------------------------------------|--------------------------------------------------|
| `immediate_fresh` (default)    | Run Selector + Allocator on clearance-day data, allocate to new targets | Production. Reflects post-crash regime forecasts. |
| `immediate_last_targets`       | Allocate to the last target weights computed before risk-off       | Faster (skips allocator); use when forecasts are unreliable mid-crash. |
| `next_rebalance`               | Stay in risk-free until next scheduled rebalance                   | Stress testing / ablation only. **Not recommended for production.** |
**Cost of `immediate_fresh`:** one extra Allocator + Selector run on clearance day. Per Section 13 budgets: ~20s XGBoost (per-asset, but only inference, not retraining — JM/XGB models are still those from the most recent walk-forward chunk) + ~0.2s HRP = ~20s total. Negligible compared to the recovery alpha being captured.
**Important nuance on forecasts at clearance.** The selector and allocator at clearance use the **most recent walk-forward models** (the JM/XGB chunk active at the time the monitor cleared), not stale pre-crash models. Because the JM/XGB walk-forward chunks are 6 months long, the active chunk is likely to span the crash itself, which means the regime forecasts have been continuously updating through the drawdown. This is one of the reasons `immediate_fresh` is the default — the model has already absorbed the new regime evidence by the time the monitor clears.
**Post-clearance rebalance alignment.** After an immediate re-entry, the next scheduled rebalance reverts to the original schedule (e.g. if the original schedule was end-of-quarter and clearance happened mid-February with quarterly cadence, the next scheduled rebalance is still end-of-March). This keeps the long-run cadence stable.
### 10.3 Operational notes
- Risk Monitor reads from the same forecast cache as Module 5; no double inference.
- Logs every state transition (`risk_on → risk_off`, `risk_off → risk_on`) with timestamp, trigger metrics, and asset-level bear probabilities at the moment of trigger. This audit trail is critical for the eval module.
### 10.4 Backtest semantics
In backtest mode, the Risk Monitor runs at every historical OOS day. In live mode, it runs once per day after market close (when JM/XGBoost forecasts refresh).
---
## 11. Module 9 — Performance & Benchmarks
### 11.1 Metrics (per `xgboost` repo conventions)
- Cumulative return
- Annualized volatility
- Sharpe (rf = `^IRX` daily)
- Sortino (downside vol)
- Max drawdown
- Turnover (annual, dollar-weighted)
- Tax drag (realized tax / start-of-year NAV)
- Calmar
- Hit rate (% positive months)
### 11.2 Benchmarks
- **S&P B&H** (`^GSPC` total return, no costs)
- **Vanguard 60/40** (`VSMGX` or EUR-equivalent for European pool)
### 11.3 Sub-period analysis (inherited from `xgboost`)
Auto-decomposition into: GFC (2007–2009), Recovery (2010–2014), Late Cycle (2015–2019), COVID (2020), Post-COVID (2021–).
### 11.4 Output
- `report_{pool}_{date}.md` — primary deliverable
- `equity_curve_{pool}_{date}.parquet` — daily NAV
- `trades_{pool}_{date}.parquet` — every trade with side, qty, cost, tax
- `risk_events_{pool}_{date}.parquet` — every Risk Monitor transition
---
## 12. Evaluation Module (per-step + end-to-end)
Each pipeline step has its own eval. The eval module is the **fastest way to catch silent regressions** — a step can produce numerically plausible outputs and still be subtly broken. Each eval is a function returning a structured dict; results are aggregated into `eval_report_{pool}_{date}.md`.
### 12.1 Data Layer eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| Cache freshness            | Last bar date ≤ 1 trading day behind market close                     |
| Missing data               | < 0.5% NaN per series after cleaning                                  |
| Outlier detection          | `|daily log return|` > 7σ flagged; rolling-median replacement audit  |
| Calendar alignment         | All assets share the same business-day index after reindex            |
| Stale-streak audit         | No flat-price runs ≥ 5 days in liquid assets                          |
### 12.2 JM Regime eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| State separation           | Mean return in Bull > Bear, vol in Bear > Bull (sign check per asset) |
| Silhouette score           | > 0.15 on z-scored feature space (paper-relative)                     |
| Davies-Bouldin             | < 2.0                                                                 |
| Regime persistence         | Median regime duration > 20 trading days (regime-switching not noise) |
| Online vs batch agreement  | `predict_online` matches batch JM ≥ 99% on training tail              |
| λ stability                | CV(λ) across chunks < 0.50                                            |
### 12.3 XGBoost Forecast eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| OOS accuracy               | > 0.55 (binary baseline = 0.50)                                       |
| ROC-AUC                    | > 0.55                                                                |
| Log-loss                   | Better than constant-base-rate predictor                              |
| MCC (Matthews CC)          | > 0.10                                                                |
| Calibration                | Reliability diagram slope ∈ [0.8, 1.2] in deciles                    |
| Rank IC vs forward returns | Mean IC > 0.02, t-stat > 2                                            |
| Information ratio          | Multi-horizon (1d, 5d, 21d) IR > 0.10                                 |
| Feature importance drift   | Top-5 features stable across walk-forward chunks (rank correlation > 0.5) |
### 12.4 Asset Selector eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| Selection size sanity      | Selected count > 0 in ≥ 95% of rebalance dates (else lower θ)        |
| Forward-return signal      | Mean fwd return of selected > mean fwd return of non-selected         |
| Selection turnover         | < 60% of universe churns rebalance-to-rebalance (too high = unstable signal) |
| Rule sensitivity           | All 5 rules backtested in parallel; reported side-by-side             |
### 12.5 Allocator eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| Weight sum                 | `Σ w_i = 1.0 ± 1e-6`                                                  |
| Non-negativity             | `w_i ≥ 0` for all i (long-only)                                       |
| HRP-specific: PSD          | Denoised covariance is PSD                                            |
| HRP-specific: trace        | Trace preserved post-denoising (within 1e-9)                          |
| Concentration              | HHI ≤ 0.4 (else single-asset domination — flag)                       |
| Cross-allocator comparison | HRP / EW / momentum / min-vol all run; Sharpe & DD reported side-by-side |
### 12.6 Executor eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| Drift-band efficacy        | Trades skipped > 0 when threshold > 0 (sanity)                        |
| Turnover monotonicity      | Annual turnover monotonically decreases as drift threshold increases  |
| Cost accounting            | Σ trade costs matches `Σ |Δw_i| × notional × bps` (within rounding)  |
| Tax accounting             | AVCO ledger matches independent recompute on full path; carryforward correctly aged ≤ 10y |
| Risk-off latency           | When trigger fires at day `t`, portfolio is 100% risk-free by day `t+1` close |
| Re-entry latency           | When monitor clears at day `t`, portfolio is reallocated by day `t+1` close (under `immediate_fresh`) |
### 12.7 Risk Monitor eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| Trigger calibration        | In-sample: triggers fire ≥ 80% of drawdowns > 15%; ≤ 1 false alarm/year |
| Trigger lead time          | Median lead time before benchmark hits its DD trough > 5 trading days |
| Clearance correctness      | No re-entry during ongoing crash (e.g. clearance ≥ 30d after 2008 trough) |
| Hysteresis check           | No oscillation: `risk_on→off→on→off` in <20-day window flagged       |
| Re-entry alpha capture     | Ablation: median 60-day return after clearance under `immediate_fresh` ≥ benchmark median 60-day return over same windows. (Catches the quarterly re-entry drag failure mode.) |
| Ablation                   | Strategy run with vs without Risk Monitor — Sharpe and Max DD compared |
### 12.8 End-to-End eval
| Check                      | Pass criterion                                                        |
|----------------------------|-----------------------------------------------------------------------|
| Beats B&H S&P              | Sharpe > S&P Sharpe AND Max DD < S&P Max DD                          |
| Beats Vanguard 60/40       | Sharpe > 60/40 Sharpe                                                |
| Sub-period robustness      | Wins ≥ 4 of 5 sub-periods on Sharpe                                  |
| Look-ahead audit           | Shuffle-OOS test: shuffling future returns ≥ chunk size reduces Sharpe to ≈ 0 |
| Causality audit            | All signals applied to next-period (`t+1`) returns, never same-period |
| Walk-forward stability     | Sharpe std across non-overlapping 3-year windows < 0.5                |
| Deflated Sharpe            | DSR > 0 (Bailey-Lopez de Prado), accounting for selection bias        |
| Pool-portability           | Run on all 3 pools; if only 1 of 3 wins → strategy is pool-specific, flag |
| Allocator ablation         | Pipeline run with each of HRP/EW/momentum/min-vol — HRP should be top-2 |
| Reproducibility            | Same config, same data → bitwise-identical equity curve              |
### 12.9 Eval CI hook
A `--gate` flag in the CLI runs the eval module and exits non-zero if any **critical** check fails (subset marked `critical=True` in the eval registry). Intended use: pre-commit hook before publishing a benchmark report.
---
## 13. Performance & Optimization
Speed is a first-class requirement. Targets, measured on a 12-asset pool with 17 years OOS:
| Stage              | Target latency | Notes                                                            |
|--------------------|----------------|------------------------------------------------------------------|
| Data fetch (cold)  | < 15s          | Parallel yfinance, async                                         |
| Data fetch (warm)  | < 0.5s         | Parquet cache, no network                                        |
| JM fitting         | < 30s          | 12 assets × ~34 chunks, parallel `joblib`                        |
| XGBoost training   | < 20s          | Parallel per-asset, XGB internal `n_jobs=1`                      |
| HRP allocation     | < 0.2s         | Single shot per rebalance, vectorized                            |
| Backtest loop      | < 5s           | Vectorized NAV calc + drift-band logic                           |
| Risk Monitor scan  | < 0.5s/day     | Single pass over forecast cache                                  |
| **Total (cold)**   | **< 75s**      |                                                                  |
| **Total (warm)**   | **< 12s**      |                                                                  |
### 13.1 Parallelization strategy
- **Outer level (per asset):** `joblib.Parallel(n_jobs=-1, backend="loky")`. Loky avoids fork-safety issues with sklearn/numpy on macOS.
- **Inner level:** XGBoost `n_jobs=1`, BLAS thread count = 1 (via `OMP_NUM_THREADS`, `MKL_NUM_THREADS`). Critical — without this, you get O(n_cores²) thread oversubscription and net *slower* execution.
- **Data fetch:** `asyncio` + `aiohttp` where yfinance allows; otherwise `ThreadPoolExecutor` (I/O-bound, GIL-friendly).
- **Eval module:** independent across checks → `concurrent.futures.ThreadPoolExecutor`.
### 13.2 Caching strategy
- **Parquet** for all on-disk caches (3× smaller than pickle, cross-process safe, columnar reads).
- **Cache keys** include `feature_version` hash so a feature-engineering change invalidates only what it should.
- **MVO/HRP results cache** keyed on `(signals_mtime, allocator_params_hash)` — same scheme as `xgboost/portfolio.py`.
- **In-process LRU** (`functools.lru_cache` or `cachetools.LRUCache`) for hot paths (JM `predict_online` per asset).
### 13.3 Algorithmic optimizations
- **Vectorized NAV path:** the backtest loop in `hrp/hrp_engine/backtest.py` should remain vectorized; do not introduce per-day Python loops for the no-trade days. Drift-band is the only event-driven part.
- **Rolling features:** use `pandas.rolling` with `engine="numba"` where the window is large (≥ 30) — measured 4–8× speedup on EWMA-based features.
- **Cov denoising:** cache eigendecomp per (asset_set, lookback) tuple — denoised cov is reused for HRP and min-vol allocators in the same rebalance.
### 13.4 Memory
- Float32 for all return panels (sufficient precision for log returns, halves memory).
- Streaming Parquet reads (`pyarrow.dataset`) for multi-year backtests where the full panel doesn't need to be in RAM.
### 13.5 Profiling discipline
- `py-spy record` on the full pipeline before any optimization PR.
- Benchmark gate: total cold-path runtime regressed by > 20% blocks merge.
---
## 14. Configuration
Single `PipelineConfig` dataclass (pydantic v2 preferred for validation):
```python
@dataclass
class PipelineConfig:
    # Pool
    pool: Literal["etf", "mutual_fund", "european"] = "etf"
    risk_free_asset: str = "BIL"          # overridden by pool default
    # Regime
    jm_lambda_grid: tuple = (4.64, 10.0, 15.0, 21.54, 30.0, 46.42, 70.0, 100.0)
    jm_lookback_years: int = 11
    # Forecast
    forecast_method: Literal["prob_threshold", "regime_and_prob", "ewma_smoothed",
                             "trend", "last_day_regime", "ma200", "hybrid"] = "ewma_smoothed"
    bull_prob_threshold: float = 0.40
    trend_window: int = 5
    ma_window: int = 200               # SMA window (days) for ma200 / hybrid
    hybrid_bear_threshold: float = 0.80  # smoothed p_bear gate for hybrid
    # Allocator
    allocator: Literal["hrp", "ew", "momentum_30d", "min_vol_30d"] = "hrp"
    hrp_lookback_years: int = 4
    hrp_linkage: Literal["single", "complete", "ward"] = "single"
    hrp_bisection: Literal["tree", "index"] = "tree"
    # Execution
    rebalance_frequency: Literal["daily", "weekly", "monthly",
                                 "quarterly", "semi-annually", "yearly"] = "quarterly"
    drift_threshold: float = 0.015
    transaction_cost_bps: float = 5.0
    pfu_rate: float = 0.314
    # Risk Monitor
    risk_monitor_enabled: bool = True
    bear_prob_threshold: float = 0.70
    universe_pct_threshold: float = 0.40
    universe_pct_clear_threshold: float = 0.25
    risk_off_dwell_days: int = 5
    reenter_mode: Literal["immediate_fresh", "immediate_last_targets",
                          "next_rebalance"] = "immediate_fresh"
    # Backtest window
    start_date: str = "2007-01-01"
    end_date: Optional[str] = None        # today if None
```
All defaults match the source repos where applicable. The Streamlit UI binds 1:1 to this dataclass.
---
## 15. Repository Layout (proposed)
```
quant-pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── config.py              # PipelineConfig
│   ├── data.py                # Module 2 — extends hrp/hrp_engine/data.py
│   ├── regime.py              # Module 3 — wraps xgboost JM
│   ├── forecast.py            # Module 4 — wraps xgboost XGB + 5 selection rules
│   ├── selector.py            # Module 5
│   ├── allocator.py           # Module 6 — dispatches to hrp / ew / momentum / min_vol
│   ├── executor.py            # Module 7 — extends hrp backtest with risk-off hook
│   ├── risk_monitor.py        # Module 8 — NEW
│   ├── performance.py         # Module 9
│   └── eval/
│       ├── data_eval.py
│       ├── jm_eval.py
│       ├── xgb_eval.py
│       ├── selector_eval.py
│       ├── allocator_eval.py
│       ├── executor_eval.py
│       ├── risk_eval.py
│       └── e2e_eval.py
├── app.py                     # Unified Streamlit UI
├── run_pipeline.py            # CLI entry (mirrors hrp/run_backtest.py)
├── tests/
│   └── test_*.py
├── cache/                     # gitignored
├── benchmarks/                # gitignored
├── vendor/                    # symlinks/submodules to xgboost & hrp internals
└── SPEC.md                    # this document
```
Vendoring rather than re-implementing: the two existing repos are pulled in as git submodules. The new code is the orchestration + the Risk Monitor + the eval module + a unified UI. This minimizes duplication and keeps the source repos as the canonical home for JM/XGB and HRP internals.
---
## 16. Open Questions & Future Work
These are deliberately out of scope for v1 but flagged for future spec revisions:
1. **Cross-asset forecast features.** Currently per-asset XGBoost; could add a regime-stacking layer that takes other assets' regimes as features.
2. **Multi-asset risk-free.** Currently single risk-free destination; could allow a defensive basket (e.g. 50% bills + 50% gold) on risk-off.
3. **Soft risk-off.** Currently binary; a graded version that scales equity exposure 100% → 0% as universe bear % crosses 25% → 60% may reduce whipsaw cost.
4. **Online learning.** Currently walk-forward with 6-month chunks; incremental XGBoost updates (Booster.update) could shrink retraining cost.
5. **Tax-aware allocation.** Drift-band already reduces tax drag; an allocator that explicitly penalizes realized gains during the optimization step would close the loop.
6. **Live trading adapter.** Spec covers backtest + daily forecast; an IBKR adapter for live execution is the natural next step (CTO account, no PEA restrictions).
---
## 17. Acceptance Criteria for v1
The pipeline is considered v1-complete when:
1. All 3 pools run end-to-end on cold cache in < 90s on a single 8-core machine.
2. All eval gates (Section 10) pass on the ETF pool with default config.
3. Strategy beats S&P B&H on Sharpe **and** Max DD on the ETF pool over 2007–today.
4. Risk Monitor ablation shows ≥ 20% reduction in Max DD vs no-monitor baseline.
5. Streamlit UI exposes every `PipelineConfig` field and runs an end-to-end backtest in < 30s from button-click (warm cache).
6. CLI `run_pipeline.py` produces a reproducible MD report + Parquet artifacts.
7. `--gate` flag exits non-zero on any critical eval failure (CI-ready).
---
*End of spec.*
