# Model card — Zip-Code Housing Market IDSS

> **DRAFT — team must review before using in slides or the report.** The facts,
> metrics, and limitations below are generated/collected from the pipeline and
> model runs; the framing sentences marked `TEAM:` are where your own words go.

## Task

Predict the **3-month-ahead percentage change in median sale price** for US zip
codes, so a property-portfolio manager can rank markets and allocate a quarterly
acquisition budget. Regression target: `median_sale_price[t+3] / median_sale_price[t] − 1`.
The decision the model supports is a **ranking** of zips, so rank correlation is
the headline metric.

## Data

- **Window:** 2012-03 to 2026-05, monthly.
- **Coverage:** 20,071 zip codes; ~2.98M zip-month rows after the Redfin×Zillow
  join (90.6% of Redfin rows matched Zillow ZHVI).
- **Sources:** Redfin zip market tracker (prices, volume, inventory, DOM,
  sale-to-list), Zillow ZHVI, FRED macro (`MORTGAGE30US`, `UNRATE`, `CPIAUCSL`,
  `HOUST`). Full lineage in [feature_dictionary.md](../feature_dictionary.md);
  data caveats in [data_quality_report.md](data_quality_report.md).
- **Modeling rows:** train split ~2.7M; low-volume and outlier-target rows
  excluded from fitting and headline metrics (~1.9M rows fit).

## Model

- **Point model:** XGBoost regressor, `max_depth=7, n_estimators=400,
  learning_rate=0.1`, selected by grid search (12 combos) on **5-fold gapped
  rolling CV** over the train split, chosen on pooled rank correlation. Seed 42.
- **Confidence bands:** multi-quantile XGBoost (p05/p10/p50/p90/p95), same
  depth/learning-rate, 200 trees (`model.quantile_n_estimators`). Quantile
  crossing repaired by per-row sorting.
- **Comparison models:** naive-zero, naive-momentum, Ridge, RandomForest,
  LightGBM — see [model_comparison.md](model_comparison.md).

## Evaluation protocol

- **Selection:** gapped rolling CV (3-month gap, 36-month window) on the **train
  split only** — the 6-month holdout is untouched during model/hyperparameter
  choice (invariant 1).
- **Holdout:** the 6-month temporal test split (2025-09 to 2026-02) is scored
  **exactly once**, in M4 — see [holdout_results.md](holdout_results.md).

## Headline results

| metric | CV (train) | Holdout |
|---|---|---|
| Rank corr (Spearman) | 0.5431 | 0.5168 |
| RMSE (fraction) | 0.1014 | 0.1086 |
| Directional accuracy | 69.6% | (see holdout report) |

- XGBoost **beat the naive-momentum baseline** on holdout rank correlation
  (0.5168 vs −0.4124). LightGBM is close on both CV and holdout; XGBoost was
  chosen on the selection metric but the margin is small and documented.
- **Band calibration (train CV):** 80% band (p10–p90) empirical coverage 76.4%;
  90% band (p05–p95) 87.3% — both within ±10 points of nominal, running slightly
  narrow. **Band labels are 80% / 90%** (not "95%"); see the M3 decision in
  [model_comparison.md](model_comparison.md).

_TEAM: one paragraph interpreting these numbers — is a ~0.52 holdout rank
correlation good enough to act on, and what does the momentum baseline's negative
score tell us about this market?_

## Known limitations

1. **Redfin data is a 90-day rolling window, not a true month.** All Redfin-based
   levels are trailing-3-month aggregates, so they are smoother and more
   autocorrelated than single-month values, and the 3-month label partially
   overlaps adjacent windows. Momentum features are therefore somewhat muted.
2. **Macro features are national, not local.** Mortgage rate, unemployment, CPI,
   and housing starts are US-wide; the model cannot see a local plant closure,
   zoning change, or regional boom. Cross-zip variation comes only from the
   Redfin/Zillow features.
3. **Low-volume zips are noisy.** ~31% of zip-months sell fewer than 10 homes;
   their median prices swing on small samples. They are flagged and excluded from
   training/headline metrics but are still *scored* at prediction time, so their
   predictions carry more uncertainty than the bands may show.
4. **Coverage bias toward dense markets.** The Zillow inner join drops ~9% of
   Redfin rows (mostly small/rural zips Zillow doesn't cover), so the model is
   fit and evaluated on a denser-market panel than the full US.
5. **Regime-change risk.** Training spans 2012–2025 — mostly a rising market with
   one rate-hike cycle. A sharp crash or a rate regime outside this range is
   out-of-distribution; the confidence bands, calibrated in-sample, would likely
   under-cover in a novel regime.
6. **Bands are indicative, not exact.** Quantile models use fewer trees than the
   point model and calibrate to ~76%/87% vs 80%/90% nominal; the decision layer
   uses band *width* for relative risk ranking, which tolerates a small level
   bias, but the bands should not be read as precise probabilities.
7. **Correlational, not causal.** The scenario engine ("mortgage +50 bps →
   re-score") assumes the learned feature–target relationships hold under an
   intervention. They are associations from history, not causal effects; treat
   scenario outputs as directional what-ifs.

_TEAM: add any limitation specific to your presentation framing; the rubric
rewards honesty here._

## Intended use

Decision **support** for a property-portfolio manager: ranking zips, sizing a
quarterly budget, and stress-testing macro scenarios. **Not** an automated
trading or bidding system, and **not** individual-property valuation. A human
makes the acquisition decision; the tool informs it.

_TEAM: one sentence on who the user is and what decision they own._
