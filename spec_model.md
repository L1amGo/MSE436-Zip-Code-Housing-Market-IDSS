# spec_model.md — Zip-Code Housing Market IDSS: Model & Decision Layer (Executable Spec)

## How to use this spec (instructions for Claude Code)

Same task discipline as spec.md — execute tasks in order (M0 → M8), verify acceptance criteria, **stop and report** if a criterion cannot be met — with two protocol changes:

1. **Never run `git commit`, `git add`, or any other git write operation.** At the end of each task: run the verification command, print the acceptance checklist with explicit pass/fail per item, print the suggested commit message from the task's **Commit:** line, then STOP and wait. The user reviews and makes the commit themselves. Do not begin the next task until the user says to proceed.
2. **DRAFT checkpoints.** Whenever a task produces a file marked DRAFT (M2's trade-offs section, M5's interpretations, M8's model card), stop after writing it, tell the user exactly which file and sections need their review, and ask whether they want to edit before continuing. The DRAFT content is scaffolding; the claims in it must end up in the team's own words.

All tunables go in `config.yaml` under a new `model:` section. Nothing in `data/` is ever committed; everything grader-facing goes in `reports/` (tracked).

Each task carries a **Maps to** line tying it to the course rubric:
- **C1** Problem (/3) · **C2** Why-IDSS (/3) · **C3** Data (/4) · **C4** Model (/4) · **C5** UI & code (/3) · **C6** Operationalization (/3) · **P4** presentation Q&A defensibility

## Project context

The pipeline (spec.md, done) produces `data/processed/features.parquet`: one row per (zip, month) with leakage-safe features, a 3-month-ahead % price-change label, and `train`/`test`/`live` splits. This spec turns that into:
1. A trained **XGBoost regressor** with honest temporal evaluation against baselines
2. **Confidence intervals** per prediction (quantile models)
3. **SHAP explanations** per zip
4. A **scenario engine** (macro overrides → re-score) fast enough for interactive use
5. A **decision layer**: risk-adjusted ranking, threshold filtering, budget allocation
6. A **retraining entry point** for the monthly operationalization story

Out of scope: Streamlit (next spec). But every function here must be importable and callable by it: `score(features, scenario) -> predictions`, `rank(predictions, filters) -> ranked_table`, `allocate(ranked_table, budget) -> allocation`.

## Invariants (apply to every task)

1. **No test-set leakage into any modeling decision.** Hyperparameters, feature selection, and model choice use only train-split rolling CV (the gapped `rolling_cv` utility from spec.md). The 6-month holdout is touched exactly once, in M4, and never again.
2. **Determinism.** Fixed seeds everywhere (`model.random_seed` in config). Same data + config → same model, same metrics.
3. **Baseline honesty.** Every headline metric is reported next to a naive baseline. A model that can't beat "predict the zip's trailing 3-month change" has no business on a slide.
4. **Grader-visible outputs.** Metrics, plots, and the model card are written to `reports/` and committed. `models/` holds binary artifacts and stays gitignored except `.gitkeep`.
5. **Low-volume and outlier rows** (`low_volume`, `target_outlier` flags) are excluded from training and from headline metrics, but reported separately — exclusion is a documented modeling decision, not silent dropping.

---

## M0 — Model scaffold + tracked reports fix

**Do:**
- Add `model/` package: `__init__.py`, `train.py`, `evaluate.py`, `predict.py`, `explain.py`, `scenario.py`, `decide.py`, `retrain.py`, `io.py`.
- Extend the CLI: `python -m model {train,evaluate,explain,retrain,all}`.
- Add to requirements: `xgboost`, `scikit-learn`, `lightgbm`, `shap`, `matplotlib`, `joblib`.
- Create tracked `reports/` directory; move the pipeline's `data_quality_report.md` and `schema_report.md` writes there (small edit to `pipeline/report.py` and `pipeline/verify_schema.py` + config paths) and commit the current reports. This fixes the known gap where grader-facing reports were gitignored.
- Add `model:` config section: `random_seed`, `quantiles: [0.05, 0.1, 0.5, 0.9, 0.95]`, `cv_folds`, `xgb_param_grid` (small, ~12 combos max), `exclude_flags: [low_volume, target_outlier]`, `top_k_shap_features`, `scenario_features` (the macro columns a user may override: mortgage rate level + its 3-month delta), `ci_levels: [80, 95]` (user-selectable band; 80 → p10/p90, 95 → p05/p95), `risk_lambda_default: 1.0` and `risk_lambda_range: [0.0, 3.0]` (risk-tolerance coefficient for ranking), `default_min_roi`, `default_max_downside`, `degradation_rmse_multiplier: 2.0`.

**Accept when:**
- [ ] `python -m model --help` lists all stages; `pytest` still fully passes
- [ ] `reports/data_quality_report.md` and `reports/schema_report.md` are tracked in git with real numbers
- [ ] Fresh `pip install -r requirements.txt` succeeds

**Verify:** `python -m model --help && git ls-files reports/`
**Commit:** `model-scaffold: package, CLI, deps, tracked reports`
**Maps to:** C5 (organized repo; fixes grader-invisible reports), C3 (limitations report now actually visible).

---

## M1 — Baselines

**Do:** Implement three baselines in `evaluate.py`, all scored with the same gapped rolling CV on train only:
- `naive_zero`: predict 0% change
- `naive_momentum`: predict the zip's trailing 3-month % change
- `linear`: Ridge regression on the full feature set
Metrics per fold and pooled: RMSE, MAE, directional accuracy (sign agreement), and rank correlation (Spearman between predicted and realized change across zips within each month — the metric closest to the actual use, since the manager acts on *rankings*). Write `reports/baselines.md`.

**Accept when:**
- [ ] All three baselines produce all four metrics via the same CV harness
- [ ] `reports/baselines.md` committed with a table and one paragraph interpreting it
- [ ] A unit test confirms the CV harness excludes flagged rows and never touches the test split

**Verify:** `python -m model evaluate --baselines-only && pytest tests/test_model_eval.py`
**Commit:** `baselines: naive and linear reference models under gapped CV`
**Maps to:** C4 (justification requires something to be justified *against*), P4 ("why is XGBoost better than something simple?" gets a numeric answer).

---

## M2 — XGBoost training + model comparison

**Do:**
- `train.py`: XGBoost regressor, hyperparameters chosen by grid search over `xgb_param_grid` using the gapped rolling CV (train split only). Save best model + chosen params via joblib to `models/`.
- Same harness run for RandomForest and LightGBM with modest default-ish grids — these are the comparison models named in the project proposal.
- Write `reports/model_comparison.md`: CV metrics for all models + baselines in one table, chosen hyperparameters, and a short trade-offs section written as *claims the team must review and own* (advantages AND disadvantages of XGBoost for THIS task — e.g., handles mixed-scale tabular features and missingness natively, strong on medium-sized panels; but opaque without SHAP, can overfit small zips, no native uncertainty). Mark this file with a header: "DRAFT — team must edit before using in slides."

**Accept when:**
- [ ] Grid search logs every combo's CV score; best params land in the saved artifact and the report
- [ ] Comparison table covers: 2 naive + linear + RF + LightGBM + XGBoost, all four metrics
- [ ] Retraining with the same seed reproduces identical CV metrics
- [ ] Total training time < ~15 min on a laptop (shrink the grid if not; note it)

**Verify:** `python -m model train && python -m model evaluate`
**Commit:** `train: XGBoost with gapped-CV selection vs RF/LightGBM`
**Maps to:** C4 directly — "justify against your specific task, not the domain" and "trade-offs from both sides" are near-verbatim rubric language; the comparison table is that slide's content. Also invariant 1 protects C4's credibility.

---

## M3 — Confidence intervals

**Do:** Train quantile XGBoost models at the configured quantiles (0.05/0.1/0.5/0.9/0.95) with the selected hyperparameters. `predict.py` returns per-row: point prediction (mean model), `p05`, `p10`, `p50`, `p90`, `p95`, and both band widths (`ci80_width = p90 − p10`, `ci90_width = p95 − p05`). Evaluate calibration on train CV for BOTH bands: empirical coverage vs. nominal (p10–p90 nominal 80%; p05–p95 nominal 90%). Write coverage numbers for both bands into `reports/model_comparison.md`, including an honest note if the wider band calibrates poorly on noisy zip-level data.

**Band-label decision:** the mockup says "95% CI", but a p05–p95 band is nominally 90%. Either (a) train p025/p975 instead of p05/p95 so the "95%" label is literally true, or (b) keep p05/p95 and label the toggle "80% / 90%". Decide at implementation, record the choice in the model card, and make the dashboard and slide labels match it exactly.

**Accept when:**
- [ ] Predictions carry all quantile columns; monotonicity `p05 ≤ p10 ≤ p50 ≤ p90 ≤ p95` enforced (crossing fix documented if needed)
- [ ] Empirical coverage reported for both bands vs. nominal; miscalibration explained honestly if outside ±10 points
- [ ] Band-label decision recorded (95% via p025/p975, or relabel to 90%)
- [ ] Unit test: quantile columns present, ordered, finite

**Verify:** `python -m model evaluate --intervals && pytest tests/test_predict.py`
**Commit:** `intervals: quantile models with calibration check`
**Maps to:** C4 (uncertainty handling), C1 (the proposal's promise of predictions "with a confidence interval"), and feeds the decision layer's risk adjustment (M6).

---

## M4 — Holdout evaluation (touch the test set ONCE)

**Do:** Final models (point + quantiles), trained on the full train split with chosen params, evaluated on the 6-month holdout. Report all four metrics vs. all baselines, plus a monthly breakdown (is performance stable across the 6 months?) and a backtest-style plot: predicted vs. realized ranking quality per month. Write `reports/holdout_results.md` + PNG plots in `reports/figures/`. After this task, the test split is frozen — later tasks may not recompute holdout metrics with different models.

**Accept when:**
- [ ] Holdout table committed: every model, every metric, monthly breakdown
- [ ] At least one figure: predicted vs. realized (scatter or rank plot), axes labeled (rubric penalizes unlabeled figures)
- [ ] The report states plainly whether XGBoost beat the naive momentum baseline on the holdout, and by how much — whatever the answer is

**Verify:** `python -m model evaluate --holdout`
**Commit:** `holdout: one-shot test evaluation with figures`
**Maps to:** C4 (evidence the model works), C3 (honest performance is part of honest data limitations), P4 (the numbers you'll be asked to defend), slide figures for the deck.

---

## M5 — SHAP explanations

**Do:** `explain.py`: global SHAP summary (top features overall) and a per-zip function `explain_zip(zip, month) -> top-k feature contributions` for the dashboard. Save the global summary plot to `reports/figures/shap_summary.png`; write `reports/explanations.md` with the top-10 features and one-line plain-English interpretations (marked DRAFT for team review — the *interpretations* are the team's intellectual contribution).

**Accept when:**
- [ ] Global plot committed with labeled axes
- [ ] `explain_zip` returns in < 1s per zip (precompute/cached explainer)
- [ ] Test: contributions approximately sum to (prediction − expected value)

**Verify:** `python -m model explain && pytest tests/test_explain.py`
**Commit:** `explain: global and per-zip SHAP`
**Maps to:** C4 (mitigates the "opaque model" disadvantage you'll list), C5 (the dashboard's SHAP panel — visualizations tied to the decision), C2 (explanations are part of why a manager can *trust* and interact with the tool).

---

## M6 — Scenario engine + decision layer

**Do:**
- `scenario.py`: `apply_scenario(features, overrides) -> features'` where overrides adjust the configured macro columns (e.g., mortgage rate +50 bps) AND consistently recompute their derived deltas; then `score(features') -> predictions`. Batch-scores all zips in one call.
- `decide.py`:
  - `rank(predictions, risk_lambda)`: risk-adjusted score = `p50 − risk_lambda × ci80_width`; rank descending. **The score band is always p10–p90 regardless of the display/filter CI level**, so rankings are comparable across toggle states; `risk_lambda` comes from the dashboard's risk-tolerance slider (config default/range).
  - `filter(ranked, min_roi, max_downside, ci_level, exclude_zips=None)`: keep zips with `p50 ≥ min_roi` and lower-bound ≥ −max_downside, where the lower bound is `p10` at ci_level 80 and the lower quantile of the wider band at the higher ci_level — the CI toggle is an eligibility knob (stricter level → fewer qualifiers). Any zip in `exclude_zips` (iterable of 5-char zip strings — the manager's watchlist/owned-markets exclusion list) is removed from the candidate set before allocation.
  - `allocate(filtered, budget)`: budget shares proportional to risk-adjusted score, normalized; returns dollar allocation per zip
- Benchmark: full re-score + re-rank of all zips under a new scenario.

**Accept when:**
- [ ] Scenario overrides change derived macro-delta features consistently (test asserts it)
- [ ] End-to-end scenario → allocation runs in ≤ 2s for the full zip universe on a laptop (log the number; the proposal's interactivity claim depends on this being snappy)
- [ ] `allocate` shares sum to the budget; filters demonstrably change the output (tests)
- [ ] λ test: raising `risk_lambda` demotes a high-uncertainty zip below a lower-return/low-uncertainty zip on a synthetic fixture (the slider genuinely re-ranks)
- [ ] CI-level test: switching to the stricter level never grows the qualifying set, and shrinks it on a fixture spanning the two lower bounds
- [ ] `exclude_zips` test: an excluded zip never appears in the ranked output or allocation, and its budget share is redistributed across the remaining qualifiers
- [ ] Functions take plain DataFrames/params — no Streamlit imports anywhere in `model/`

**Verify:** `pytest tests/test_scenario.py tests/test_decide.py && python -m model scenario-bench`
**Commit:** `decision: scenario re-scoring, risk-adjusted ranking, allocation`
**Maps to:** C2 — this is the crux of "why an IDSS": the interactive scenario re-ranking is your argument that a static dashboard can't substitute. C5 — rubric demands controls that change model outputs, not cosmetic filters; this layer is what makes that literally true. C1 — allocation is the actual decision the stakeholder makes.

---

## M7 — Retraining entry point

**Do:** `retrain.py`: one command (`python -m model retrain`) that re-runs pipeline `download → split` for fresh data, retrains point + quantile models on the rolling 36-month window, recomputes monitoring metrics (RMSE/MAE/directional accuracy on the most recent labeled months) and simple feature-drift stats (mean/std shift vs. training distribution), appends a dated entry to `reports/retrain_log.md`, and versions the model artifact (timestamped filename + `latest` pointer).
- **Degradation alert:** compare the fresh RMSE against the baseline RMSE recorded in the model card (config: `model.degradation_rmse_multiplier: 2.0`). If fresh RMSE > multiplier × baseline, print a prominent `DEGRADATION ALERT` line, write it into the `retrain_log.md` entry, and exit with a nonzero status code so any scheduler (cron etc.) registers the run as failed and can notify. The `latest` pointer is still updated only if the operator re-runs with `--accept-degraded`; by default a degraded model is saved but not promoted. Document in the README how this maps to the proposal's "monthly overnight retrain" (cron/scheduler invokes this command).

**Accept when:**
- [ ] Single command performs the full refresh cycle; failures in any stage abort loudly
- [ ] `retrain_log.md` gains a dated entry with metrics and drift stats
- [ ] Degradation alert test: with a synthetically inflated baseline comparison (or forced bad metrics), retrain prints the alert, logs it, exits nonzero, and does not promote `latest`
- [ ] Old model artifacts retained (rollback possible); `latest` pointer updated
- [ ] README gains an "Operations" section: schedule, what runs, what's monitored, rollback

**Verify:** `python -m model retrain` (may use cached data)
**Commit:** `retrain: monthly refresh cycle with monitoring and versioning`
**Maps to:** C6 — all four rubric elements: pipeline keeps running (reuses download), when/how retraining happens (this command + schedule), monitoring (metrics + drift), infrastructure (documented; the t3.large story from the proposal). Also C4's "how the model handles evolving data."

---

## M8 — Model card + README update

**Do:** Write `reports/model_card.md`: task, data window, features (link to feature_dictionary), model + params, evaluation protocol (gapped CV, one-shot holdout), all headline results, calibration, known limitations (low-volume zips, macro features are national not local, regime-change risk), and intended use ("decision support for a portfolio manager; not automated trading"). Mark DRAFT for team review. Update README: model section, how to reproduce every number in `reports/`.

**Accept when:**
- [ ] Every number on a future slide is reproducible via a documented command
- [ ] Model card's limitations section is substantive (≥5 real limitations), not boilerplate
- [ ] `pytest` fully green; `python -m model all` runs end-to-end

**Verify:** `python -m model all && pytest`
**Commit:** `docs: model card and reproducibility guide`
**Maps to:** C4 (justification in citable form), C3 (limitations), C5 (documentation), P4 (the model card is your Q&A crib sheet), AI-disclosure defense (DRAFT markers force team review, keeping the intellectual core yours).

---

## Rubric coverage summary

| Task | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|
| M0 scaffold + reports fix | | | ✔ | | ✔ | |
| M1 baselines | | | | ✔ | | |
| M2 XGBoost + comparison | | | | ✔✔ | | |
| M3 intervals | ✔ | | | ✔ | | |
| M4 holdout | | | ✔ | ✔✔ | | |
| M5 SHAP | | ✔ | | ✔ | ✔ | |
| M6 scenario + decision | ✔ | ✔✔ | | | ✔✔ | |
| M7 retrain | | | | ✔ | | ✔✔✔ |
| M8 model card | | | ✔ | ✔ | ✔ | |

C1 and C2 are ultimately argued on slides, but M6 is the implementation evidence behind the C2 argument. The remaining unclaimed points sit in the dashboard spec (C5's UI half) and the deck itself.
