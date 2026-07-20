# spec_dashboard.md — Streamlit Decision Dashboard (D0–D6)

**Prereqs:** `spec.md` (pipeline, T0–T8) and `spec_model.md` (M0–M8) complete. `model/` exposes `score()`, `rank()`, `filter()`, `allocate()`, `explain_zip()`, and a trained artifact in `models/latest/`.

**Scope:** the Streamlit UI layer only. No modelling logic, no feature engineering, no retraining code in `app/`.

---

## How to use this spec (protocol for Claude Code)

1. Work through tasks **D0 → D6 strictly in order**. Do not skip ahead or merge tasks.
2. **Never run git commands.** At each task boundary: run the verify command, print the acceptance checklist with pass/fail per item, print the suggested commit message, then **stop and wait**. The human commits and says "proceed."
3. If an acceptance criterion cannot be met, **stop and report**. Do not improvise a workaround.
4. **Halt at every DRAFT file** (D5 screenshot captions, D6 user guide). Name the sections needing the team's words and ask before continuing.

---

## Global invariants (apply to every task)

- **Thin UI.** `app/` contains layout, widgets, caching, and plotting only. Any computation lives behind a `model/` function call. A grep for modelling logic in `app/` must come back clean.
- **Controls are load-bearing.** Every widget must change a model output or the qualifying/allocation set. No widget may only filter a rendered table. This is a hard rubric requirement.
- **Fast enough to be interactive.** Any control change must re-rank the full zip universe in ≤ 2 s (measured, not asserted).
- **No secrets, no absolute paths.** Everything reads from `config.yaml` and `.env`.
- **Fail loudly, in the interface's voice.** Missing model artifact or missing `features.parquet` produces a clear on-screen message telling the user which command to run, not a stack trace.
- **Stranger-runnable.** `streamlit run app/main.py` works from a fresh clone after the documented setup steps.

---

## D0 — App scaffold and data/model loading

**Do**
- Create `app/main.py`, `app/state.py`, `app/components/` (`sidebar.py`, `table.py`, `map.py`, `drilldown.py`).
- `app/state.py`: `load_model()` and `load_live_features()` wrapped in `@st.cache_resource` / `@st.cache_data`.
- Page config: title, wide layout, sidebar expanded.
- If `models/latest/` or the live feature slice is missing, render a single clear message naming the command to run, and stop.

**Acceptance**
- [ ] `streamlit run app/main.py` starts with no exceptions
- [ ] App renders row count, zip count, and the as-of month of the live slice
- [ ] Deleting `models/latest/` produces the friendly message, not a traceback
- [ ] `grep -rE "xgboost|shap|sklearn" app/` returns nothing

**Verify:** `python -c "import app.main"` and a manual launch
**Commit:** `dashboard: app scaffold with cached model and feature loading`
**Maps to:** C5 (organized, documented, runnable code)

---

## D1 — Parameter sidebar (all controls wired)

**Do** — build every control, each bound to a `model/` parameter:

| Control | Widget | Feeds |
|---|---|---|
| Target metros | multiselect | universe subset |
| Min expected ROI | slider (%) | `filter(min_roi=...)` |
| Max acceptable downside | slider (%) | `filter(max_downside=...)` |
| CI level | radio, 80% / 95% | `filter()` lower bound (p10 vs p05) and displayed band |
| Risk tolerance λ | select_slider Low/Moderate/High | `rank(lam=...)`, score `= p50 − λ·ci80_width` |
| Rate scenario | radio, −50 / −25 / 0 / +25 / +50 / +100 bps | `score(features, scenario=...)` |
| Quarterly budget | number_input | `allocate(budget=...)` |
| Exclusions upload | file_uploader (CSV of zips) | `filter(exclude_zips=...)` |

- A "Reset to defaults" button. Defaults live in `config.yaml`, not hardcoded.

**Acceptance**
- [ ] Every control above exists and its value reaches a `model/` call
- [ ] Changing rate scenario changes predicted values (not just the display)
- [ ] Raising CI level from 80% to 95% can only shrink the qualifying set
- [ ] Raising λ re-orders the ranked table
- [ ] Uploaded exclusions never appear in the allocation, and their budget share redistributes
- [ ] Malformed exclusions CSV produces a clear message naming the expected format

**Verify:** `pytest tests/test_dashboard_controls.py -v` (tests call the same handler functions the widgets call)
**Commit:** `dashboard: parameter sidebar wired to scoring and decision layer`
**Maps to:** C2 (interactive control is necessary), C5 (controls genuinely change model outputs)

---

## D2 — Ranked table and allocation view

**Do**
- Main panel: ranked table of qualifying zips with columns zip, metro, predicted 3-month change (p50), CI band per the toggle, score, rank, allocated dollars, allocated share.
- Header strip: number of zips evaluated, number qualifying, budget deployed, budget unallocated.
- Footnote defining the score formula on screen: `score = p50 − λ · (p90 − p10)`.
- Allocation bar or treemap beneath the table.
- CSV download of the current allocation.

**Acceptance**
- [ ] Table reflects the current control state on every change
- [ ] Allocated dollars sum to the budget (or to the deployed figure, with the remainder shown)
- [ ] Score formula visible in the UI
- [ ] Every figure has an axis label and units

**Verify:** `pytest tests/test_dashboard_allocation.py -v`
**Commit:** `dashboard: ranked table and budget allocation view`
**Maps to:** C1 (the decision the system supports), C5 (visualizations tied to the allocation decision)

---

## D3 — Choropleth map

**Do**
- Zip-level choropleth colored by predicted appreciation, restricted to selected metros.
- Diverging color scale centered at 0, with a fixed domain so colors mean the same thing across scenarios.
- Hover shows zip, p50, CI band, rank, allocation.
- Non-qualifying zips visibly de-emphasized (greyed), not removed, so the manager sees what was rejected.

**Acceptance**
- [ ] Map re-colors when the rate scenario changes
- [ ] Color scale domain is fixed and documented in the legend
- [ ] Geometry file is cached; map renders in ≤ 2 s after first load
- [ ] Zips with no geometry match are counted and reported, not silently dropped

**Verify:** manual launch plus `pytest tests/test_geo_join.py -v`
**Commit:** `dashboard: zip-level choropleth with scenario-responsive coloring`
**Maps to:** C3 (coverage honesty), C5

---

## D4 — Per-zip drill-down

**Do**
- Select a zip from the table or map to open a detail panel calling `explain_zip()`.
- Panel contains: SHAP waterfall for that zip's prediction, prediction with CI band, feature values vs metro median, and the backtest chart of predicted vs realized for that zip over the holdout.
- One plain-language line summarizing the top three drivers.

**Acceptance**
- [ ] Drill-down opens for any zip in the table
- [ ] SHAP values come from `explain_zip()`, computed nowhere in `app/`
- [ ] Backtest chart plots holdout months only, never training months
- [ ] Panel renders in ≤ 2 s

**Verify:** `pytest tests/test_drilldown.py -v`
**Commit:** `dashboard: per-zip drill-down with SHAP and backtest`
**Maps to:** C4 (model is explainable), C5

---

## D5 — Performance benchmark and screenshots

**Do**
- `scripts/benchmark_dashboard.py`: time a full-universe re-rank under a scenario change, 10 runs, report median and p95 to `reports/dashboard_benchmark.md`.
- Capture screenshots to `reports/figures/`: full dashboard, sidebar with controls, scenario before/after pair, drill-down panel.
- Write `reports/figures/CAPTIONS.md` with **DRAFT** captions. Halt for the team to write final wording.

**Acceptance**
- [ ] `reports/dashboard_benchmark.md` exists with a measured median re-rank time
- [ ] Median re-rank ≤ 2 s, or the shortfall is reported rather than hidden
- [ ] Four screenshots exist in `reports/figures/`, PNG, legible at slide size
- [ ] Before/after pair shows a visible ranking change from one control move
- [ ] `reports/` is tracked, not gitignored

**Verify:** `python scripts/benchmark_dashboard.py && ls reports/figures/`
**Commit:** `dashboard: performance benchmark and deck screenshots`
**Maps to:** C5 (demo of a working system), and replaces the unmeasured "~50 ms per zip" slide claim with a real number

---

## D6 — README, user guide, fresh-clone test

**Do**
- Extend `README.md` with a "Run the dashboard" section: setup, pipeline, train, launch, expected first-load time.
- `docs/user_guide.md` (**DRAFT**): one short paragraph per control, written from the manager's side of the screen, saying what the control does to the decision. Halt for team wording.
- Fresh-clone test: clone to a temp directory, follow the README exactly, record any missing step and fix it.

**Acceptance**
- [ ] A person who has never seen the repo can launch the dashboard from the README alone
- [ ] No step in the README requires oral explanation
- [ ] Fresh-clone test performed and any gaps fixed
- [ ] DRAFT user guide flagged for the team

**Verify:** `git clone <repo> /tmp/fresh && cd /tmp/fresh && <README steps>`
**Commit:** `dashboard: README run instructions and user guide`
**Maps to:** C5 ("runnable by a third party without oral instruction")

---

## Rubric coverage matrix

| Criterion | Covered by | Notes |
|---|---|---|
| C1 Problem (/3) | D2 | Allocation view makes the decision visible; argued on slides |
| C2 Why an IDSS (/3) | D1, D3 | Scenario re-ranking is the substitution test: a static report cannot do this |
| C3 Data (/4) | D0, D3 | As-of month, row/zip counts, unmatched-geometry count feed the limitations slide |
| C4 Model (/4) | D4 | Explainability; the numbers themselves come from M4 |
| C5 UI and code (/3) | D0–D6 | The bulk of this criterion lives here |
| C6 Operationalization (/3) | D0, D6 | How users access the system; infrastructure argued on slides |

## Slide artifacts this spec produces

- Real dashboard screenshot replacing the mockup (C5)
- Before/after scenario pair proving controls change model outputs (C2, C5)
- Measured re-rank time replacing "~50 ms per zip" (C5, C6)
- Zip/row counts and unmatched-geometry count for the data slide (C3)

## Out of scope

- Authentication, multi-user state, deployment to AWS
- Any change to features, labels, or model training
- Additional prediction horizons