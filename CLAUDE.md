# CLAUDE.md — project context & how to work in this repo

Context for teammates and for Claude Code. Read this before making changes.

## What this is

An **Intelligent Decision Support System (IDSS)** for a property-portfolio
manager (MSCI 436, University of Waterloo). It predicts each US zip code's
**3-month-ahead median sale price % change**, attaches confidence bands, explains
the drivers, lets the user run macro "what-if" scenarios, and turns the results
into a **risk-adjusted ranking + budget allocation**. The decision it supports is
*which zips to buy into this quarter, and how to split the budget*.

Two layers, built by two executable specs:
- **`pipeline/`** (spec.md, done): public data → `data/processed/features.parquet`
  — leakage-safe features, a 3-month label, temporal train/test/live splits.
- **`model/`** (spec_model.md, done): XGBoost point + quantile models, SHAP,
  scenario engine, decision layer, monthly retrain. Ends before the Streamlit
  dashboard (a later spec); `app/` is a placeholder.

## Use the outputs WITHOUT retraining (start here)

The trained models and a ready-made ranked table are **committed**, so you don't
need to run the ~30-minute training:

- **`outputs/zip_predictions.csv`** — every zip for the latest month, ranked
  best-first, with point prediction + 80%/90% confidence bands. Open it directly.
- **`models/xgb_point.joblib`, `models/xgb_quantiles.joblib`, `models/cv_results.json`**
  — the trained models, committed. Enough to score, run scenarios, and explain.

Minimal setup to use them (no FRED key needed):
```bash
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m model export        # regenerate outputs/zip_predictions.csv from the committed models
```

Programmatic use (this is the API the dashboard will call):
```python
from pipeline.io_utils import load_config
from model.scenario import live_features, score_scenario
from model import decide

cfg = load_config()
feats = live_features(cfg)                          # all zips, most recent month
preds = score_scenario(feats, {"MORTGAGE30US": 0.5}, cfg)   # +50 bps scenario, batch-scored
ranked = decide.rank(preds, risk_lambda=1.0, config=cfg)    # score = p50 - lambda*ci80_width
kept   = decide.filter(ranked, min_roi=0.0, max_downside=0.05, ci_level=80)
alloc  = decide.allocate(kept, budget=1_000_000)            # dollar split across qualifiers
```

## Only if you need to rebuild data or retrain (needs a FRED key)

```bash
copy .env.example .env         # then add a free FRED key: https://fred.stlouisfed.org/docs/api/api_key.html
python -m pipeline all         # rebuild data/processed/features.parquet (~15 min first run; downloads ~1.5 GB)
python -m model all            # retrain point+quantile, rebuild comparison + SHAP (~30 min)
python -m model retrain        # monthly refresh cycle (see README "Operations")
```

## Repo map

| path | what |
|---|---|
| `pipeline/` | data stages: verify-schema, download, clean, join, featurize, split, report |
| `model/` | train, evaluate, predict, explain, scenario, decide, retrain, export |
| `reports/` | **tracked** grader-facing outputs: model_card, model_comparison, holdout_results, baselines, explanations, data_quality_report, figures/ |
| `outputs/` | **tracked** ready-to-use predictions (`zip_predictions.csv/.parquet`) |
| `models/` | committed trained models; per-machine retrain artifacts are gitignored |
| `data/` | gitignored; rebuilt by `python -m pipeline all` |
| `feature_dictionary.md` | every feature: definition, formula, source |
| `spec.md`, `spec_model.md` | the executable specs both layers were built from |

## Key facts to get right (don't contradict these)

- **Confidence bands are labelled 80% (p10–p90) and 90% (p05–p95) — NOT 95%.**
  This was a deliberate M3 decision; slides and any dashboard must match.
- **No leakage / temporal integrity:** a row for month *t* uses only data ≤ *t*;
  the label is realized at *t+3*. Model selection uses gapped rolling CV on the
  **train** split only; the 6-month holdout was scored exactly once.
- **Determinism:** everything is seeded (config `model.random_seed: 42`); same
  data + config → same numbers.
- **Flagged rows** (`low_volume`, `target_outlier`) are excluded from training and
  headline metrics but kept and reported — a documented decision, not silent drops.
- All tunables live in `config.yaml`; don't hardcode URLs, columns, or thresholds.

## Team to-do: DRAFT sections need your own words

These are marked `TEAM:` / "DRAFT" and are the team's intellectual contribution —
rewrite before using in the report/slides:
- `reports/model_comparison.md` → "Trade-offs" section
- `reports/explanations.md` → SHAP feature interpretations
- `reports/model_card.md` → interpretation lines and framing

## Reproduce any number

Every reported figure maps to one command — see the table in `README.md`
("Reproducing every reported number"). All deterministic at seed 42.

## Working conventions

- Tests are synthetic + offline: `pytest` (76 tests) runs with no network/key.
  The critical ones are the featurize leakage audit and the label-correctness test.
- The specs use **one task = one commit**. If you extend them, keep that, and run
  the task's verify command before committing.
- Reports in `reports/` and outputs in `outputs/` are committed so graders/teammates
  see them without a run; large data and per-machine artifacts stay gitignored.
