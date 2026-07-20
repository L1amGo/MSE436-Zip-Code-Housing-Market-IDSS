# Zip-Code Housing Market IDSS

Predicts the 3-month-ahead median sale price % change for US zip codes so a
property portfolio manager can rank markets, stress-test macro scenarios, and
allocate a quarterly acquisition budget. Two layers:

1. **Data pipeline** (`pipeline/`): raw public data → `data/processed/features.parquet`
   with leakage-safe features, a 3-month label, and temporal splits.
2. **Model & decision layer** (`model/`): XGBoost point + quantile models, SHAP
   explanations, a scenario engine, and a risk-adjusted ranking/allocation
   decision layer. The Streamlit dashboard is a later spec; `app/` is a placeholder.

MSCI 436 course project, University of Waterloo.

## Architecture

```
 Redfin S3 (gz TSV)   Zillow ZHVI (CSV)   FRED API (JSON)
        |                    |                  |
        +--------------------+------------------+
                             |
                     [ verify-schema ]   live header checks + drift detection
                             |
                       [ download ]      cached pulls -> data/raw/ + manifest.json
                             |
                        [ clean ]        tidy per-source parquets -> data/interim/
                             |
                        [ join ]         zip-month panel -> data/interim/joined.parquet
                             |
                      [ featurize ]      features + 3-month label -> data/processed/features.parquet
                             |
                        [ split ]        train / test / live column + rolling-CV utility
                             |
                        [ report ]       reports/data_quality_report.md
                             |
        ============ model & decision layer (model/) ============
                             |
   [ train ]  XGBoost point + quantiles, gapped-CV selection -> models/ + model_comparison.md
   [ evaluate --holdout ]  one-shot 6-month test eval -> holdout_results.md + figures/
   [ explain ]  global + per-zip SHAP -> shap_summary.png + explanations.md
   [ scenario + decide ]  re-score under macro shifts; rank / filter / allocate
   [ retrain ]  monthly refresh + monitoring + versioning -> retrain_log.md
```

## Setup

Requires Python 3.11+ and a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).

```bash
git clone https://github.com/L1amGo/MSE436-Zip-Code-Housing-Market-IDSS.git
cd MSE436-Zip-Code-Housing-Market-IDSS
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then put your key after FRED_API_KEY=
```

## Running

```bash
python -m pipeline all          # everything, in order
python -m pipeline <stage>      # one stage: verify-schema | download | clean | join | featurize | split | report
python -m pipeline download --force   # re-download even if cached
```

Every stage logs rows in/out per filter step and fails loudly on schema
drift, misaligned joins, or empty outputs.

**Expected runtime** (laptop, home connection): download ~8 min on the first
run (the Redfin file is ~1.5 GB; later runs are fully cached), clean ~2.5 min,
everything after < 1 min each. **Disk use:** ~1.6 GB in `data/raw/`, ~0.5 GB
in `data/interim/` + `data/processed/`.

## Outputs

| path | what |
|---|---|
| `data/raw/` | untouched source files + `manifest.json` (URL, timestamp, size, sha256) + `schema_report.md` |
| `data/interim/` | tidy per-source parquets (`redfin`, `zillow`, `fred`) and the merged `joined.parquet` |
| `data/processed/features.parquet` | modeling table: features, `target`, `split` (train/test/live) |
| `reports/` | **tracked** grader-facing reports (see below) |
| `feature_dictionary.md` | every column: definition, formula, source |

`data/` and `models/` are gitignored; a fresh clone rebuilds them with
`python -m pipeline all` then `python -m model all`. Reports in `reports/` are
committed so they're visible without a run.

## Model & decision layer

```bash
python -m model train           # grid-search XGBoost vs RF/LightGBM; fit point + quantile models
python -m model evaluate        # rebuild the CV comparison report from cached results
python -m model evaluate --baselines-only   # M1 reference baselines
python -m model evaluate --intervals        # confidence-band calibration
python -m model evaluate --holdout          # one-shot 6-month holdout eval + figures
python -m model explain         # global + per-zip SHAP; writes shap_summary.png
python -m model scenario-bench  # time a full scenario -> allocation over all zips
python -m model export          # write outputs/zip_predictions.csv (ranked table, no retrain)
python -m model all             # train, evaluate, explain in order
```

**Using the outputs without retraining:** the trained models
(`models/xgb_point.joblib`, `models/xgb_quantiles.joblib`, `models/cv_results.json`)
and a ranked prediction table (`outputs/zip_predictions.csv`) are committed, so a
teammate can `pip install -r requirements.txt` and immediately run
`python -m model export` or the importable `predict`/`scenario`/`decide` API — no
FRED key, no training. See [CLAUDE.md](CLAUDE.md).

The decision layer is importable for the dashboard (no UI deps):
`scenario.apply_scenario` / `score_scenario`, `decide.rank` / `filter` / `allocate`,
and `predict.score` (point + p05/p10/p50/p90/p95 + band widths).

**Confidence bands are labelled 80% (p10–p90) and 90% (p05–p95)** — not "95%".

**Runtime:** `model train` ~12 min (grid) + a multi-quantile fit; SHAP ~10 s;
`scenario-bench` well under 1 s for ~19k zips. `models/` binaries are gitignored.

## Operations (monthly retrain)

```bash
python -m model retrain                  # refresh data -> retrain on rolling 36-month window
python -m model retrain --accept-degraded # promote even if the degradation gate trips
```

Intended to run on a schedule (e.g. cron / a small VM overnight each month). Each
run: refreshes data (`download → split`, download cached), retrains point +
quantile models on the most recent 36 labeled months, computes out-of-sample
**monitoring metrics** and **feature-drift** stats, versions the artifact
(timestamped file + a `latest` pointer, so rollback is possible), and appends a
dated entry to `reports/retrain_log.md`.

**Degradation gate:** if the fresh RMSE exceeds `model.degradation_rmse_multiplier`
× the baseline RMSE, the run prints a `DEGRADATION ALERT`, logs it, and **exits
nonzero (code 3) without promoting** `latest` — a scheduler registers the run as
failed and can notify. The degraded model is saved for inspection; promote it
manually with `--accept-degraded`.

## Reproducing every reported number

Every figure in the reports and slides traces to one command (all deterministic,
seed 42):

| number / artifact | command | lands in |
|---|---|---|
| Panel size, coverage, missing values | `python -m pipeline all` | `reports/data_quality_report.md` |
| Baseline CV metrics | `python -m model evaluate --baselines-only` | `reports/baselines.md` |
| Model comparison table + chosen params | `python -m model train` | `reports/model_comparison.md` |
| Band calibration (80% / 90% coverage) | `python -m model evaluate --intervals` | `reports/model_comparison.md` |
| Holdout metrics + figures | `python -m model evaluate --holdout` | `reports/holdout_results.md`, `reports/figures/` |
| SHAP top features + summary plot | `python -m model explain` | `reports/explanations.md`, `reports/figures/` |
| Scenario→allocation latency | `python -m model scenario-bench` | stdout log |
| Model card (all headline results) | — (curated) | `reports/model_card.md` |

## Tests

```bash
pytest            # whole suite; synthetic fixtures, no network needed
```

`tests/test_featurize.py` holds the two critical safety tests (label correctness
and the lookahead/leakage audit). `tests/test_predict.py`, `test_decide.py`,
`test_scenario.py`, and `test_retrain.py` cover the model layer's quantile
monotonicity, ranking/allocation logic, scenario consistency, and the
degradation gate.

## Data & model notes

- Feature definitions: [feature_dictionary.md](feature_dictionary.md)
- Data limitations (coverage, low-volume flags, the Redfin 90-day window
  convention, the Oct-2025 FRED shutdown gap): [reports/data_quality_report.md](reports/data_quality_report.md)
- Model card (task, protocol, results, limitations, intended use):
  [reports/model_card.md](reports/model_card.md) — **DRAFT, team to finalize**
- Splits: `test` = most recent 6 labeled months; CV folds keep a 3-month gap
  and a 36-month training window (`pipeline.split.rolling_cv`)
