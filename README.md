# Zip-Code Housing Market IDSS — Data Pipeline

Predicts 3-month-ahead median sale price % change for US zip codes so a
property portfolio manager can allocate a quarterly acquisition budget.
**This repo currently contains the data pipeline only**: raw public data in →
`data/processed/features.parquet` with leakage-safe features, labels, and
temporal splits out. Model (XGBoost) and dashboard (Streamlit) come later;
`models/` and `app/` are placeholders.

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
                        [ report ]       data/data_quality_report.md
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
| `data/data_quality_report.md` | honest panel summary: coverage, flags, missing values, caveats |
| `feature_dictionary.md` | every column: definition, formula, source |

`data/` is gitignored; a fresh clone rebuilds it with `python -m pipeline all`.

## Tests

```bash
pytest            # whole suite; synthetic fixtures, no network needed
```

`tests/test_featurize.py` contains the two critical safety tests: the label
exactly equals the change realized at t+3, and a lookahead audit proving that
corrupting all data after month t changes no feature value at t.

## Data notes

- Feature definitions: [feature_dictionary.md](feature_dictionary.md)
- Data limitations (coverage, low-volume flags, outliers, the Redfin 90-day
  window convention, the Oct-2025 FRED shutdown gap): `data/data_quality_report.md`
  after a run
- Splits: `test` = most recent 6 labeled months; CV folds keep a 3-month gap
  and a 36-month training window (`pipeline.split.rolling_cv`)
