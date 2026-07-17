# spec.md — Zip-Code Housing Market IDSS: Data Pipeline (Executable Spec)

## How to use this spec (instructions for Claude Code)

- Execute tasks **in order** (T0 → T8). Do not start a task until the previous task's acceptance criteria all pass.
- Each task ends with: (1) run the verification command, (2) confirm every acceptance criterion, (3) make one git commit using the given commit message. One task = one commit.
- If an acceptance criterion cannot be met (e.g., a URL is dead, a column doesn't exist), **stop and report** the discrepancy with what you found. Do not silently work around it or invent data.
- All tunable values live in `config.yaml`. Never hardcode URLs, column names, thresholds, or window sizes in Python.
- Never commit anything in `data/`, any `.env` file, or any API key.

## Project context (one paragraph)

MSCI 436 course project (University of Waterloo). An IDSS that predicts 3-month-ahead median sale price % change for US zip codes so a property portfolio manager can allocate a quarterly acquisition budget. This spec covers the **data pipeline only**: raw public data in → `features.parquet` with labels and splits out. Model (XGBoost) and Streamlit dashboard are later specs; leave `models/` and `app/` as empty placeholder dirs.

## Non-negotiable invariants (apply to every task)

1. **Temporal integrity.** A row for (zip, month `t`) may only contain information available at month `t`. The label is the price change realized at `t+3`. No feature may leak future information.
2. **Determinism.** Same inputs → same outputs. No randomness anywhere in the pipeline.
3. **Visible data flow.** Every stage logs row counts in/out and reasons for drops.
4. **Fail loudly.** Schema mismatches, failed joins, and empty outputs raise errors with actionable messages; they never pass silently.
5. **Reproducible by a stranger.** A third party with a fresh clone, Python 3.11+, and a FRED API key must be able to run everything from the README alone.

## Data sources (reference)

| Source | What | Access | Config keys |
|---|---|---|---|
| Redfin Data Center | Zip-code market tracker: median sale price, inventory, homes sold, new listings, median DOM, sale-to-list, price drops, etc. 2012+. Multi-GB gzipped TSV. | Public S3 URL, historically `https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz` — **verify in T1** | `sources.redfin.url`, `schema.redfin.*` |
| FRED API | National macro: `MORTGAGE30US` (weekly → monthly mean), `UNRATE`, `CPIAUCSL`, `HOUST` | REST, key from env var `FRED_API_KEY` | `sources.fred.series`, `sources.fred.base_url` |
| Zillow Research | ZHVI, zip-level, smoothed + seasonally adjusted. Wide CSV (one column per month). | Public CSV from zillow.com/research/data — **verify in T1** | `sources.zillow.url`, `schema.zillow.*` |

Fixed parameters (put in `config.yaml`): label horizon = 3 months; holdout = most recent 6 labeled months; CV gap = 3 months; training window = 36 months; low-volume threshold = 10 homes sold/month; property type = "All Residential"; monthly frequency everywhere; dev metro filter (optional list of metros to shrink data during development, empty = all).

---

## T0 — Repo scaffold

**Do:** Create the structure below, `requirements.txt` (pandas, pyarrow, requests, pyyaml, python-dotenv, pytest), `.gitignore` (`data/`, `.env`, `__pycache__/`, `*.parquet`), `.env.example` containing `FRED_API_KEY=`, a `config.yaml` skeleton with all keys named above (schema section empty until T1), and a stub CLI (`python -m pipeline <stage>`) that dispatches to not-yet-implemented stage functions with a clear "not implemented" message.

```
├── README.md  ├── spec.md  ├── config.yaml  ├── .env.example  ├── .gitignore  ├── requirements.txt
├── pipeline/ (__init__.py, __main__.py, verify_schema.py, download.py, clean.py, join.py, featurize.py, split.py, io_utils.py)
├── tests/    ├── data/ (gitignored)    ├── models/ (empty, .gitkeep)    ├── app/ (empty, .gitkeep)
```

**Accept when:**
- [ ] `pip install -r requirements.txt` succeeds in a fresh venv
- [ ] `python -m pipeline download` prints a not-implemented message and exits nonzero
- [ ] `git status` shows no `data/` or `.env` tracked

**Verify:** `python -m pipeline --help` lists all stages.
**Commit:** `scaffold: repo structure, config skeleton, CLI dispatch`

---

## T1 — verify-schema

**Do:** Implement `verify-schema`:
- Redfin: stream only the first 2 lines (`requests` with streaming + gzip, or subprocess `curl | gunzip | head -2`); print the exact header.
- Zillow: read only the CSV header row.
- FRED: request 1 observation per series; confirm each series ID exists and note its native frequency.
- Interactively-discovered real column names go into `config.yaml` under `schema:` as a logical→actual mapping (logical names: `period_begin`, `period_end`, `region_zip`, `property_type`, `median_sale_price`, `homes_sold`, `inventory`, `new_listings`, `median_dom`, `avg_sale_to_list`, `sold_above_list`, `price_drops`; drop any that genuinely don't exist and record that in the report).
- Write `data/raw/schema_report.md`: per source — confirmed columns, observed dtypes, verification date, and any expected column that was missing.
- On later runs, if a live header no longer matches `schema:`, raise with a diff (expected vs found).

**Accept when:**
- [ ] `schema_report.md` exists and lists real (not guessed) headers for all 3 sources
- [ ] `config.yaml` `schema:` section is populated from the live files
- [ ] Total bytes downloaded in this task < 10 MB (headers only, not the full Redfin file)
- [ ] Corrupting one expected name in `config.yaml` and re-running produces a loud diff error (then restore it)

**Verify:** `python -m pipeline verify-schema`
**Commit:** `verify-schema: live schema confirmation and drift check`

---

## T2 — download

**Do:** Full downloads into `data/raw/` (Redfin gz, Zillow CSV, FRED JSON per series in `data/raw/fred/`). Skip cached files unless `--force`. Stream the Redfin file to disk (no loading into memory). Write `data/raw/manifest.json`: per file — URL, timestamp, size bytes, sha256.

**Accept when:**
- [ ] All raw files present; manifest lists each with nonzero size and hash
- [ ] Re-running without `--force` downloads nothing (log says "cached")
- [ ] `FRED_API_KEY` read from env only; helpful error if unset

**Verify:** `python -m pipeline download && python -m pipeline download` (second run all-cached)
**Commit:** `download: cached raw pulls with manifest`

---

## T3 — clean

**Do:** One tidy parquet per source in `data/interim/` (`redfin.parquet`, `zillow.parquet`, `fred.parquet`), all keyed on `zip` (5-char zero-padded string) and/or `month` (month-begin date).
- Redfin: read gz in chunks with explicit dtypes and only schema-mapped columns; filter property type per config; keep monthly periods (if only weekly/4-week windows exist, aggregate to calendar month and document the rule in code + README); parse dates; extract 5-digit zip from the region field; drop null `median_sale_price`; deduplicate (zip, month); add boolean `low_volume` = homes_sold < threshold (flag, don't drop).
- Zillow: melt wide→long to (zip, month, zhvi); same zip/month conventions.
- FRED: all series to monthly (weekly series → monthly mean); one row per month, one column per series; document any forward-fill.

**Accept when:**
- [ ] All three parquets exist, keyed consistently; spot-check: a known zip (e.g., 90210) appears in redfin and zillow with same format
- [ ] Log shows rows in → rows out per filter step for Redfin
- [ ] `low_volume` present; no rows dropped for low volume
- [ ] `tests/test_clean.py` passes: zip padding, dedup, low-volume flagging (synthetic fixtures, no network)

**Verify:** `python -m pipeline clean && pytest tests/test_clean.py`
**Commit:** `clean: tidy per-source parquets with low-volume flag`

---

## T4 — join

**Do:** Inner join redfin+zillow on (zip, month); left join fred on month. Output `data/interim/joined.parquet`. Compute and log join coverage: % of redfin (zip, month) rows matched by zillow, counts before/after each join. Append a "Join coverage" section to `data/data_quality_report.md` (create file if absent).

**Accept when:**
- [ ] `joined.parquet` exists; macro columns are non-null for all months within FRED's range
- [ ] Coverage stats printed and written to `data_quality_report.md`
- [ ] If coverage < 50%, the stage raises (misaligned keys), per fail-loudly

**Verify:** `python -m pipeline join`
**Commit:** `join: merged zip-month panel with coverage report`

---

## T5 — featurize

**Do:** From `joined.parquet`, build `data/processed/features.parquet`. Feature groups (all computed with only data ≤ month `t`):
- Levels: median_sale_price, inventory, median_dom, avg_sale_to_list, zhvi, each macro series
- Momentum: 1/3/6/12-month % change of median_sale_price and zhvi
- Supply/demand: new_listings ÷ homes_sold; 3-month change in inventory
- Macro deltas: 3-month change in mortgage rate and unemployment
- Calendar: month-of-year integer
- Label: `target = median_sale_price[t+3] / median_sale_price[t] − 1` (as %). Rows within 3 months of data end keep `target = NaN` (live prediction set). Flag `target_outlier` where |target| > 50% (winsorize threshold in config; flag, don't drop).
Also write `feature_dictionary.md`: every column — definition, formula, source.

**Accept when:**
- [ ] `tests/test_featurize.py` passes, including the two critical tests: (a) on a synthetic series with known prices, the label at `t` exactly equals the change realized at `t+3`; (b) a lookahead audit — shifting all data after month `t` must not change any feature value at `t`
- [ ] Rows near the data end have NaN targets and are retained
- [ ] `feature_dictionary.md` covers every column in the parquet (test asserts set equality)

**Verify:** `python -m pipeline featurize && pytest tests/test_featurize.py`
**Commit:** `featurize: leakage-safe features and 3-month label`

---

## T6 — split

**Do:** Add a `split` column to `features.parquet` (or sidecar file): `test` = most recent 6 labeled months; `train` = labeled rows before that. Implement `rolling_cv(n_folds)` utility returning (train_idx, val_idx) pairs where every fold has a ≥3-month gap between train end and validation start, and train windows are capped at 36 months.

**Accept when:**
- [ ] `tests/test_split.py` passes: holdout is exactly the last 6 labeled months; every CV fold's gap ≥ 3 months; no train window exceeds 36 months; train/test are disjoint
- [ ] Unlabeled (NaN-target) rows are in neither split (they're `live`)

**Verify:** `python -m pipeline split && pytest tests/test_split.py`
**Commit:** `split: temporal holdout and gapped rolling CV`

---

## T7 — end-to-end run + data quality report

**Do:** `python -m pipeline all` runs T1–T6 stages in order. Finalize `data/data_quality_report.md` with: date range covered, zip count, row count, join coverage, % low-volume rows, % outlier targets, missing-value summary per feature, and any schema discrepancies found in T1. This report feeds the "Data limitations" slide of the course report — be honest, not flattering.

**Accept when:**
- [ ] Fresh clone + `.env` + `pipeline all` produces `features.parquet` with no manual steps
- [ ] `pytest` (whole suite) passes
- [ ] `data_quality_report.md` contains real numbers from the actual run

**Verify:** `python -m pipeline all && pytest`
**Commit:** `pipeline: end-to-end run with data quality report`

---

## T8 — README

**Do:** Write `README.md`: project one-liner, architecture diagram (ASCII fine), setup (venv, requirements, FRED key via `.env`), how to run each stage and `all`, expected runtime and disk use, where outputs land, how to run tests, and a short "Data notes" section pointing to `feature_dictionary.md` and `data_quality_report.md`.

**Accept when:**
- [ ] Following the README verbatim in a clean environment reproduces T7's result
- [ ] No secrets or absolute local paths appear anywhere in the repo

**Verify:** read-through + `git grep -i "api_key\|/Users\|/home/"` returns nothing sensitive
**Commit:** `docs: reproducible README`

---

## Out of scope

Model training, tuning, SHAP, confidence intervals, Streamlit. The pipeline ends at `features.parquet` + splits + reports.
