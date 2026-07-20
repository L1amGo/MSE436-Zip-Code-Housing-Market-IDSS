# Data quality report

_Generated 2026-07-20 by `python -m pipeline report`._

## Panel summary

- Date range: **2012-03 to 2026-05** (monthly)
- Zip codes: **20,071**
- Rows: **2,984,149** (2,727,408 train / 107,776 test / 148,965 live)
- Labeled rows: **2,835,184** (95.0%)

## Join coverage

- Redfin (zip, month) rows: **3,292,619**
- Survived the Zillow ZHVI inner join: **2,984,149** (**90.6%**)
- Unmatched rows are mostly small/rural zips where Zillow does not publish ZHVI; dropping them biases the panel toward denser markets.

## Flags (rows kept, not dropped)

- Low-volume rows (homes_sold < 10): **31.1%** — zip-month medians on thin volume are noisy.
- Outlier targets (|target| > 50% over 3 months): **10.0%** of labeled rows.

## Missing values per column (only columns with any missing)

| column | % missing |
|---|---|
| `zhvi_mom_12m` | 10.3% |
| `price_mom_12m` | 10.3% |
| `inventory_chg_3m` | 8.7% |
| `price_mom_6m` | 6.8% |
| `zhvi_mom_6m` | 6.8% |
| `unrate_delta_3m` | 5.0% |
| `target` | 5.0% |
| `price_mom_3m` | 5.0% |
| `zhvi_mom_3m` | 5.0% |
| `mortgage_delta_3m` | 5.0% |
| `inventory` | 3.9% |
| `avg_sale_to_list` | 3.4% |
| `new_listings` | 2.7% |
| `listings_to_sales` | 2.7% |
| `price_mom_1m` | 2.2% |
| `zhvi_mom_1m` | 2.2% |
| `sold_above_list` | 1.0% |
| `median_dom` | 0.5% |

Momentum lags are structurally missing early in each zip's history (a 12-month change needs 12 months of history); `target` is missing for the live prediction set by construction.

## Known data caveats

- **Redfin publishes only 90-day rolling windows at zip level** (no true monthly rows exist in the file). Each window is assigned to the month it ends in, so all Redfin-derived levels are trailing-3-month aggregates — smoother and more autocorrelated than single-month values, and the 3-month label partially overlaps adjacent windows.
- **Oct-2025 government shutdown hole:** UNRATE and CPIAUCSL have no published Oct-2025 value; the pipeline forward-fills that single interior month from Sep-2025 (leakage-safe, documented in `pipeline/clean.py`).
- Zillow ZHVI months are published as month-end dates and are shifted to the month-begin convention used everywhere in this pipeline.
- **`price_drops` is dropped**: Redfin includes the column in the header but publishes it as NA on every zip-level row (verified across all 9.7M raw rows), so it carries no signal.
- T1 schema verification found **no missing expected columns** in any source (see `data/raw/schema_report.md`).
