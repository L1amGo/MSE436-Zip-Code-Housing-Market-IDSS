# Feature dictionary

Every column of `data/processed/features.parquet`. All features at month `t`
use only information available at `t`; only `target` looks forward.

| column | definition | formula | source |
|---|---|---|---|
| `zip` | 5-char zero-padded zip code (panel key) | extracted from Redfin region field | Redfin |
| `month` | calendar month, month-begin date (panel key) | Redfin 90-day window end month | Redfin |
| `median_sale_price` | median sale price, trailing-90-day window ending at t ($) | level | Redfin |
| `homes_sold` | homes sold, trailing-90-day window (count) | level | Redfin |
| `inventory` | active listings at end of window | level | Redfin |
| `new_listings` | new listings in window | level | Redfin |
| `median_dom` | median days on market | level | Redfin |
| `avg_sale_to_list` | average sale-to-list price ratio | level | Redfin |
| `sold_above_list` | share of sales above list price | level | Redfin |
| `low_volume` | homes_sold below config low_volume_threshold (flag, rows kept) | homes_sold < threshold | derived (Redfin) |
| `zhvi` | Zillow Home Value Index, smoothed + seasonally adjusted ($) | level | Zillow |
| `MORTGAGE30US` | 30-year fixed mortgage rate, monthly mean of weekly obs (%) | level (national, monthly) | FRED |
| `UNRATE` | US unemployment rate (%) | level (national, monthly) | FRED |
| `CPIAUCSL` | CPI, all urban consumers (index) | level (national, monthly) | FRED |
| `HOUST` | housing starts (thousands, SAAR) | level (national, monthly) | FRED |
| `price_mom_1m` | 1-month % change in median sale price | price[t] / price[t-1] - 1 | derived (Redfin) |
| `price_mom_3m` | 3-month % change in median sale price | price[t] / price[t-3] - 1 | derived (Redfin) |
| `price_mom_6m` | 6-month % change in median sale price | price[t] / price[t-6] - 1 | derived (Redfin) |
| `price_mom_12m` | 12-month % change in median sale price | price[t] / price[t-12] - 1 | derived (Redfin) |
| `zhvi_mom_1m` | 1-month % change in ZHVI | zhvi[t] / zhvi[t-1] - 1 | derived (Zillow) |
| `zhvi_mom_3m` | 3-month % change in ZHVI | zhvi[t] / zhvi[t-3] - 1 | derived (Zillow) |
| `zhvi_mom_6m` | 6-month % change in ZHVI | zhvi[t] / zhvi[t-6] - 1 | derived (Zillow) |
| `zhvi_mom_12m` | 12-month % change in ZHVI | zhvi[t] / zhvi[t-12] - 1 | derived (Zillow) |
| `listings_to_sales` | supply/demand pressure | new_listings / homes_sold (inf -> NaN) | derived (Redfin) |
| `inventory_chg_3m` | 3-month % change in inventory | inventory[t] / inventory[t-3] - 1 | derived (Redfin) |
| `mortgage_delta_3m` | 3-month change in mortgage rate (percentage points) | rate[t] - rate[t-3] | derived (FRED) |
| `unrate_delta_3m` | 3-month change in unemployment rate (percentage points) | unrate[t] - unrate[t-3] | derived (FRED) |
| `month_of_year` | calendar month integer 1-12 (seasonality) | month(t) | derived |
| `target` | label: 3-month-ahead % change in median sale price (fraction; NaN = live row) | price[t+3] / price[t] - 1 | derived (Redfin, future) |
| `target_outlier` | |target| exceeds config target_outlier_threshold (flag, rows kept) | |target| > threshold | derived |
| `split` | temporal split (added by the split stage) | test = last holdout_months labeled months; train = labeled rows before; live = NaN target | derived |
