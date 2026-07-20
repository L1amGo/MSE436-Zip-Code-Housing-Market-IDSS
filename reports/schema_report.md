# Schema report

Verified against live sources on **2026-07-20** (headers only, no full downloads).

## Redfin zip-code market tracker (gzipped TSV)

Columns found (58):

| column | observed dtype (from first data row) |
|---|---|
| `PERIOD_BEGIN` | date |
| `PERIOD_END` | date |
| `PERIOD_DURATION` | int |
| `REGION_TYPE` | str |
| `REGION_TYPE_ID` | int |
| `TABLE_ID` | int |
| `IS_SEASONALLY_ADJUSTED` | str |
| `REGION` | str |
| `CITY` | null |
| `STATE` | str |
| `STATE_CODE` | str |
| `PROPERTY_TYPE` | str |
| `PROPERTY_TYPE_ID` | int |
| `MEDIAN_SALE_PRICE` | int |
| `MEDIAN_SALE_PRICE_MOM` | float |
| `MEDIAN_SALE_PRICE_YOY` | float |
| `MEDIAN_LIST_PRICE` | int |
| `MEDIAN_LIST_PRICE_MOM` | float |
| `MEDIAN_LIST_PRICE_YOY` | float |
| `MEDIAN_PPSF` | float |
| `MEDIAN_PPSF_MOM` | float |
| `MEDIAN_PPSF_YOY` | float |
| `MEDIAN_LIST_PPSF` | float |
| `MEDIAN_LIST_PPSF_MOM` | int |
| `MEDIAN_LIST_PPSF_YOY` | float |
| `HOMES_SOLD` | int |
| `HOMES_SOLD_MOM` | float |
| `HOMES_SOLD_YOY` | float |
| `PENDING_SALES` | int |
| `PENDING_SALES_MOM` | float |
| `PENDING_SALES_YOY` | float |
| `NEW_LISTINGS` | int |
| `NEW_LISTINGS_MOM` | float |
| `NEW_LISTINGS_YOY` | float |
| `INVENTORY` | int |
| `INVENTORY_MOM` | float |
| `INVENTORY_YOY` | float |
| `MONTHS_OF_SUPPLY` | str |
| `MONTHS_OF_SUPPLY_MOM` | str |
| `MONTHS_OF_SUPPLY_YOY` | str |
| `MEDIAN_DOM` | int |
| `MEDIAN_DOM_MOM` | float |
| `MEDIAN_DOM_YOY` | int |
| `AVG_SALE_TO_LIST` | float |
| `AVG_SALE_TO_LIST_MOM` | float |
| `AVG_SALE_TO_LIST_YOY` | float |
| `SOLD_ABOVE_LIST` | float |
| `SOLD_ABOVE_LIST_MOM` | float |
| `SOLD_ABOVE_LIST_YOY` | float |
| `PRICE_DROPS` | str |
| `PRICE_DROPS_MOM` | str |
| `PRICE_DROPS_YOY` | str |
| `OFF_MARKET_IN_TWO_WEEKS` | float |
| `OFF_MARKET_IN_TWO_WEEKS_MOM` | float |
| `OFF_MARKET_IN_TWO_WEEKS_YOY` | float |
| `PARENT_METRO_REGION` | str |
| `PARENT_METRO_REGION_METRO_CODE` | int |
| `LAST_UPDATED` | str |

All 12 expected logical columns matched a live column.

## Zillow ZHVI zip-level (wide CSV)

- Columns found: 327 total.
- Static columns: ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'State', 'City', 'Metro', 'CountyName']
- Month columns: `2000-01-31` … `2026-06-30` (one column per month).
- Zip column: `RegionName` (dtypes not sampled — header-only fetch).

## FRED series

| series | title | native frequency | latest obs | dtype |
|---|---|---|---|---|
| `MORTGAGE30US` | 30-Year Fixed Rate Mortgage Average in the United States | W | 2026-07-16 | float |
| `UNRATE` | Unemployment Rate | M | 2026-06-01 | float |
| `CPIAUCSL` | Consumer Price Index for All Urban Consumers: All Items in U.S. City Average | M | 2026-06-01 | float |
| `HOUST` | New Privately-Owned Housing Units Started: Total Units | M | 2026-06-01 | float |

Total bytes downloaded during verification: 139,262
