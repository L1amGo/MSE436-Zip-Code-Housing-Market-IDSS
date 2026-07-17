"""Stage: clean — one tidy parquet per source in data/interim/.

Outputs are keyed on `zip` (5-char zero-padded string) and/or `month`
(month-begin date):
  - redfin.parquet: (zip, month) + market metrics + low_volume flag
  - zillow.parquet: (zip, month, zhvi)
  - fred.parquet:   (month) + one column per series

Monthly aggregation rules:
  - Redfin's zip-code tracker contains ONLY 90-day rolling windows (verified
    2026-07: all 9.7M rows have PERIOD_DURATION=90; no monthly rows exist).
    Windows roll monthly per zip (period_begin = 1st of month m, period_end =
    last day of month m+2), so each well-formed window is assigned to the
    calendar month it ENDS in: the row for (zip, t) holds trailing-90-day
    stats fully observable at the end of month t — no future information.
    Malformed windows are dropped and counted in the log. Consequence: levels
    are 3-month trailing aggregates, smoother than true single-month values.
  - Zillow month columns are month-END dates; they are shifted to the
    month-begin convention used everywhere else.
  - FRED weekly series are averaged to calendar months (monthly series pass
    through unchanged). Interior gap months (e.g. UNRATE's Oct-2025 shutdown
    hole) are forward-filled with the last published value and logged;
    trailing months FRED has not published yet stay NaN.

Every filter step logs rows in -> rows out. Low-volume rows are flagged
(`low_volume`), never dropped.
"""

import re
from pathlib import Path

import pandas as pd

from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("clean")

REDFIN_NUMERIC = [
    "median_sale_price",
    "homes_sold",
    "inventory",
    "new_listings",
    "median_dom",
    "avg_sale_to_list",
    "sold_above_list",
    "price_drops",
]
REDFIN_CHUNKSIZE = 1_000_000


def _log_step(source: str, step: str, before: int, after: int) -> None:
    log.info("%s: %-38s %10d -> %10d rows (-%d)", source, step, before, after, before - after)


def extract_zip(region: pd.Series) -> pd.Series:
    """'Zip Code: 90210' / '210' -> 5-char zero-padded string; no digits -> NaN."""
    return region.astype(str).str.extract(r"(\d+)\s*$", expand=False).str.zfill(5)


def tidy_redfin(df: pd.DataFrame, low_volume_threshold: int) -> pd.DataFrame:
    """Tidy a logical-named Redfin frame (already property-type filtered).

    Steps: keep well-formed 90-day windows and assign each to the month it
    ends in (see module docstring), extract zip, drop null median_sale_price,
    deduplicate (zip, month), flag low volume.
    """
    n = len(df)
    well_formed = (df["period_begin"].dt.day == 1) & (
        df["period_end"] == df["period_begin"] + pd.offsets.MonthEnd(3)
    )
    df = df[well_formed]
    _log_step("redfin", "well-formed 90-day windows only", n, len(df))

    month = df["period_end"].dt.to_period("M").dt.to_timestamp()
    n = len(df)
    df = df.assign(zip=extract_zip(df["region_zip"]), month=month.values)
    df = df.dropna(subset=["zip"])
    _log_step("redfin", "zip extracted from region", n, len(df))

    n = len(df)
    df = df.dropna(subset=["median_sale_price"])
    _log_step("redfin", "null median_sale_price dropped", n, len(df))

    n = len(df)
    # Sort on every value column so dedup picks the same winner regardless of
    # input row order (stable sort alone would keep whichever came last).
    df = df.sort_values(["zip", "month", "period_end", *REDFIN_NUMERIC], kind="mergesort")
    df = df.drop_duplicates(subset=["zip", "month"], keep="last")
    _log_step("redfin", "dedup (zip, month)", n, len(df))

    df = df.assign(low_volume=df["homes_sold"].fillna(0) < low_volume_threshold)
    df = df[["zip", "month", *REDFIN_NUMERIC, "low_volume"]]
    return df.sort_values(["zip", "month"], kind="mergesort").reset_index(drop=True)


def tidy_zillow(df: pd.DataFrame, zip_col: str, month_col_regex: str) -> pd.DataFrame:
    """Wide ZHVI -> long (zip, month, zhvi); month-end column names -> month-begin."""
    month_cols = [c for c in df.columns if re.fullmatch(month_col_regex, str(c))]
    if not month_cols:
        raise RuntimeError("Zillow frame has no month columns — wrong file or layout change")
    out = df.melt(
        id_vars=[zip_col], value_vars=month_cols, var_name="month_end", value_name="zhvi"
    )
    out["zip"] = out[zip_col].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
    out["month"] = (
        pd.to_datetime(out["month_end"]).dt.to_period("M").dt.to_timestamp()
    )
    out["zhvi"] = pd.to_numeric(out["zhvi"], errors="coerce")
    n = len(out)
    out = out.dropna(subset=["zip", "zhvi"])
    _log_step("zillow", "null zhvi dropped (pre-coverage months)", n, len(out))
    n = len(out)
    out = out.sort_values(["zip", "month"], kind="mergesort").drop_duplicates(
        subset=["zip", "month"], keep="last"
    )
    _log_step("zillow", "dedup (zip, month)", n, len(out))
    return out[["zip", "month", "zhvi"]].reset_index(drop=True)


def tidy_fred(observations: dict[str, list[dict]], frequencies: dict[str, str]) -> pd.DataFrame:
    """{series: FRED observation dicts} -> one row per month, one column per series.

    Weekly (and any sub-monthly) series are averaged over the calendar month;
    monthly series pass through. FRED encodes missing values as '.'.
    No forward-fill: months without a published value stay NaN.
    """
    monthly = {}
    for series, obs in observations.items():
        s = pd.DataFrame(obs)[["date", "value"]]
        s["value"] = pd.to_numeric(s["value"].replace(".", None), errors="coerce")
        s["month"] = pd.to_datetime(s["date"]).dt.to_period("M").dt.to_timestamp()
        agg = s.dropna(subset=["value"]).groupby("month")["value"].mean()
        log.info(
            "fred %s (%s): %d obs -> %d months", series, frequencies.get(series), len(s), len(agg)
        )
        monthly[series] = agg
    out = pd.DataFrame(monthly).sort_index()
    # Forward-fill INTERIOR gaps only (e.g. the Oct-2025 shutdown hole in
    # UNRATE): a gap month reuses the last published value, which was known at
    # the time — leakage-safe. Trailing months beyond a series' latest
    # publication stay NaN; they are unknown, not missing.
    for col in out.columns:
        s = out[col]
        interior = s.loc[s.first_valid_index() : s.last_valid_index()]
        gaps = interior[interior.isna()].index
        if len(gaps):
            log.info(
                "fred %s: forward-filling %d interior gap month(s): %s",
                col,
                len(gaps),
                [g.strftime("%Y-%m") for g in gaps],
            )
            filled = s.ffill()
            filled.loc[filled.index > s.last_valid_index()] = float("nan")
            out[col] = filled
    out.index.name = "month"
    return out.reset_index()


def _read_redfin_chunks(path: Path, config: dict) -> pd.DataFrame:
    """Stream the raw gz in chunks, keeping only schema columns and the
    configured property type; returns a logical-named frame."""
    cols = config["schema"]["redfin"]["columns"]  # logical -> actual
    actual_to_logical = {v: k for k, v in cols.items()}
    property_type = config["params"]["property_type"]
    metro_filter = config["params"]["dev_metro_filter"] or []

    usecols = list(cols.values())
    if metro_filter:
        usecols.append("PARENT_METRO_REGION")

    parts = []
    rows_in = kept_property = kept_metro = 0
    reader = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=lambda c: c in usecols,
        dtype={cols["region_zip"]: str, cols["property_type"]: str},
        parse_dates=[cols["period_begin"], cols["period_end"]],
        chunksize=REDFIN_CHUNKSIZE,
    )
    for chunk in reader:
        rows_in += len(chunk)
        chunk = chunk[chunk[cols["property_type"]] == property_type]
        kept_property += len(chunk)
        if metro_filter:
            chunk = chunk[chunk["PARENT_METRO_REGION"].isin(metro_filter)]
        kept_metro += len(chunk)
        parts.append(chunk.rename(columns=actual_to_logical))
        log.info("redfin: read %d rows so far ...", rows_in)
    _log_step("redfin", f"property_type == {property_type!r}", rows_in, kept_property)
    if metro_filter:
        _log_step("redfin", f"dev metro filter {metro_filter}", kept_property, kept_metro)
    df = pd.concat(parts, ignore_index=True)
    for c in REDFIN_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run(config: dict, force: bool = False) -> None:
    raw = REPO_ROOT / config["paths"]["raw"]
    interim = REPO_ROOT / config["paths"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)

    redfin_gz = raw / Path(config["sources"]["redfin"]["url"]).name
    zillow_csv = raw / "zillow_zhvi_zip.csv"
    fred_dir = raw / "fred"
    for p in (redfin_gz, zillow_csv, fred_dir):
        if not p.exists():
            raise RuntimeError(f"Raw input missing: {p}. Run `python -m pipeline download` first.")

    if not config.get("schema"):
        raise RuntimeError("config.yaml `schema:` is empty. Run `python -m pipeline verify-schema` first.")

    # Redfin
    redfin = tidy_redfin(
        _read_redfin_chunks(redfin_gz, config), config["params"]["low_volume_threshold"]
    )
    redfin.to_parquet(interim / "redfin.parquet", index=False)
    log.info("wrote %s (%d rows, %d zips)", interim / "redfin.parquet", len(redfin), redfin["zip"].nunique())

    # Zillow
    zcfg = config["schema"]["zillow"]
    zillow_wide = pd.read_csv(zillow_csv, dtype={zcfg["columns"]["region_zip"]: str})
    log.info("zillow: %d rows x %d cols (wide)", *zillow_wide.shape)
    zillow = tidy_zillow(zillow_wide, zcfg["columns"]["region_zip"], zcfg["month_col_regex"])
    zillow.to_parquet(interim / "zillow.parquet", index=False)
    log.info("wrote %s (%d rows, %d zips)", interim / "zillow.parquet", len(zillow), zillow["zip"].nunique())

    # FRED
    import json

    observations = {}
    for sid in config["sources"]["fred"]["series"]:
        payload = json.loads((fred_dir / f"{sid}.json").read_text(encoding="utf-8"))
        observations[sid] = payload["observations"]
    fred = tidy_fred(observations, config["schema"]["fred"]["frequencies"])
    fred.to_parquet(interim / "fred.parquet", index=False)
    log.info("wrote %s (%d months)", interim / "fred.parquet", len(fred))

    for name, df in (("redfin", redfin), ("zillow", zillow), ("fred", fred)):
        if df.empty:
            raise RuntimeError(f"clean produced an empty {name} output — check filters/schema")
