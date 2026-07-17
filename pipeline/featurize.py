"""Stage: featurize — leakage-safe features and the 3-month label.

Every feature at (zip, month t) is computed only from data at months <= t.
Lags are taken by merging on explicit (zip, month-k) keys, never by positional
shifting, so gaps in a zip's history can never smuggle in the wrong month.

The label is target = median_sale_price[t+3] / median_sale_price[t] - 1,
expressed as a fraction (0.05 = +5%). Rows within 3 months of the data end
have no realized future price yet; they keep target = NaN and stay in the
output as the live prediction set.
"""

import pandas as pd

from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("featurize")

MOMENTUM_LAGS = [1, 3, 6, 12]


def _lagged(df: pd.DataFrame, col: str, months: int, keys: list[str]) -> pd.Series:
    """Value of `col` exactly `months` months earlier for the same keys (NaN if absent)."""
    shifted = df[[*keys, "month", col]].copy()
    shifted["month"] = shifted["month"] + pd.DateOffset(months=months)
    shifted = shifted.rename(columns={col: "_lagged"})
    merged = df[[*keys, "month"]].merge(shifted, on=[*keys, "month"], how="left")
    return merged["_lagged"].set_axis(df.index)


def _future(df: pd.DataFrame, col: str, months: int, keys: list[str]) -> pd.Series:
    """Value of `col` exactly `months` months LATER (label use only — never a feature)."""
    shifted = df[[*keys, "month", col]].copy()
    shifted["month"] = shifted["month"] - pd.DateOffset(months=months)
    shifted = shifted.rename(columns={col: "_future"})
    merged = df[[*keys, "month"]].merge(shifted, on=[*keys, "month"], how="left")
    return merged["_future"].set_axis(df.index)


def build_features(joined: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Pure feature builder: tidy joined panel in -> modeling frame out."""
    series = config["sources"]["fred"]["series"]
    horizon = config["params"]["label_horizon_months"]
    outlier_threshold = config["params"]["target_outlier_threshold"]

    df = joined.sort_values(["zip", "month"], kind="mergesort").reset_index(drop=True)
    out = df.copy()

    # Momentum: % change of price and zhvi over 1/3/6/12 months.
    for col, prefix in (("median_sale_price", "price"), ("zhvi", "zhvi")):
        for k in MOMENTUM_LAGS:
            out[f"{prefix}_mom_{k}m"] = df[col] / _lagged(df, col, k, ["zip"]) - 1

    # Supply/demand.
    out["listings_to_sales"] = (df["new_listings"] / df["homes_sold"]).replace(
        [float("inf"), -float("inf")], pd.NA
    ).astype("float64")
    out["inventory_chg_3m"] = (df["inventory"] / _lagged(df, "inventory", 3, ["zip"]) - 1).replace(
        [float("inf"), -float("inf")], pd.NA
    ).astype("float64")

    # Macro deltas: percentage-POINT difference over 3 months (these series are
    # already rates/levels; a difference is the standard delta).
    for sid, name in (("MORTGAGE30US", "mortgage_delta_3m"), ("UNRATE", "unrate_delta_3m")):
        if sid in series:
            out[name] = df[sid] - _lagged(df, sid, 3, ["zip"])

    # Calendar.
    out["month_of_year"] = df["month"].dt.month.astype("int64")

    # Label (future information by construction — used for nothing else).
    out["target"] = _future(df, "median_sale_price", horizon, ["zip"]) / df["median_sale_price"] - 1
    out["target_outlier"] = out["target"].abs() > outlier_threshold

    return out


def _feature_dictionary(config: dict) -> dict[str, tuple[str, str, str]]:
    """column -> (definition, formula, source) for every column in features.parquet."""
    h = config["params"]["label_horizon_months"]
    d = {
        "zip": ("5-char zero-padded zip code (panel key)", "extracted from Redfin region field", "Redfin"),
        "month": ("calendar month, month-begin date (panel key)", "Redfin 90-day window end month", "Redfin"),
        "median_sale_price": ("median sale price, trailing-90-day window ending at t ($)", "level", "Redfin"),
        "homes_sold": ("homes sold, trailing-90-day window (count)", "level", "Redfin"),
        "inventory": ("active listings at end of window", "level", "Redfin"),
        "new_listings": ("new listings in window", "level", "Redfin"),
        "median_dom": ("median days on market", "level", "Redfin"),
        "avg_sale_to_list": ("average sale-to-list price ratio", "level", "Redfin"),
        "sold_above_list": ("share of sales above list price", "level", "Redfin"),
        "price_drops": ("share of listings with a price drop", "level", "Redfin"),
        "low_volume": ("homes_sold below config low_volume_threshold (flag, rows kept)", "homes_sold < threshold", "derived (Redfin)"),
        "zhvi": ("Zillow Home Value Index, smoothed + seasonally adjusted ($)", "level", "Zillow"),
        "listings_to_sales": ("supply/demand pressure", "new_listings / homes_sold (inf -> NaN)", "derived (Redfin)"),
        "inventory_chg_3m": ("3-month % change in inventory", "inventory[t] / inventory[t-3] - 1", "derived (Redfin)"),
        "month_of_year": ("calendar month integer 1-12 (seasonality)", "month(t)", "derived"),
        "target": (f"label: {h}-month-ahead % change in median sale price (fraction; NaN = live row)", f"price[t+{h}] / price[t] - 1", "derived (Redfin, future)"),
        "target_outlier": ("|target| exceeds config target_outlier_threshold (flag, rows kept)", "|target| > threshold", "derived"),
        "split": ("temporal split (added by the split stage)", "test = last holdout_months labeled months; train = labeled rows before; live = NaN target", "derived"),
    }
    for k in MOMENTUM_LAGS:
        d[f"price_mom_{k}m"] = (f"{k}-month % change in median sale price", f"price[t] / price[t-{k}] - 1", "derived (Redfin)")
        d[f"zhvi_mom_{k}m"] = (f"{k}-month % change in ZHVI", f"zhvi[t] / zhvi[t-{k}] - 1", "derived (Zillow)")
    macro_titles = {
        "MORTGAGE30US": "30-year fixed mortgage rate, monthly mean of weekly obs (%)",
        "UNRATE": "US unemployment rate (%)",
        "CPIAUCSL": "CPI, all urban consumers (index)",
        "HOUST": "housing starts (thousands, SAAR)",
    }
    for sid in config["sources"]["fred"]["series"]:
        d[sid] = (macro_titles.get(sid, "macro series"), "level (national, monthly)", "FRED")
    if "MORTGAGE30US" in config["sources"]["fred"]["series"]:
        d["mortgage_delta_3m"] = ("3-month change in mortgage rate (percentage points)", "rate[t] - rate[t-3]", "derived (FRED)")
    if "UNRATE" in config["sources"]["fred"]["series"]:
        d["unrate_delta_3m"] = ("3-month change in unemployment rate (percentage points)", "unrate[t] - unrate[t-3]", "derived (FRED)")
    return d


def write_feature_dictionary(columns: list[str], config: dict, path) -> None:
    d = _feature_dictionary(config)
    undocumented = set(columns) - set(d)
    if undocumented:
        raise RuntimeError(f"features.parquet has undocumented columns: {sorted(undocumented)}")
    lines = [
        "# Feature dictionary",
        "",
        "Every column of `data/processed/features.parquet`. All features at month `t`",
        "use only information available at `t`; only `target` looks forward.",
        "",
        "| column | definition | formula | source |",
        "|---|---|---|---|",
        *[f"| `{c}` | {d[c][0]} | {d[c][1]} | {d[c][2]} |" for c in columns],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s (%d columns)", path, len(columns))


def run(config: dict, force: bool = False) -> None:
    interim = REPO_ROOT / config["paths"]["interim"]
    processed = REPO_ROOT / config["paths"]["processed"]
    processed.mkdir(parents=True, exist_ok=True)
    src = interim / "joined.parquet"
    if not src.exists():
        raise RuntimeError(f"{src} missing. Run `python -m pipeline join` first.")

    joined = pd.read_parquet(src)
    log.info("joined panel in: %d rows x %d cols", *joined.shape)
    features = build_features(joined, config)
    n_labeled = int(features["target"].notna().sum())
    log.info(
        "features out: %d rows x %d cols (%d labeled, %d live/NaN-target, %d target outliers)",
        len(features),
        features.shape[1],
        n_labeled,
        len(features) - n_labeled,
        int(features["target_outlier"].sum()),
    )
    if n_labeled == 0:
        raise RuntimeError("No labeled rows produced — label horizon exceeds data range?")

    features.to_parquet(processed / "features.parquet", index=False)
    log.info("wrote %s", (processed / "features.parquet").relative_to(REPO_ROOT))
    # `split` is appended to the parquet by the split stage; document it now so
    # the dictionary always covers the finished file.
    write_feature_dictionary(
        [*features.columns, "split"], config, REPO_ROOT / "feature_dictionary.md"
    )
