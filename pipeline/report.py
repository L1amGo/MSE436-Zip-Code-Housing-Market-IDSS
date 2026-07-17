"""Stage: report — final data_quality_report.md from the finished pipeline outputs.

Rewrites data/data_quality_report.md in full (deterministic for a given
features.parquet), so repeated runs never stack sections. This report feeds the
"Data limitations" slide of the course report — it states what the data
actually looks like, including the unflattering parts.
"""

import datetime

import pandas as pd

from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("report")


def run(config: dict, force: bool = False) -> None:
    interim = REPO_ROOT / config["paths"]["interim"]
    processed = REPO_ROOT / config["paths"]["processed"]
    features_path = processed / "features.parquet"
    if not features_path.exists():
        raise RuntimeError(f"{features_path} missing. Run the pipeline stages first.")
    f = pd.read_parquet(features_path)
    if "split" not in f.columns:
        raise RuntimeError("features.parquet has no split column. Run `python -m pipeline split`.")
    n_redfin = len(pd.read_parquet(interim / "redfin.parquet", columns=["zip"]))

    labeled = f["target"].notna()
    coverage = len(f) / n_redfin
    low_volume_pct = 100 * f["low_volume"].mean()
    outlier_pct = 100 * f.loc[labeled, "target_outlier"].mean()
    splits = f["split"].value_counts()

    missing = (f.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    lines = [
        "# Data quality report",
        "",
        f"_Generated {datetime.date.today().isoformat()} by `python -m pipeline report`._",
        "",
        "## Panel summary",
        "",
        f"- Date range: **{f['month'].min():%Y-%m} to {f['month'].max():%Y-%m}** (monthly)",
        f"- Zip codes: **{f['zip'].nunique():,}**",
        f"- Rows: **{len(f):,}** ({splits.get('train', 0):,} train / {splits.get('test', 0):,} test / {splits.get('live', 0):,} live)",
        f"- Labeled rows: **{int(labeled.sum()):,}** ({100 * labeled.mean():.1f}%)",
        "",
        "## Join coverage",
        "",
        f"- Redfin (zip, month) rows: **{n_redfin:,}**",
        f"- Survived the Zillow ZHVI inner join: **{len(f):,}** (**{100 * coverage:.1f}%**)",
        "- Unmatched rows are mostly small/rural zips where Zillow does not publish ZHVI;"
        " dropping them biases the panel toward denser markets.",
        "",
        "## Flags (rows kept, not dropped)",
        "",
        f"- Low-volume rows (homes_sold < {config['params']['low_volume_threshold']}): "
        f"**{low_volume_pct:.1f}%** — zip-month medians on thin volume are noisy.",
        f"- Outlier targets (|target| > {config['params']['target_outlier_threshold']:.0%} "
        f"over {config['params']['label_horizon_months']} months): **{outlier_pct:.1f}%** of labeled rows.",
        "",
        "## Missing values per column (only columns with any missing)",
        "",
        "| column | % missing |",
        "|---|---|",
        *[f"| `{c}` | {v:.1f}% |" for c, v in missing.items()],
        "",
        "Momentum lags are structurally missing early in each zip's history "
        "(a 12-month change needs 12 months of history); `target` is missing for "
        "the live prediction set by construction.",
        "",
        "## Known data caveats",
        "",
        "- **Redfin publishes only 90-day rolling windows at zip level** (no true monthly"
        " rows exist in the file). Each window is assigned to the month it ends in, so"
        " all Redfin-derived levels are trailing-3-month aggregates — smoother and more"
        " autocorrelated than single-month values, and the 3-month label partially"
        " overlaps adjacent windows.",
        "- **Oct-2025 government shutdown hole:** UNRATE and CPIAUCSL have no published"
        " Oct-2025 value; the pipeline forward-fills that single interior month from"
        " Sep-2025 (leakage-safe, documented in `pipeline/clean.py`).",
        "- Zillow ZHVI months are published as month-end dates and are shifted to the"
        " month-begin convention used everywhere in this pipeline.",
        "- **`price_drops` is dropped**: Redfin includes the column in the header but"
        " publishes it as NA on every zip-level row (verified across all 9.7M raw rows),"
        " so it carries no signal.",
        "- T1 schema verification found **no missing expected columns** in any source"
        " (see `data/raw/schema_report.md`).",
        "",
    ]
    out = REPO_ROOT / "data" / "data_quality_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", out.relative_to(REPO_ROOT))
