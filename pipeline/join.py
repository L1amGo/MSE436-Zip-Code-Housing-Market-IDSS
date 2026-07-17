"""Stage: join — merge the tidy per-source parquets into one zip-month panel.

Inner join redfin+zillow on (zip, month); left join fred on month. Logs and
reports join coverage; raises if the zillow match rate falls below
params.join_coverage_min (a sign of misaligned keys) or if any macro column is
null for a month inside FRED's published range.
"""

import datetime

import pandas as pd

from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("join")


def _append_report(section: str) -> None:
    """Idempotent append: an existing Join coverage section is replaced, so
    repeated runs never stack duplicates."""
    report = REPO_ROOT / "data" / "data_quality_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    text = report.read_text(encoding="utf-8") if report.exists() else "# Data quality report\n"
    head, marker, _ = text.partition("\n## Join coverage")
    report.write_text(head + section, encoding="utf-8")
    log.info("wrote Join coverage section to %s", report.relative_to(REPO_ROOT))


def run(config: dict, force: bool = False) -> None:
    interim = REPO_ROOT / config["paths"]["interim"]
    for name in ("redfin", "zillow", "fred"):
        if not (interim / f"{name}.parquet").exists():
            raise RuntimeError(
                f"{interim / f'{name}.parquet'} missing. Run `python -m pipeline clean` first."
            )
    redfin = pd.read_parquet(interim / "redfin.parquet")
    zillow = pd.read_parquet(interim / "zillow.parquet")
    fred = pd.read_parquet(interim / "fred.parquet")

    n_redfin = len(redfin)
    joined = redfin.merge(zillow, on=["zip", "month"], how="inner", validate="one_to_one")
    coverage = len(joined) / n_redfin if n_redfin else 0.0
    log.info(
        "redfin x zillow inner join: %d -> %d rows (%.1f%% of redfin matched)",
        n_redfin,
        len(joined),
        100 * coverage,
    )
    min_coverage = config["params"]["join_coverage_min"]
    if coverage < min_coverage:
        raise RuntimeError(
            f"Zillow matched only {coverage:.1%} of redfin (zip, month) rows "
            f"(< {min_coverage:.0%}). Keys are likely misaligned — check zip padding "
            "and month conventions in the clean stage."
        )

    n_before_fred = len(joined)
    joined = joined.merge(fred, on="month", how="left", validate="many_to_one")
    assert len(joined) == n_before_fred, "left join on month must not change row count"
    log.info("+ fred left join on month: %d rows, %d cols", *joined.shape)

    series = config["sources"]["fred"]["series"]
    in_range = joined["month"].between(fred["month"].min(), fred["month"].max())
    for sid in series:
        n_null = int(joined.loc[in_range, sid].isna().sum())
        if n_null:
            raise RuntimeError(
                f"Macro column {sid} is null for {n_null} rows within FRED's "
                f"published range — check the clean stage's monthly aggregation."
            )

    out = interim / "joined.parquet"
    joined = joined.sort_values(["zip", "month"], kind="mergesort").reset_index(drop=True)
    joined.to_parquet(out, index=False)
    log.info("wrote %s (%d rows, %d zips)", out.relative_to(REPO_ROOT), len(joined), joined["zip"].nunique())

    today = datetime.date.today().isoformat()
    _append_report(
        "\n## Join coverage\n\n"
        f"_Run date: {today}_\n\n"
        f"- Redfin (zip, month) rows in: **{n_redfin:,}**\n"
        f"- Matched by Zillow ZHVI (inner join): **{len(joined):,}** "
        f"(**{100 * coverage:.1f}%** coverage)\n"
        f"- Redfin rows without a ZHVI value (dropped): **{n_redfin - len(joined):,}**\n"
        f"- FRED left join on month: row count unchanged ({len(joined):,}); "
        f"macro columns non-null for all months within FRED's range "
        f"({fred['month'].min():%Y-%m} to {fred['month'].max():%Y-%m}).\n"
    )
