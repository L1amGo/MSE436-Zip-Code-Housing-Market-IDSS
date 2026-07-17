"""Stage: split — temporal holdout plus a gapped rolling-CV utility.

Adds a `split` column to data/processed/features.parquet:
  - `test`:  labeled rows in the most recent `holdout_months` labeled months
  - `train`: labeled rows before that
  - `live`:  rows with NaN target (no realized future price yet) — usable for
             prediction, never for fitting or evaluation

`rolling_cv` yields (train_idx, val_idx) row-index pairs for time-series CV on
the train split: each fold validates on one month, keeps a >= `cv_gap_months`
calendar gap between the last train month and the validation month (so no
label window overlaps), and caps the train window at `train_window_months`.
"""

import pandas as pd

from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("split")


def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def assign_split(features: pd.DataFrame, holdout_months: int) -> pd.Series:
    """'train' / 'test' / 'live' per row (see module docstring)."""
    labeled = features["target"].notna()
    if not labeled.any():
        raise RuntimeError("No labeled rows — cannot assign splits.")
    labeled_months = sorted(features.loc[labeled, "month"].unique())
    if len(labeled_months) <= holdout_months:
        raise RuntimeError(
            f"Only {len(labeled_months)} labeled months; cannot hold out {holdout_months}."
        )
    test_months = set(labeled_months[-holdout_months:])
    split = pd.Series("live", index=features.index)
    split[labeled & features["month"].isin(test_months)] = "test"
    split[labeled & ~features["month"].isin(test_months)] = "train"
    return split


def rolling_cv(features: pd.DataFrame, config: dict, n_folds: int) -> list[tuple[pd.Index, pd.Index]]:
    """(train_idx, val_idx) pairs over the train split, newest fold last.

    Fold i validates on the i-th of the last `n_folds` train months; its train
    rows are the up-to-`train_window_months` months ending `cv_gap_months`
    before the validation month.
    """
    gap = config["params"]["cv_gap_months"]
    window = config["params"]["train_window_months"]
    if "split" not in features.columns:
        raise RuntimeError("features frame has no `split` column — run the split stage first.")

    train = features[features["split"] == "train"]
    train_months = sorted(pd.Index(train["month"].unique()))
    val_months = train_months[-n_folds:]
    if len(val_months) < n_folds:
        raise RuntimeError(f"Only {len(train_months)} train months for {n_folds} folds.")

    folds = []
    for v in val_months:
        v = pd.Timestamp(v)
        train_end = v - pd.DateOffset(months=gap)
        train_start = train_end - pd.DateOffset(months=window - 1)
        in_window = train["month"].between(train_start, train_end)
        train_idx, val_idx = train.index[in_window], train.index[train["month"] == v]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise RuntimeError(f"Empty fold for validation month {v:%Y-%m}.")
        folds.append((train_idx, val_idx))
    return folds


def run(config: dict, force: bool = False) -> None:
    path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} missing. Run `python -m pipeline featurize` first.")
    features = pd.read_parquet(path)
    features["split"] = assign_split(features, config["params"]["holdout_months"])

    counts = features["split"].value_counts()
    test_months = sorted(features.loc[features["split"] == "test", "month"].unique())
    log.info(
        "split: train=%d test=%d live=%d rows; holdout months %s -> %s",
        counts.get("train", 0),
        counts.get("test", 0),
        counts.get("live", 0),
        pd.Timestamp(test_months[0]).strftime("%Y-%m"),
        pd.Timestamp(test_months[-1]).strftime("%Y-%m"),
    )
    features.to_parquet(path, index=False)
    log.info("wrote %s (with split column)", path.relative_to(REPO_ROOT))

    # Smoke-check the CV utility on the real panel so a broken fold fails here,
    # not in the modeling notebook.
    folds = rolling_cv(features, config, n_folds=3)
    for train_idx, val_idx in folds:
        vmonth = pd.Timestamp(features.loc[val_idx, "month"].iloc[0])
        tmax = pd.Timestamp(features.loc[train_idx, "month"].max())
        log.info(
            "  cv fold: val %s | train %d rows through %s (gap %d months)",
            vmonth.strftime("%Y-%m"),
            len(train_idx),
            tmax.strftime("%Y-%m"),
            _months_between(vmonth, tmax),
        )
