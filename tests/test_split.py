"""Split-stage tests: temporal holdout boundaries and gapped rolling CV."""

import pandas as pd
import pytest

from pipeline.split import _months_between, assign_split, rolling_cv

CONFIG = {"params": {"cv_gap_months": 3, "train_window_months": 36, "holdout_months": 6}}


def synthetic_features(n_months: int = 40, live_tail: int = 3) -> pd.DataFrame:
    """Two-zip panel; the last `live_tail` months have NaN targets."""
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    live_cutoff = months[n_months - live_tail]
    rows = [
        {
            "zip": z,
            "month": m,
            "median_sale_price": 100_000.0,
            "target": float("nan") if m >= live_cutoff else 0.01,
        }
        for z in ("00001", "77002")
        for m in months
    ]
    return pd.DataFrame(rows)


class TestHoldout:
    def test_test_split_is_exactly_the_last_6_labeled_months(self):
        f = synthetic_features()
        f["split"] = assign_split(f, CONFIG["params"]["holdout_months"])
        labeled_months = sorted(f.loc[f["target"].notna(), "month"].unique())
        expected_test = set(labeled_months[-6:])
        assert set(f.loc[f["split"] == "test", "month"]) == expected_test
        assert all(m < min(expected_test) for m in f.loc[f["split"] == "train", "month"])

    def test_unlabeled_rows_are_live_not_train_or_test(self):
        f = synthetic_features()
        f["split"] = assign_split(f, CONFIG["params"]["holdout_months"])
        assert (f.loc[f["target"].isna(), "split"] == "live").all()
        assert f.loc[f["split"].isin(["train", "test"]), "target"].notna().all()

    def test_train_and_test_are_disjoint(self):
        f = synthetic_features()
        f["split"] = assign_split(f, CONFIG["params"]["holdout_months"])
        train_months = set(f.loc[f["split"] == "train", "month"])
        test_months = set(f.loc[f["split"] == "test", "month"])
        assert not train_months & test_months

    def test_too_few_labeled_months_raises(self):
        with pytest.raises(RuntimeError, match="labeled months"):
            assign_split(synthetic_features(n_months=8, live_tail=3), 6)


class TestRollingCV:
    def _fixture(self, n_months=40, window=36):
        config = {"params": {**CONFIG["params"], "train_window_months": window}}
        f = synthetic_features(n_months=n_months)
        f["split"] = assign_split(f, config["params"]["holdout_months"])
        return f, config

    def test_every_fold_has_gap_of_at_least_3_months(self):
        f, config = self._fixture()
        for train_idx, val_idx in rolling_cv(f, config, n_folds=4):
            val_month = f.loc[val_idx, "month"].iloc[0]
            train_end = f.loc[train_idx, "month"].max()
            assert _months_between(val_month, train_end) >= 3

    def test_no_train_window_exceeds_36_months(self):
        f, config = self._fixture(n_months=60, window=36)
        for train_idx, _ in rolling_cv(f, config, n_folds=3):
            months = f.loc[train_idx, "month"]
            assert _months_between(months.max(), months.min()) + 1 <= 36

    def test_small_window_cap_is_respected(self):
        f, config = self._fixture(window=6)
        for train_idx, _ in rolling_cv(f, config, n_folds=2):
            months = f.loc[train_idx, "month"]
            assert _months_between(months.max(), months.min()) + 1 <= 6

    def test_folds_use_only_train_rows_and_are_disjoint(self):
        f, config = self._fixture()
        for train_idx, val_idx in rolling_cv(f, config, n_folds=4):
            assert (f.loc[train_idx, "split"] == "train").all()
            assert (f.loc[val_idx, "split"] == "train").all()
            assert not set(train_idx) & set(val_idx)

    def test_validation_months_are_the_most_recent_train_months(self):
        f, config = self._fixture()
        folds = rolling_cv(f, config, n_folds=3)
        train_months = sorted(f.loc[f["split"] == "train", "month"].unique())
        got = [f.loc[v, "month"].iloc[0] for _, v in folds]
        assert got == list(train_months[-3:])
