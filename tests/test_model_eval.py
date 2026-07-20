"""M1 evaluation-harness tests (synthetic fixtures, no pipeline outputs).

Core acceptance test: the gapped-CV harness excludes flagged rows and never
touches the test split. Plus honesty checks on the metric computation.
"""

import math

import pandas as pd
import pytest

from model.evaluate import (
    BASELINES,
    cv_evaluate,
    prepared_folds,
    _predict_momentum,
    _predict_zero,
)

CONFIG = {
    "params": {"cv_gap_months": 3, "train_window_months": 36},
    "model": {"cv_folds": 3, "exclude_flags": ["low_volume", "target_outlier"], "random_seed": 42},
}


def _features(n_months=28, n_train=22, target_val=0.01, momentum_equals_target=False, flagged=True):
    """Two-zip panel with an explicit train/test split and optional flagged rows.

    n_train months are `train`, the remainder `test`. Flagged rows are injected
    into one train-window month (2021-05) and one validation month (2021-08).
    """
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    train_months = set(months[:n_train])
    rows = []
    for zi, z in enumerate(("00001", "77002")):
        for i, m in enumerate(months):
            tgt = target_val + 0.001 * zi + 0.0001 * i if momentum_equals_target else target_val
            rows.append(
                {
                    "zip": z,
                    "month": m,
                    "median_sale_price": 100_000.0 + i,
                    "price_mom_3m": tgt if momentum_equals_target else 0.02,
                    "month_of_year": m.month,
                    "target": tgt,
                    "low_volume": False,
                    "target_outlier": False,
                    "split": "train" if m in train_months else "test",
                }
            )
    if flagged:
        for m in (pd.Timestamp("2021-05-01"), pd.Timestamp("2021-08-01")):
            base = {
                "zip": "00001",
                "month": m,
                "median_sale_price": 1.0,
                "price_mom_3m": 9.9,
                "month_of_year": m.month,
                "target": 9.9,
                "split": "train",
            }
            rows.append({**base, "low_volume": True, "target_outlier": False})
            rows.append({**base, "low_volume": False, "target_outlier": True})
    return pd.DataFrame(rows)


class TestHarnessExclusions:
    def test_prepared_folds_exclude_flagged_and_only_use_train(self):
        folds = prepared_folds(_features(), CONFIG)
        assert len(folds) == 3
        for train_df, val_df in folds:
            assert (train_df["split"] == "train").all()
            assert (val_df["split"] == "train").all()
            for d in (train_df, val_df):
                assert not d["low_volume"].any()
                assert not d["target_outlier"].any()

    def test_test_split_rows_are_never_used(self):
        used = pd.concat(
            [pd.concat([t, v]) for t, v in prepared_folds(_features(), CONFIG)]
        )
        assert "test" not in set(used["split"])
        assert "live" not in set(used["split"])

    def test_pooled_n_counts_only_unflagged_validation_rows(self):
        f = _features()
        res = cv_evaluate(f, CONFIG, _predict_zero)
        expected = sum(len(v) for _, v in prepared_folds(f, CONFIG))
        assert res["pooled"]["n"] == expected


class TestMetricHonesty:
    def test_zero_baseline_metrics_are_honest(self):
        res = cv_evaluate(_features(target_val=0.01), CONFIG, _predict_zero)
        assert res["pooled"]["dir_acc"] == 0.0  # sign(0) never matches a positive target
        assert math.isnan(res["pooled"]["rank_corr"])  # constant prediction -> undefined

    def test_momentum_perfect_recovers_zero_error(self):
        res = cv_evaluate(_features(momentum_equals_target=True), CONFIG, _predict_momentum)
        assert res["pooled"]["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert res["pooled"]["mae"] == pytest.approx(0.0, abs=1e-12)
        assert res["pooled"]["dir_acc"] == pytest.approx(1.0)

    def test_all_three_baselines_run(self):
        f = _features()
        for fn in BASELINES.values():
            assert cv_evaluate(f, CONFIG, fn)["pooled"]["n"] > 0
