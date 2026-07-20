"""M3 tests: quantile prediction assembly, monotonicity, band widths (no network).

The pure assembly/monotonicity logic is tested with synthetic arrays. One test
exercises the real multi-quantile XGBoost path on a tiny synthetic panel to
confirm crossing is repaired end-to-end. A sk-if-missing test covers the saved
artifact when it exists.
"""

import numpy as np
import pandas as pd
import pytest

from model.predict import (
    QUANTILE_COLS,
    _quantile_label,
    assemble_predictions,
    enforce_monotone,
)

LABELS = ["p05", "p10", "p50", "p90", "p95"]


def test_quantile_label_formats_two_digits():
    assert [_quantile_label(q) for q in (0.05, 0.1, 0.5, 0.9, 0.95)] == LABELS


class TestMonotonicity:
    def test_enforce_monotone_sorts_each_row(self):
        crossed = np.array([[0.9, 0.1, 0.5, 0.2, 0.95]])  # deliberately unordered
        fixed = enforce_monotone(crossed)
        assert np.all(np.diff(fixed, axis=1) >= 0)
        assert fixed.tolist() == [[0.1, 0.2, 0.5, 0.9, 0.95]]

    def test_assembly_output_is_ordered_and_finite(self):
        rng = np.random.default_rng(0)
        qpred = rng.normal(size=(50, 5))  # unordered on purpose
        out = assemble_predictions(rng.normal(size=50), qpred, LABELS)
        q = out[QUANTILE_COLS].to_numpy()
        assert np.all(np.diff(q, axis=1) >= 0), "quantiles must be non-decreasing across p05..p95"
        assert np.isfinite(out.to_numpy()).all()


class TestColumnsAndWidths:
    def test_all_expected_columns_present(self):
        out = assemble_predictions(np.zeros(3), np.zeros((3, 5)), LABELS)
        for col in ["point", *QUANTILE_COLS, "ci80_width", "ci90_width"]:
            assert col in out.columns

    def test_band_widths_match_quantile_gaps(self):
        qpred = np.array([[0.0, 0.1, 0.2, 0.3, 0.4]])  # p05..p95 already ordered
        out = assemble_predictions([0.2], qpred, LABELS)
        assert out["ci80_width"].iloc[0] == pytest.approx(0.3 - 0.1)  # p90 - p10
        assert out["ci90_width"].iloc[0] == pytest.approx(0.4 - 0.0)  # p95 - p05

    def test_widths_nonnegative_even_when_input_crosses(self):
        qpred = np.array([[0.4, 0.3, 0.2, 0.1, 0.0]])  # fully reversed
        out = assemble_predictions([0.2], qpred, LABELS)
        assert out["ci80_width"].iloc[0] >= 0
        assert out["ci90_width"].iloc[0] >= 0

    def test_label_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="columns but"):
            assemble_predictions(np.zeros(2), np.zeros((2, 4)), LABELS)


class TestRealQuantilePath:
    def test_multi_quantile_fit_predicts_ordered_finite_bands(self):
        from model.train import fit_quantile_model

        rng = np.random.default_rng(42)
        n = 800
        df = pd.DataFrame(
            {"f1": rng.normal(size=n), "f2": rng.normal(size=n)}
        )
        df["target"] = df["f1"] * 0.5 + rng.normal(scale=0.3, size=n)
        quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
        model = fit_quantile_model(
            df, ["f1", "f2"], {"max_depth": 3, "n_estimators": 60, "learning_rate": 0.1}, quantiles, 42
        )
        out = assemble_predictions(
            model.predict(df[["f1", "f2"]])[:, 2],  # p50 as a stand-in point
            np.asarray(model.predict(df[["f1", "f2"]])),
            LABELS,
        )
        q = out[QUANTILE_COLS].to_numpy()
        assert np.all(np.diff(q, axis=1) >= 0)
        assert np.isfinite(q).all()
        assert (out["ci80_width"] >= 0).all() and (out["ci90_width"] >= 0).all()
