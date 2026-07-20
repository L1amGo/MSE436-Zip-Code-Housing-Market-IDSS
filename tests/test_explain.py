"""M5 tests: SHAP additivity and top-k contribution selection.

The additivity test (contributions sum to prediction - expected value) uses a
tiny synthetic XGBoost so it needs no artifacts. A sk-if-missing test exercises
the real per-zip path and its < 1s latency when the trained model + features
exist.
"""

import time

import numpy as np
import pandas as pd
import pytest

from model.explain import top_contributions


def test_top_contributions_ranks_by_absolute_value():
    row = np.array([0.01, -0.5, 0.2, -0.05])
    cols = ["a", "b", "c", "d"]
    top = top_contributions(row, cols, k=2)
    assert list(top.index) == ["b", "c"], "largest |contribution| first"
    assert top["b"] == -0.5, "sign is preserved"


def test_top_contributions_k_larger_than_features_returns_all():
    top = top_contributions(np.array([0.1, -0.2]), ["a", "b"], k=10)
    assert len(top) == 2


class TestAdditivity:
    def test_shap_values_sum_to_prediction_minus_expected(self):
        shap = pytest.importorskip("shap")
        from xgboost import XGBRegressor

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
        y = X["a"] * 1.5 - X["b"] * 0.8 + rng.normal(scale=0.2, size=300)
        model = XGBRegressor(
            n_estimators=60, max_depth=3, learning_rate=0.1,
            tree_method="hist", random_state=0, objective="reg:squarederror",
        ).fit(X, y)

        explainer = shap.TreeExplainer(model)
        sv = np.asarray(explainer.shap_values(X), dtype=float)
        expected = float(np.ravel(explainer.expected_value)[0])
        reconstructed = expected + sv.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model.predict(X), atol=1e-4)


class TestRealArtifact:
    def _ready(self, config):
        from model.io import models_dir
        from pipeline.io_utils import REPO_ROOT

        return (models_dir(config) / "xgb_point.joblib").exists() and (
            REPO_ROOT / config["paths"]["processed"] / "features.parquet"
        ).exists()

    def test_explain_zip_is_additive_and_fast(self):
        from pipeline.io_utils import load_config

        config = load_config()
        if not self._ready(config):
            pytest.skip("trained model or features.parquet missing (fresh clone)")
        from model.explain import explain_zip, get_explainer

        feats_path_row = pd.read_parquet(
            __import__("pipeline.io_utils", fromlist=["REPO_ROOT"]).REPO_ROOT
            / config["paths"]["processed"]
            / "features.parquet",
            columns=["zip", "month", "split"],
        )
        sample = feats_path_row[feats_path_row["split"] == "test"].iloc[0]

        get_explainer(config)  # warm the cache (excluded from the latency check)
        explain_zip(sample["zip"], sample["month"], config)  # warm feature cache
        t = time.time()
        out = explain_zip(sample["zip"], sample["month"], config)
        elapsed = time.time() - t
        assert elapsed < 1.0, f"explain_zip took {elapsed:.2f}s (must be < 1s)"

        # Full additivity for this zip: expected + all contributions == prediction.
        full = explain_zip(sample["zip"], sample["month"], config, k=10_000)
        assert abs(full["expected_value"] + full["contributions"].sum() - full["prediction"]) < 1e-3
        assert len(out["contributions"]) == config["model"]["top_k_shap_features"]
