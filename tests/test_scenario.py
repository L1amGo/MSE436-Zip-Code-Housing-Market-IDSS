"""M6 tests: scenario overrides adjust macro levels and derived deltas consistently."""

import pandas as pd
import pytest

from model.scenario import apply_scenario

CONFIG = {"model": {"scenario_features": ["MORTGAGE30US", "mortgage_delta_3m"]}}


def _features():
    return pd.DataFrame(
        {
            "zip": ["00001", "00002"],
            "month": pd.Timestamp("2026-05-01"),
            "MORTGAGE30US": [6.0, 6.0],
            "mortgage_delta_3m": [0.2, 0.2],
            "UNRATE": [4.0, 4.0],
            "median_sale_price": [100.0, 200.0],
        }
    )


def test_level_override_shifts_linked_delta_consistently():
    out = apply_scenario(_features(), {"MORTGAGE30US": 0.5}, CONFIG)
    assert (out["MORTGAGE30US"] == 6.5).all(), "+50 bps applied to the level"
    assert out["mortgage_delta_3m"].tolist() == pytest.approx([0.7, 0.7]), "3-mo delta shifts by the same +0.5"
    assert (out["UNRATE"] == 4.0).all(), "unrelated macro untouched"


def test_negative_override():
    out = apply_scenario(_features(), {"MORTGAGE30US": -0.25}, CONFIG)
    assert (out["MORTGAGE30US"] == 5.75).all()
    assert out["mortgage_delta_3m"].tolist() == pytest.approx([-0.05, -0.05])


def test_overriding_delta_directly_adds_only_to_it():
    out = apply_scenario(_features(), {"mortgage_delta_3m": 0.1}, CONFIG)
    assert out["mortgage_delta_3m"].tolist() == pytest.approx([0.3, 0.3])
    assert (out["MORTGAGE30US"] == 6.0).all(), "level unchanged when only the delta is overridden"


def test_original_frame_not_mutated():
    feats = _features()
    apply_scenario(feats, {"MORTGAGE30US": 1.0}, CONFIG)
    assert (feats["MORTGAGE30US"] == 6.0).all(), "apply_scenario must not mutate its input"


def test_non_scenario_feature_rejected():
    with pytest.raises(RuntimeError, match="not an overridable scenario feature"):
        apply_scenario(_features(), {"median_sale_price": 10.0}, CONFIG)
