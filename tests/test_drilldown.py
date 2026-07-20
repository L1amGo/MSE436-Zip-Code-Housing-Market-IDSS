"""Per-zip drill-down (task D4).

Two claims carry real risk and are tested against the real artifacts:
explanations come from `model/` rather than being recomputed in the UI, and the
backtest shows holdout months only. The rest — labelling, the driver sentence,
chart anatomy — runs offline on synthetic input.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from app import controls as C
from app.components import drilldown as D
from app.components import theme
from pipeline.io_utils import REPO_ROOT, load_config

FEATURES = REPO_ROOT / "data" / "processed" / "features.parquet"
MODELS = REPO_ROOT / "models" / "xgb_point.joblib"
needs_artifacts = pytest.mark.skipif(
    not (FEATURES.exists() and MODELS.exists()),
    reason="feature matrix or trained model not built",
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def contributions():
    """Signed contributions, deliberately unsorted so ordering logic is exercised."""
    return pd.Series(
        {
            "price_mom_3m": 0.021,
            "median_dom": -0.014,
            "mortgage_delta_3m": -0.031,
            "inventory": 0.004,
            "sold_above_list": 0.002,
        }
    )


@pytest.fixture
def explanation(contributions):
    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    return {
        "zip": "06492",
        "month": "2026-05",
        "expected_value": 0.012,
        "prediction": 0.012 + float(ordered.sum()),
        "contributions": ordered,
    }


# --- the plain-language driver line ----------------------------------------


def test_driver_sentence_names_the_top_three(contributions):
    line = D.driver_sentence(contributions)
    assert "3-month mortgage rate change" in line
    assert "3-month price momentum" in line
    assert "median days on market" in line
    assert "inventory" not in line, "only the top three drivers belong in the sentence"


def test_driver_sentence_states_direction(contributions):
    line = D.driver_sentence(contributions)
    assert "push this forecast up" in line
    assert "pull it down" in line


def test_driver_sentence_with_only_positive_drivers():
    line = D.driver_sentence(pd.Series({"price_mom_3m": 0.02, "inventory": 0.01}))
    assert "push this forecast up" in line
    assert "pull it down" not in line


def test_driver_sentence_on_empty_contributions():
    assert "No feature contributions" in D.driver_sentence(pd.Series(dtype=float))


def test_feature_labels_are_plain_language():
    assert D.label_for("mortgage_delta_3m") == "3-month mortgage rate change"
    assert D.label_for("some_new_feature") == "some_new_feature", "unknown names pass through"


# --- the contribution waterfall --------------------------------------------


def test_contribution_chart_is_labelled_with_units(explanation):
    fig = D.contribution_chart(explanation, theme.LIGHT)
    assert "pp" in fig.layout.yaxis.title.text
    assert fig.layout.xaxis.title.text


def test_contribution_chart_reconciles_to_the_prediction(explanation):
    """Base + every bar must equal the point forecast, or the explanation is
    decorative rather than an accounting of the prediction."""
    fig = D.contribution_chart(explanation, theme.LIGHT)
    bar = fig.data[0]
    total = float(bar.base) + sum(float(v) for v in bar.y)
    assert total == pytest.approx(explanation["prediction"] * 100, abs=1e-6)


def test_contribution_chart_includes_a_residual_bar(explanation):
    fig = D.contribution_chart(explanation, theme.LIGHT)
    assert D.OTHER_LABEL in list(fig.data[0].x)


# --- the backtest chart ----------------------------------------------------


@pytest.fixture
def backtest():
    months = pd.date_range("2025-09-01", periods=6, freq="MS")
    return pd.DataFrame(
        {
            "month": months,
            "predicted": [0.01, 0.02, 0.00, -0.01, 0.03, 0.02],
            "realized": [0.02, 0.01, -0.01, 0.00, 0.02, 0.03],
            "lo": [-0.02, -0.01, -0.03, -0.04, 0.00, -0.01],
            "hi": [0.04, 0.05, 0.03, 0.02, 0.06, 0.05],
        }
    )


def test_backtest_chart_axes_are_labelled_with_units(backtest):
    fig = D.backtest_chart(backtest, theme.LIGHT)
    assert "%" in fig.layout.yaxis.title.text
    assert "month" in fig.layout.xaxis.title.text.lower()


def test_backtest_chart_has_a_legend_for_its_two_series(backtest):
    """Two series with different meanings — identity must not be colour-alone."""
    fig = D.backtest_chart(backtest, theme.LIGHT)
    assert fig.layout.showlegend is True
    names = {t.name for t in fig.data}
    assert {"Predicted", "Realized"} <= names


def test_backtest_chart_plots_every_holdout_month(backtest):
    fig = D.backtest_chart(backtest, theme.LIGHT)
    predicted = next(t for t in fig.data if t.name == "Predicted")
    assert len(predicted.x) == len(backtest)


# --- metro comparison ------------------------------------------------------


def test_metro_comparison_uses_the_metro_not_the_universe(contributions):
    features = pd.DataFrame(
        {
            "zip": ["00001", "00002", "00003", "00099"],
            "price_mom_3m": [0.10, 0.02, 0.06, 99.0],
            "median_dom": [10.0, 20.0, 30.0, 999.0],
            "mortgage_delta_3m": [0.1, 0.2, 0.3, 9.9],
        }
    )
    lookup = pd.DataFrame(
        {"zip": ["00001", "00002", "00003", "00099"],
         "metro": ["Alpha, XX"] * 3 + ["Beta, YY"]}
    )
    out = D.metro_comparison("00001", "Alpha, XX", features, lookup, contributions)
    row = out[out["feature"] == "3-month price momentum"].iloc[0]
    assert row["this_zip"] == pytest.approx(0.10)
    # median of Alpha's 0.10/0.02/0.06 is 0.06 — the Beta outlier must not leak in
    assert row["metro_median"] == pytest.approx(0.06)


def test_metro_comparison_on_unknown_zip(contributions):
    features = pd.DataFrame({"zip": ["00001"], "price_mom_3m": [0.1]})
    lookup = pd.DataFrame({"zip": ["00001"], "metro": ["Alpha, XX"]})
    assert D.metro_comparison("99999", "Alpha, XX", features, lookup, contributions).empty


# --- against the real models and feature matrix ----------------------------


@needs_artifacts
def test_backtest_uses_holdout_months_only(config):
    """The criterion that would silently flatter the model if broken."""
    frame = D.backtest_frame("06492", config)
    assert not frame.empty

    raw = pd.read_parquet(FEATURES, filters=[("zip", "==", "06492")])
    holdout_months = set(raw[raw["split"] == D.HOLDOUT_SPLIT]["month"])
    train_months = set(raw[raw["split"] == "train"]["month"])

    plotted = set(frame["month"])
    assert plotted <= holdout_months, "backtest plotted a non-holdout month"
    assert not (plotted & train_months), "backtest plotted a training month"


@needs_artifacts
def test_backtest_realized_matches_the_recorded_label(config):
    frame = D.backtest_frame("06492", config)
    raw = pd.read_parquet(FEATURES, filters=[("zip", "==", "06492")])
    expected = raw[raw["split"] == D.HOLDOUT_SPLIT].set_index("month")["target"]
    for _, r in frame.iterrows():
        assert r["realized"] == pytest.approx(float(expected.loc[r["month"]]))


@needs_artifacts
def test_backtest_of_an_unknown_zip_is_empty_not_an_error(config):
    assert D.backtest_frame("99999", config).empty


@needs_artifacts
def test_explanation_comes_from_the_model_layer(config):
    """Values must match `model.explain.explain_zip` exactly — the UI must not
    recompute or round them."""
    from model.explain import explain_zip

    features = pd.read_parquet(FEATURES, columns=["zip", "month", "split"])
    month = features[features["split"] == "live"]["month"].max()

    mine = D.explain("06492", month, config)
    theirs = explain_zip("06492", month, config)
    assert mine["prediction"] == theirs["prediction"]
    assert mine["contributions"].equals(theirs["contributions"])


@needs_artifacts
def test_drilldown_renders_within_the_interactivity_budget(config):
    """Spec bar: the panel must come back in <= 2s once caches are warm."""
    features = pd.read_parquet(FEATURES, columns=["zip", "month", "split"])
    month = features[features["split"] == "live"]["month"].max()

    D.explain("06492", month, config)  # warm the explainer and feature cache
    D.backtest_frame("06492", config)

    start = time.perf_counter()
    D.explain("21221", month, config)
    D.backtest_frame("21221", config)
    elapsed = time.perf_counter() - start
    assert elapsed <= 2.0, f"drill-down took {elapsed:.2f}s, over the 2s budget"


@needs_artifacts
def test_explanations_are_not_computed_in_the_app_layer():
    """The C4 invariant: app/ may display contributions, never derive them."""
    import ast

    banned = {"TreeExplainer", "shap_values", "Explainer"}
    for path in (REPO_ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in banned:
                raise AssertionError(f"{path.name} computes explanations itself")
            if isinstance(node, ast.Name) and node.id in banned:
                raise AssertionError(f"{path.name} computes explanations itself")
