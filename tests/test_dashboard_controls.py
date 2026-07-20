"""Dashboard sidebar controls (task D1).

These call the same handler functions the widgets call: `app.components.sidebar`
collects widget values into a `Controls`, and everything downstream of that is
`app.controls.evaluate()`, which is what is exercised here. A control that
passed these tests but did nothing on screen would require the sidebar to build
a `Controls` it never passes on — which `test_sidebar_passes_controls_to_evaluate`
guards against.

Synthetic and offline: the scoring call is stubbed with a deterministic fake that
reacts to the rate override the way the real model does (higher rates -> lower
predicted appreciation). One integration test uses the real committed models and
skips when the feature matrix hasn't been built.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from app import controls as C
from pipeline.io_utils import REPO_ROOT, load_config

FEATURES = REPO_ROOT / "data" / "processed" / "features.parquet"


@pytest.fixture
def config():
    return load_config()


# Three groups chosen so each control has something to actually change. Return
# and band width are deliberately NOT correlated: if they were, every lambda
# would produce the same ordering and the risk-tolerance test could not fail.
#
#   00001-00004  tight   modest return, narrow band  -> win at LOW tolerance
#   00005-00007  wide    high return, wide band      -> win at HIGH tolerance
#   00008-00010  marginal low return, wide band      -> qualify at 80%, fail at 90%
TIGHT = ["00001", "00002", "00003", "00004"]
WIDE = ["00005", "00006", "00007"]
MARGINAL = ["00008", "00009", "00010"]

# Widths are set so the tight/wide ordering flips *inside* the configured lambda
# range: with width_t = 0.01 and width_w = 0.10, the groups swap where
# lambda = (p50_w - p50_t) / (width_w - width_t) = 0.025 / 0.09 ~= 0.28, which
# sits between High (0.10) and Low (0.50).
_SPEC = [
    ("00001", 0.029, 0.005), ("00002", 0.028, 0.005),
    ("00003", 0.027, 0.005), ("00004", 0.026, 0.005),
    ("00005", 0.054, 0.050), ("00006", 0.052, 0.050), ("00007", 0.050, 0.050),
    ("00008", 0.012, 0.025), ("00009", 0.010, 0.025), ("00010", 0.008, 0.025),
]


@pytest.fixture
def universe():
    """Ten synthetic zips spanning the three groups above."""
    return pd.DataFrame(
        [
            {
                "zip": z,
                "month": pd.Timestamp("2026-05-01"),
                "MORTGAGE30US": 6.5,
                "mortgage_delta_3m": 0.1,
                "p50": p50,
                "half80": half80,
                "half90": half80 * 1.6,
            }
            for z, p50, half80 in _SPEC
        ]
    )


@pytest.fixture
def metro_lookup():
    return pd.DataFrame(
        {
            "zip": [f"{i:05d}" for i in range(1, 11)],
            "metro": ["Alpha, XX"] * 5 + ["Beta, YY"] * 5,
        }
    )


@pytest.fixture
def stub_score(monkeypatch, universe):
    """Replace the model call with a deterministic scorer that honours overrides.

    Mirrors the real contract: a MORTGAGE30US shift moves every p50, and the
    quantile columns stay monotone so `decide.filter` behaves as in production.
    """
    import model.scenario

    def fake_score_scenario(features, overrides, config):
        shift = float(overrides.get("MORTGAGE30US", 0.0))
        base = features.set_index("zip")
        p50 = base["p50"] - 0.02 * shift  # +100 bps -> -2pp of appreciation
        out = pd.DataFrame(
            {
                "zip": base.index,
                "month": base["month"].to_numpy(),
                "point": p50.to_numpy(),
                "p05": (p50 - base["half90"]).to_numpy(),
                "p10": (p50 - base["half80"]).to_numpy(),
                "p50": p50.to_numpy(),
                "p90": (p50 + base["half80"]).to_numpy(),
                "p95": (p50 + base["half90"]).to_numpy(),
            }
        )
        out["ci80_width"] = out["p90"] - out["p10"]
        out["ci90_width"] = out["p95"] - out["p05"]
        return out.reset_index(drop=True)

    monkeypatch.setattr(model.scenario, "score_scenario", fake_score_scenario)
    return fake_score_scenario


def run(controls, universe, metro_lookup, config):
    return C.evaluate(controls, universe, metro_lookup, config)


# --- defaults come from config.yaml, not from code -------------------------


def test_defaults_read_from_config(config):
    d = C.defaults(config)
    assert d.min_roi == config["model"]["default_min_roi"]
    assert d.max_downside == config["model"]["default_max_downside"]
    assert d.ci_level == config["dashboard"]["default_ci_level"]
    assert d.budget == config["dashboard"]["default_budget"]
    assert d.rate_bps == config["dashboard"]["default_rate_scenario_bps"]
    assert d.risk_tolerance == config["dashboard"]["default_risk_tolerance"]


def test_default_ci_level_is_a_supported_band(config):
    """Guards the M3 decision: the selectable bands are 80 and 90, never 95."""
    assert config["dashboard"]["default_ci_level"] in config["model"]["ci_levels"]
    assert config["model"]["ci_levels"] == [80, 90]


def test_risk_tolerance_lambdas_are_in_range(config):
    lo, hi = config["model"]["risk_lambda_range"]
    for label, lam in config["dashboard"]["risk_tolerance_lambda"].items():
        assert lo <= lam <= hi, f"{label} lambda {lam} outside {lo}-{hi}"


def test_lower_risk_tolerance_penalises_uncertainty_more(config):
    table = config["dashboard"]["risk_tolerance_lambda"]
    assert table["Low"] > table["Moderate"] > table["High"]


# --- rate scenario actually changes predictions ----------------------------


def test_rate_scenario_translates_bps_to_rate_shift():
    assert C.scenario_overrides(C.Controls((), 0, 0.05, 80, "Moderate", 50, 1e6)) == {
        "MORTGAGE30US": 0.5
    }
    assert C.scenario_overrides(C.Controls((), 0, 0.05, 80, "Moderate", -25, 1e6)) == {
        "MORTGAGE30US": -0.25
    }


def test_zero_bps_sends_no_override():
    """The baseline must go through the same code path, not a special case."""
    assert C.scenario_overrides(C.Controls((), 0, 0.05, 80, "Moderate", 0, 1e6)) == {}


def test_rate_scenario_changes_predicted_values(stub_score, universe, metro_lookup, config):
    base = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    up = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 100, 1e6), universe, metro_lookup, config)

    b = base.ranked.set_index("zip")["p50"]
    u = up.ranked.set_index("zip")["p50"]
    assert not b.equals(u), "rate scenario did not change any prediction"
    assert (u < b).all(), "a rate rise should lower predicted appreciation everywhere"


# --- CI level: stricter band can only shrink the qualifying set -------------


def test_raising_ci_level_can_only_shrink_qualifying_set(
    stub_score, universe, metro_lookup, config
):
    """90% uses p05, which is <= p10, so no zip can newly qualify."""
    at80 = run(C.Controls((), 0.0, 0.02, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    at90 = run(C.Controls((), 0.0, 0.02, 90, "Moderate", 0, 1e6), universe, metro_lookup, config)

    kept80 = set(at80.qualifying["zip"])
    kept90 = set(at90.qualifying["zip"])
    assert kept90 <= kept80, f"90% admitted zips 80% rejected: {kept90 - kept80}"
    assert len(kept90) < len(kept80), "fixture should show a strict shrink"


# --- risk tolerance re-orders the table ------------------------------------


def test_risk_tolerance_reorders_ranked_table(stub_score, universe, metro_lookup, config):
    low = run(C.Controls((), -1.0, 1.0, 80, "Low", 0, 1e6), universe, metro_lookup, config)
    high = run(C.Controls((), -1.0, 1.0, 80, "High", 0, 1e6), universe, metro_lookup, config)

    assert low.ranked["zip"].tolist() != high.ranked["zip"].tolist(), (
        "changing risk tolerance did not re-order the ranking"
    )


def test_tolerance_flips_which_group_wins(stub_score, universe, metro_lookup, config):
    """The whole point of lambda, stated as the manager would: a cautious buyer
    tops the table with narrow-band zips, an aggressive one with high-return
    wide-band zips — from identical predictions."""
    low = run(C.Controls((), -1.0, 1.0, 80, "Low", 0, 1e6), universe, metro_lookup, config)
    high = run(C.Controls((), -1.0, 1.0, 80, "High", 0, 1e6), universe, metro_lookup, config)

    assert set(low.ranked.head(len(TIGHT))["zip"]) == set(TIGHT)
    assert set(high.ranked.head(len(WIDE))["zip"]) == set(WIDE)


# --- eligibility filters ---------------------------------------------------


def test_raising_min_roi_shrinks_qualifying_set(stub_score, universe, metro_lookup, config):
    loose = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    tight = run(C.Controls((), 0.04, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    assert tight.qualified < loose.qualified
    assert (tight.qualifying["p50"] >= 0.04).all()


def test_tightening_max_downside_shrinks_qualifying_set(
    stub_score, universe, metro_lookup, config
):
    loose = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    tight = run(C.Controls((), -1.0, 0.01, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    assert tight.qualified < loose.qualified


# --- metro selection subsets the scoring universe --------------------------


def test_metro_selection_subsets_the_universe(stub_score, universe, metro_lookup, config):
    everything = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    alpha = run(
        C.Controls(("Alpha, XX",), -1.0, 1.0, 80, "Moderate", 0, 1e6),
        universe, metro_lookup, config,
    )
    assert everything.evaluated == 10
    assert alpha.evaluated == 5, "metro control must narrow what is scored, not just what is shown"
    assert set(alpha.ranked["metro"]) == {"Alpha, XX"}


def test_unmatched_zips_are_labelled_not_dropped(stub_score, universe, config):
    """Coverage honesty: a zip with no metro still gets scored and ranked."""
    partial = pd.DataFrame({"zip": ["00001"], "metro": ["Alpha, XX"]})
    out = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, partial, config)
    assert out.evaluated == 10
    assert (out.ranked["metro"] == C.UNKNOWN_METRO).sum() == 9


# --- exclusions ------------------------------------------------------------


def test_exclusions_never_appear_in_allocation(stub_score, universe, metro_lookup, config):
    base = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    dropped = base.qualifying["zip"].tolist()[:3]

    excluded = run(
        C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6, tuple(dropped)),
        universe, metro_lookup, config,
    )
    assert not set(dropped) & set(excluded.qualifying["zip"])
    assert "allocation" in excluded.qualifying


def test_excluded_budget_share_redistributes(stub_score, universe, metro_lookup, config):
    """Excluding a funded zip must push its dollars onto the others, not shrink
    the deployed total."""
    base = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)

    # Only zips scoring above zero receive money (allocate clips at 0), so pick
    # the excluded zip and the survivor from among those actually funded.
    funded = base.qualifying[base.qualifying["allocation"] > 0]["zip"].tolist()
    assert len(funded) >= 2, "fixture must fund at least two zips for this test to mean anything"
    dropped, survivor = funded[0], funded[-1]
    before = float(base.qualifying.set_index("zip").loc[survivor, "allocation"])

    excluded = run(
        C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6, (dropped,)),
        universe, metro_lookup, config,
    )
    after = float(excluded.qualifying.set_index("zip").loc[survivor, "allocation"])

    assert dropped not in set(excluded.qualifying["zip"])
    assert after > before, "freed budget was not redistributed"
    assert excluded.deployed == pytest.approx(base.budget)


def test_exclusions_are_zero_padded_to_match_features():
    """A CSV holding 2139 must exclude zip 02139."""
    assert C.normalize_zips(["2139", 2139, "02139", "2139.0"]) == ["02139"] * 4


# --- exclusions file parsing -----------------------------------------------


def test_parse_exclusions_with_zip_column():
    assert C.parse_exclusions(io.BytesIO(b"zip\n94110\n02139\n")) == ["94110", "02139"]


def test_parse_exclusions_is_case_insensitive_about_the_header():
    assert C.parse_exclusions(io.BytesIO(b"ZIP\n94110\n")) == ["94110"]


def test_parse_exclusions_headerless_single_column():
    assert C.parse_exclusions(io.BytesIO(b"94110\n02139\n")) == ["94110", "02139"]


def test_parse_exclusions_rejects_file_without_zip_column():
    with pytest.raises(C.ExclusionFormatError) as exc:
        C.parse_exclusions(io.BytesIO(b"city,state\nBoston,MA\n"))
    assert "zip" in str(exc.value).lower()
    assert "for example" in str(exc.value).lower(), "message must name the expected format"


def test_parse_exclusions_rejects_unreadable_file():
    with pytest.raises(C.ExclusionFormatError) as exc:
        C.parse_exclusions(io.BytesIO(b"\x00\x01\x02binary garbage"))
    assert "for example" in str(exc.value).lower()


# --- budget ----------------------------------------------------------------


def test_allocation_sums_to_budget(stub_score, universe, metro_lookup, config):
    out = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 2_500_000), universe, metro_lookup, config)
    assert out.qualifying["allocation"].sum() == pytest.approx(2_500_000)
    assert out.unallocated == pytest.approx(0.0)


def test_budget_change_rescales_allocation(stub_score, universe, metro_lookup, config):
    small = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    big = run(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 4e6), universe, metro_lookup, config)
    assert big.deployed == pytest.approx(4 * small.deployed)


def test_no_qualifiers_leaves_budget_unallocated(stub_score, universe, metro_lookup, config):
    """An impossible bar is reported as undeployed capital, not a crash."""
    out = run(C.Controls((), 0.99, 0.0, 80, "Moderate", 0, 1e6), universe, metro_lookup, config)
    assert out.qualified == 0
    assert out.deployed == 0.0
    assert out.unallocated == pytest.approx(1e6)


# --- the widgets really do hand their values to the model ------------------


def test_sidebar_passes_controls_to_evaluate():
    """`render` must return a Controls carrying every widget value.

    This is the link that makes the rest of this file meaningful: the sidebar's
    only job is to build this object, and main.py's only job is to pass it to
    evaluate().
    """
    import inspect

    from app.components import sidebar

    src = inspect.getsource(sidebar.render)
    for field in ("metros", "min_roi", "max_downside", "ci_level", "risk_tolerance",
                  "rate_bps", "budget", "exclude_zips"):
        assert f"{field}=" in src, f"sidebar.render does not pass {field} into Controls"


def test_app_imports_no_modelling_libraries():
    """The thin-UI invariant, enforced rather than asserted in a docstring.

    Checks import statements rather than raw text: the spec's shorthand grep
    (`grep -rE "xgboost|shap|sklearn" app/`) also trips on the substring in
    `frame.shape` and on the word SHAP in prose, neither of which is a
    violation. What actually matters is that no modelling library is imported
    here — computation belongs behind a `model/` call.
    """
    import ast

    banned = {"xgboost", "shap", "sklearn", "lightgbm"}
    for path in (REPO_ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            offending = names & banned
            assert not offending, f"{path.name} imports modelling logic: {offending}"


def test_app_source_keeps_the_spec_grep_clean():
    """Belt and braces: the spec's documented grep must also come back empty,
    so a reviewer running it verbatim sees what the spec promises."""
    import re

    pattern = re.compile(r"xgboost|shap|sklearn", re.IGNORECASE)
    hits = [
        f"{path.name}:{i}"
        for path in (REPO_ROOT / "app").rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not hits, f"spec grep `xgboost|shap|sklearn` matched in app/: {hits}"


# --- integration against the real committed models -------------------------


@pytest.mark.skipif(not FEATURES.exists(), reason="features.parquet not built")
def test_real_model_rate_scenario_moves_predictions(config):
    """End-to-end: the real XGBoost models, the real feature slice, one control."""
    from app import state

    feats = state.live_features.__wrapped__()
    metros = state.zip_metro.__wrapped__()

    base = C.evaluate(C.Controls((), -1.0, 1.0, 80, "Moderate", 0, 1e6), feats, metros, config)
    up = C.evaluate(C.Controls((), -1.0, 1.0, 80, "Moderate", 100, 1e6), feats, metros, config)

    b = base.ranked.set_index("zip")["p50"]
    u = up.ranked.set_index("zip")["p50"]
    assert not b.equals(u.reindex(b.index)), "+100 bps left every prediction unchanged"
