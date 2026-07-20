"""Ranked table and budget allocation view (task D2).

Covers the two claims the main panel makes: that the table reflects the current
control state rather than a cached earlier one, and that the dollars shown add up
to the budget the user typed. Table shaping is pure, so these run offline against
a synthetic decision.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import controls as C
from app.components import table
from pipeline.io_utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def qualifying():
    """A decision's qualifying set, already ranked and allocated."""
    n = 6
    p50 = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    half80 = [0.01, 0.012, 0.008, 0.02, 0.015, 0.01]
    weight = [0.30, 0.25, 0.18, 0.12, 0.10, 0.05]
    return pd.DataFrame(
        {
            "rank": range(1, n + 1),
            "zip": [f"{i:05d}" for i in range(1, n + 1)],
            "metro": ["Alpha, XX"] * 3 + ["Beta, YY"] * 3,
            "p50": p50,
            "p10": [p - h for p, h in zip(p50, half80)],
            "p90": [p + h for p, h in zip(p50, half80)],
            "p05": [p - 1.6 * h for p, h in zip(p50, half80)],
            "p95": [p + 1.6 * h for p, h in zip(p50, half80)],
            "ci80_width": [2 * h for h in half80],
            "score": [p - 2 * h for p, h in zip(p50, half80)],
            "weight": weight,
            "allocation": [w * 1_000_000 for w in weight],
        }
    )


# --- the table shows what the controls selected ----------------------------


def test_table_has_every_column_the_spec_names(qualifying):
    display = table.build_display_table(qualifying, 80)
    for col in ("rank", "zip", "metro", "pred_pct", "ci_lo_pct", "ci_hi_pct",
                "score_pp", "allocation", "share_pct"):
        assert col in display.columns


def test_ci_toggle_changes_which_band_is_displayed(qualifying):
    """80% must show p10/p90 and 90% must show p05/p95 — not the same numbers
    with a different label."""
    at80 = table.build_display_table(qualifying, 80)
    at90 = table.build_display_table(qualifying, 90)

    assert at80["ci_lo_pct"].tolist() == pytest.approx(qualifying["p10"] * 100)
    assert at90["ci_lo_pct"].tolist() == pytest.approx(qualifying["p05"] * 100)
    assert (at90["ci_lo_pct"] < at80["ci_lo_pct"]).all(), "90% band must be wider"
    assert (at90["ci_hi_pct"] > at80["ci_hi_pct"]).all()


def test_rejects_an_unsupported_ci_level(qualifying):
    """95% is not a band this model produces — fail loudly rather than mislabel."""
    with pytest.raises(RuntimeError, match="80, 90"):
        table.build_display_table(qualifying, 95)


def test_display_units_are_percentages_not_fractions(qualifying):
    display = table.build_display_table(qualifying, 80)
    assert display["pred_pct"].iloc[0] == pytest.approx(5.0)  # 0.05 -> 5%
    assert display["share_pct"].iloc[0] == pytest.approx(30.0)  # 0.30 -> 30%


def test_table_preserves_rank_order(qualifying):
    display = table.build_display_table(qualifying, 80)
    assert display["rank"].tolist() == sorted(display["rank"].tolist())


def test_empty_qualifying_set_yields_empty_table_not_an_error():
    display = table.build_display_table(pd.DataFrame(), 80)
    assert display.empty
    assert "allocation" in display.columns


# --- the dollars add up ----------------------------------------------------


def test_allocated_dollars_sum_to_the_budget(qualifying):
    display = table.build_display_table(qualifying, 80)
    assert display["allocation"].sum() == pytest.approx(1_000_000)


def test_shares_sum_to_one_hundred_percent(qualifying):
    display = table.build_display_table(qualifying, 80)
    assert display["share_pct"].sum() == pytest.approx(100.0)


def test_end_to_end_allocation_sums_to_budget_and_remainder_is_reported(config):
    """Through the real decision layer: deployed + unallocated == budget, always."""
    universe = pd.DataFrame(
        {
            "zip": [f"{i:05d}" for i in range(1, 6)],
            "month": pd.Timestamp("2026-05-01"),
            "p50": [0.05, 0.04, 0.03, 0.02, 0.01],
            "ci80_width": [0.02, 0.02, 0.02, 0.02, 0.02],
            "p10": [0.04, 0.03, 0.02, 0.01, 0.0],
            "p05": [0.03, 0.02, 0.01, 0.0, -0.01],
            "p90": [0.06, 0.05, 0.04, 0.03, 0.02],
            "p95": [0.07, 0.06, 0.05, 0.04, 0.03],
        }
    )
    from model import decide

    ranked = decide.rank(universe, 1.0, config)
    kept = decide.filter(ranked, min_roi=0.0, max_downside=0.05, ci_level=80)
    alloc = decide.allocate(kept, budget=750_000)
    assert alloc["allocation"].sum() == pytest.approx(750_000)


def test_unallocated_is_the_whole_budget_when_nothing_qualifies():
    d = C.Decision(pd.DataFrame(), pd.DataFrame(), 100, 0, 1_000_000.0, 0.0)
    assert d.unallocated == pytest.approx(1_000_000)


def test_deployed_plus_unallocated_equals_budget():
    d = C.Decision(pd.DataFrame(), pd.DataFrame(), 100, 4, 1_000_000.0, 600_000.0)
    assert d.deployed + d.unallocated == pytest.approx(d.budget)


# --- the CSV download matches what is on screen ----------------------------


def test_csv_headers_name_their_units(qualifying):
    csv = table.allocation_csv(table.build_display_table(qualifying, 80), 80).decode()
    header = csv.splitlines()[0]
    for expected in ("predicted_3mo_change_pct", "ci80_low_pct", "ci80_high_pct",
                     "allocated_usd", "allocated_share_pct"):
        assert expected in header


def test_csv_band_headers_follow_the_ci_toggle(qualifying):
    csv90 = table.allocation_csv(table.build_display_table(qualifying, 90), 90).decode()
    assert "ci90_low_pct" in csv90.splitlines()[0]


def test_csv_row_count_matches_the_table(qualifying):
    display = table.build_display_table(qualifying, 80)
    csv = table.allocation_csv(display, 80).decode().strip().splitlines()
    assert len(csv) - 1 == len(display)


# --- the figure is labelled ------------------------------------------------


def test_allocation_chart_axes_are_labelled_with_units(qualifying):
    """C5: every figure carries an axis label and units."""
    display = table.build_display_table(qualifying, 80)
    fig = table.allocation_chart(display, table.theme.LIGHT)

    assert "USD" in fig.layout.xaxis.title.text
    assert fig.layout.yaxis.title.text
    assert fig.layout.showlegend is False, "single-series chart needs no legend"


def test_allocation_chart_is_capped_and_ordered(qualifying):
    display = table.build_display_table(qualifying, 80)
    fig = table.allocation_chart(display, table.theme.LIGHT, top_n=3)
    bar = fig.data[0]
    assert len(bar.x) == 3
    assert list(bar.x) == sorted(bar.x), "horizontal bars read best ascending"


def test_score_formula_is_defined_for_the_ui():
    """The formula must be on screen, so it must exist as displayable text."""
    assert table.SCORE_FORMULA == "score = p50 − λ · (p90 − p10)"
    assert "SCORE_FORMULA" in open(
        table.__file__, encoding="utf-8"
    ).read().split("def render")[1], "render must show the score formula"


# --- concentration guard ---------------------------------------------------


def test_concentration_note_fires_on_a_single_bet(qualifying):
    """The lambda=1.0 pathology: one zip takes everything."""
    q = qualifying.copy()
    q["weight"] = [1.0] + [0.0] * (len(q) - 1)
    q["allocation"] = [1_000_000.0] + [0.0] * (len(q) - 1)
    note = table.concentration_note(table.build_display_table(q, 80))
    assert note is not None
    assert "100.0%" in note and "1 of" in note


def test_concentration_note_fires_at_thirty_percent(qualifying):
    """The fixture's top zip holds 30% of the budget — above the 25% guard, and
    genuinely concentrated for a portfolio, so the note is expected."""
    assert table.concentration_note(table.build_display_table(qualifying, 80)) is not None


def test_concentration_note_silent_on_a_spread_allocation(qualifying):
    q = qualifying.copy()
    even = 1.0 / len(q)
    q["weight"] = even
    q["allocation"] = even * 1_000_000
    assert table.concentration_note(table.build_display_table(q, 80)) is None


def test_concentration_note_handles_empty_table():
    assert table.concentration_note(table.build_display_table(pd.DataFrame(), 80)) is None


def test_rank_is_universe_position_not_row_number(qualifying):
    """Rank must survive filtering, so a gap means 'that zip was rejected'."""
    filtered = qualifying[qualifying["rank"].isin([1, 3, 5])]
    display = table.build_display_table(filtered, 80)
    assert display["rank"].tolist() == [1, 3, 5]
