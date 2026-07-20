"""M6 tests: risk-adjusted ranking, CI-level eligibility, exclusion, allocation."""

import pandas as pd
import pytest

from model.decide import allocate, filter, rank


def _preds(rows: list[dict]) -> pd.DataFrame:
    """Build a predictions frame; fills quantile columns from shorthand."""
    return pd.DataFrame(rows)


class TestRank:
    def test_score_is_p50_minus_lambda_times_ci80_width(self):
        df = _preds([{"zip": "A", "p50": 0.10, "ci80_width": 0.20, "p10": 0.0, "p05": -0.1}])
        assert rank(df, 2.0)["score"].iloc[0] == pytest.approx(0.10 - 2.0 * 0.20)

    def test_higher_lambda_demotes_high_uncertainty_zip(self):
        # A: higher return but very uncertain; B: lower return, tight band.
        df = _preds(
            [
                {"zip": "A", "p50": 0.10, "ci80_width": 0.20, "p10": -0.02, "p05": -0.05},
                {"zip": "B", "p50": 0.08, "ci80_width": 0.02, "p10": 0.06, "p05": 0.05},
            ]
        )
        assert rank(df, 0.0)["zip"].iloc[0] == "A", "no risk penalty -> higher return wins"
        assert rank(df, 1.0)["zip"].iloc[0] == "B", "risk penalty promotes the tight-band zip"

    def test_missing_column_raises(self):
        with pytest.raises(RuntimeError, match="ci80_width"):
            rank(_preds([{"zip": "A", "p50": 0.1}]), 1.0)


class TestFilter:
    def _ranked(self):
        return rank(
            _preds(
                [
                    {"zip": "A", "p50": 0.10, "ci80_width": 0.10, "p10": -0.02, "p05": -0.08},
                    {"zip": "B", "p50": 0.05, "ci80_width": 0.04, "p10": 0.01, "p05": -0.03},
                    {"zip": "C", "p50": -0.01, "ci80_width": 0.04, "p10": -0.05, "p05": -0.09},
                ]
            ),
            1.0,
        )

    def test_min_roi_removes_below_threshold(self):
        kept = filter(self._ranked(), min_roi=0.0, max_downside=1.0, ci_level=80)
        assert set(kept["zip"]) == {"A", "B"}, "C has p50 < 0"

    def test_stricter_ci_level_never_grows_and_can_shrink(self):
        ranked = self._ranked()
        at80 = set(filter(ranked, min_roi=0.0, max_downside=0.05, ci_level=80)["zip"])
        at90 = set(filter(ranked, min_roi=0.0, max_downside=0.05, ci_level=90)["zip"])
        assert at90 <= at80, "stricter level (wider band) can only shrink the set"
        # A: p10=-0.02 (ok at 80) but p05=-0.08 (fails -0.05 at 90) -> drops.
        assert "A" in at80 and "A" not in at90

    def test_exclude_zips_removed(self):
        kept = filter(self._ranked(), min_roi=0.0, max_downside=1.0, ci_level=80, exclude_zips=["A"])
        assert "A" not in set(kept["zip"])


class TestAllocate:
    def test_shares_sum_to_budget(self):
        ranked = rank(
            _preds(
                [
                    {"zip": "A", "p50": 0.10, "ci80_width": 0.05, "p10": 0.05, "p05": 0.0},
                    {"zip": "B", "p50": 0.06, "ci80_width": 0.03, "p10": 0.03, "p05": 0.0},
                ]
            ),
            1.0,
        )
        alloc = allocate(ranked, budget=1_000_000)
        assert alloc["allocation"].sum() == pytest.approx(1_000_000)
        assert alloc["weight"].sum() == pytest.approx(1.0)
        assert (alloc["allocation"] >= 0).all()

    def test_excluded_zip_budget_redistributed(self):
        ranked = rank(
            _preds(
                [
                    {"zip": "A", "p50": 0.10, "ci80_width": 0.05, "p10": 0.05, "p05": 0.0},
                    {"zip": "B", "p50": 0.06, "ci80_width": 0.03, "p10": 0.03, "p05": 0.0},
                ]
            ),
            1.0,
        )
        kept = filter(ranked, min_roi=0.0, max_downside=1.0, ci_level=80, exclude_zips=["A"])
        alloc = allocate(kept, budget=1_000_000)
        assert set(alloc["zip"]) == {"B"}
        assert alloc["allocation"].sum() == pytest.approx(1_000_000), "budget fully reallocated to B"

    def test_all_nonpositive_scores_fall_back_to_equal_split(self):
        # Both scores negative after the risk penalty -> equal split, budget still spent.
        ranked = rank(
            _preds(
                [
                    {"zip": "A", "p50": 0.01, "ci80_width": 0.10, "p10": -0.05, "p05": -0.1},
                    {"zip": "B", "p50": 0.01, "ci80_width": 0.10, "p10": -0.05, "p05": -0.1},
                ]
            ),
            1.0,
        )
        alloc = allocate(ranked, budget=1000)
        assert alloc["allocation"].sum() == pytest.approx(1000)
        assert alloc["weight"].tolist() == pytest.approx([0.5, 0.5])

    def test_empty_input_gives_empty_allocation(self):
        empty = pd.DataFrame(columns=["zip", "score"])
        alloc = allocate(empty, budget=1000)
        assert alloc.empty
