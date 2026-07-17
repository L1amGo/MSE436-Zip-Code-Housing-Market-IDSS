"""Unit tests for the clean stage using synthetic fixtures (no network, no raw files)."""

import pandas as pd
import pytest

from pipeline.clean import extract_zip, tidy_fred, tidy_redfin, tidy_zillow

THRESHOLD = 10


def _redfin_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a logical-named Redfin frame from compact row dicts."""
    defaults = {
        "median_sale_price": 500_000.0,
        "homes_sold": 20.0,
        "inventory": 40.0,
        "new_listings": 12.0,
        "median_dom": 30.0,
        "avg_sale_to_list": 0.99,
        "sold_above_list": 0.3,
        "price_drops": 0.1,
    }
    out = []
    for r in rows:
        r = dict(r)
        begin = pd.Timestamp(r.pop("begin"))
        # Real Redfin zip data ships 90-day rolling windows: begin = 1st of
        # month m, end = last day of month m+2. The row lands in month m+2.
        end = pd.Timestamp(r.pop("end", begin + pd.offsets.MonthEnd(3)))
        out.append(
            {
                "period_begin": begin,
                "period_end": end,
                "region_zip": r.pop("region", "Zip Code: 90210"),
                "property_type": "All Residential",
                **defaults,
                **r,
            }
        )
    return pd.DataFrame(out)


class TestZipPadding:
    def test_short_zips_zero_padded_to_5(self):
        s = pd.Series(["Zip Code: 210", "Zip Code: 90210", "2109"])
        assert extract_zip(s).tolist() == ["00210", "90210", "02109"]

    def test_no_digits_becomes_nan(self):
        assert extract_zip(pd.Series(["not a zip"])).isna().all()

    def test_redfin_and_zillow_zip_formats_agree(self):
        redfin = tidy_redfin(_redfin_frame([{"begin": "2024-01-01", "region": "Zip Code: 210"}]), THRESHOLD)
        zillow = tidy_zillow(
            pd.DataFrame({"RegionName": ["210"], "2024-01-31": [400_000.0]}),
            "RegionName",
            r"^\d{4}-\d{2}-\d{2}$",
        )
        assert redfin["zip"].iloc[0] == zillow["zip"].iloc[0] == "00210"


class TestWindowFilter:
    def test_window_assigned_to_month_it_ends_in(self):
        out = tidy_redfin(_redfin_frame([{"begin": "2024-01-01"}]), THRESHOLD)
        assert out["month"].tolist() == [pd.Timestamp("2024-03-01")]

    def test_malformed_windows_dropped(self):
        df = _redfin_frame(
            [
                {"begin": "2024-01-01"},  # well-formed 90-day window: kept
                {"begin": "2024-01-01", "end": "2024-01-31"},  # single month: dropped
                {"begin": "2024-01-08", "end": "2024-01-14"},  # weekly: dropped
            ]
        )
        out = tidy_redfin(df, THRESHOLD)
        assert len(out) == 1
        assert out["month"].iloc[0] == pd.Timestamp("2024-03-01")


class TestDedup:
    def test_duplicate_zip_month_keeps_one_row(self):
        df = _redfin_frame(
            [
                {"begin": "2024-01-01", "median_sale_price": 100.0},
                {"begin": "2024-01-01", "median_sale_price": 200.0},
            ]
        )
        out = tidy_redfin(df, THRESHOLD)
        assert len(out) == 1

    def test_dedup_is_order_insensitive(self):
        rows = [
            {"begin": "2024-01-01", "median_sale_price": 100.0},
            {"begin": "2024-01-01", "median_sale_price": 200.0},
        ]
        a = tidy_redfin(_redfin_frame(rows), THRESHOLD)
        b = tidy_redfin(_redfin_frame(rows[::-1]), THRESHOLD)
        pd.testing.assert_frame_equal(a, b)


class TestLowVolume:
    def test_flagged_not_dropped(self):
        df = _redfin_frame(
            [
                {"begin": "2024-01-01", "homes_sold": 9.0},
                {"begin": "2024-02-01", "homes_sold": 10.0},
                {"begin": "2024-03-01", "homes_sold": None},
            ]
        )
        out = tidy_redfin(df, THRESHOLD)
        assert len(out) == 3, "low-volume rows must be kept"
        assert out["low_volume"].tolist() == [True, False, True]


class TestNullPrice:
    def test_null_median_sale_price_dropped(self):
        df = _redfin_frame(
            [{"begin": "2024-01-01", "median_sale_price": None}, {"begin": "2024-02-01"}]
        )
        assert len(tidy_redfin(df, THRESHOLD)) == 1


class TestZillow:
    def test_melt_and_month_begin_convention(self):
        wide = pd.DataFrame(
            {
                "RegionName": ["90210"],
                "State": ["CA"],
                "2024-01-31": [1_000_000.0],
                "2024-02-29": [None],
            }
        )
        out = tidy_zillow(wide, "RegionName", r"^\d{4}-\d{2}-\d{2}$")
        assert out["month"].tolist() == [pd.Timestamp("2024-01-01")], "month-end -> month-begin; null zhvi dropped"
        assert out["zhvi"].tolist() == [1_000_000.0]


class TestFred:
    def test_weekly_to_monthly_mean_and_missing_dot(self):
        obs = {
            "WEEKLY": [
                {"date": "2024-01-04", "value": "6.0"},
                {"date": "2024-01-11", "value": "7.0"},
                {"date": "2024-01-18", "value": "."},
                {"date": "2024-02-01", "value": "5.0"},
            ],
            "MONTHLY": [{"date": "2024-01-01", "value": "3.7"}],
        }
        out = tidy_fred(obs, {"WEEKLY": "W", "MONTHLY": "M"})
        jan = out[out["month"] == "2024-01-01"].iloc[0]
        assert jan["WEEKLY"] == pytest.approx(6.5), "'.' excluded from the mean"
        assert jan["MONTHLY"] == pytest.approx(3.7)
        assert out[out["month"] == "2024-02-01"]["MONTHLY"].isna().all(), "no forward-fill"
