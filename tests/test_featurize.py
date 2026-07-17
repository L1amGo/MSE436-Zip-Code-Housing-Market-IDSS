"""Featurize tests: exact label correctness and the lookahead (leakage) audit.

All fixtures are synthetic — no network, no pipeline outputs required. The
feature-dictionary test is the exception: it audits the real features.parquet
when present and is skipped on a fresh clone.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from pipeline.featurize import build_features
from pipeline.io_utils import REPO_ROOT, load_config

CONFIG = {
    "sources": {"fred": {"series": ["MORTGAGE30US", "UNRATE", "CPIAUCSL", "HOUST"]}},
    "params": {"label_horizon_months": 3, "target_outlier_threshold": 0.50},
}

LABEL_COLS = {"target", "target_outlier"}


def synthetic_joined(n_months: int = 30, zips: tuple[str, ...] = ("00001", "77002")) -> pd.DataFrame:
    """Deterministic panel: each zip gets a distinct, known price path."""
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    rows = []
    for zi, z in enumerate(zips):
        for i, m in enumerate(months):
            rows.append(
                {
                    "zip": z,
                    "month": m,
                    "median_sale_price": 100_000.0 * (zi + 1) + 1_000.0 * i,
                    "homes_sold": 20.0 + i + zi,
                    "inventory": 50.0 + 2.0 * i,
                    "new_listings": 10.0 + i,
                    "median_dom": 30.0 - 0.1 * i,
                    "avg_sale_to_list": 0.98 + 0.001 * i,
                    "sold_above_list": 0.25,
                    "price_drops": 0.10,
                    "low_volume": False,
                    "zhvi": 200_000.0 * (zi + 1) + 500.0 * i,
                    "MORTGAGE30US": 6.0 + 0.01 * i,
                    "UNRATE": 4.0 + 0.02 * i,
                    "CPIAUCSL": 300.0 + i,
                    "HOUST": 1400.0 + 10.0 * i,
                }
            )
    return pd.DataFrame(rows)


class TestLabel:
    def test_label_exactly_equals_realized_change_at_t_plus_3(self):
        joined = synthetic_joined()
        out = build_features(joined, CONFIG).set_index(["zip", "month"])
        price = joined.set_index(["zip", "month"])["median_sale_price"]
        for z in ("00001", "77002"):
            for t in pd.date_range("2020-01-01", "2021-06-01", freq="MS"):
                expected = price[(z, t + pd.DateOffset(months=3))] / price[(z, t)] - 1
                assert out.loc[(z, t), "target"] == pytest.approx(expected, abs=1e-12)

    def test_rows_near_data_end_keep_nan_target_and_are_retained(self):
        out = build_features(synthetic_joined(n_months=30), CONFIG)
        last3 = out[out["month"] > "2022-03-01"]  # months 28..30
        assert len(last3) == 6, "live rows must be retained"
        assert last3["target"].isna().all()
        assert out[out["month"] <= "2022-03-01"]["target"].notna().all()

    def test_target_outlier_flagged_not_dropped(self):
        joined = synthetic_joined(n_months=8, zips=("00001",))
        # Force a >50% jump realized at t+3 for t = month 3.
        joined.loc[joined["month"] == "2020-06-01", "median_sale_price"] = 1_000_000.0
        out = build_features(joined, CONFIG)
        flagged = out[out["target_outlier"]]
        assert len(flagged) >= 1
        assert pd.Timestamp("2020-03-01") in set(flagged["month"])
        assert len(out) == 8, "outlier rows must be kept"


class TestLookaheadAudit:
    def test_corrupting_future_months_changes_no_feature_at_t(self):
        cutoff = pd.Timestamp("2021-06-01")
        base = synthetic_joined()
        before = build_features(base, CONFIG)

        corrupted = base.copy()
        future = corrupted["month"] > cutoff
        numeric = corrupted.select_dtypes("number").columns
        corrupted.loc[future, numeric] = corrupted.loc[future, numeric] * 7.0 + 13.0
        after = build_features(corrupted, CONFIG)

        feature_cols = [c for c in before.columns if c not in LABEL_COLS]
        past_before = before[before["month"] <= cutoff][feature_cols].reset_index(drop=True)
        past_after = after[after["month"] <= cutoff][feature_cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(past_before, past_after)

    def test_momentum_uses_calendar_lag_not_row_offset(self):
        joined = synthetic_joined(n_months=6, zips=("00001",))
        joined = joined[joined["month"] != "2020-04-01"]  # gap in the panel
        out = build_features(joined, CONFIG).set_index("month")
        # 1m momentum at 2020-05 needs 2020-04, which is missing -> NaN, never
        # the previous available row (2020-03).
        assert pd.isna(out.loc["2020-05-01", "price_mom_1m"])
        assert out.loc["2020-03-01", "price_mom_1m"] == pytest.approx(
            joined.set_index("month").loc["2020-03-01", "median_sale_price"]
            / joined.set_index("month").loc["2020-02-01", "median_sale_price"]
            - 1
        )


class TestFeatureDictionary:
    FEATURES_PARQUET = REPO_ROOT / "data" / "processed" / "features.parquet"
    DICTIONARY = REPO_ROOT / "feature_dictionary.md"

    @pytest.mark.skipif(
        not FEATURES_PARQUET.exists(), reason="features.parquet not built yet (fresh clone)"
    )
    def test_dictionary_covers_exactly_the_parquet_columns(self):
        parquet_cols = set(pd.read_parquet(self.FEATURES_PARQUET).columns)
        documented = set(
            re.findall(r"^\| `([^`]+)`", self.DICTIONARY.read_text(encoding="utf-8"), re.MULTILINE)
        )
        assert documented == parquet_cols
