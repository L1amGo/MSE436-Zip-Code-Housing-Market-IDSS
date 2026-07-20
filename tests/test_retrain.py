"""M7 tests: degradation gate, promotion logic, rolling window, drift stats.

The full retrain cycle is a multi-minute pipeline+training run; the risk logic
that the acceptance criteria care about (alert, nonzero exit, no promotion) is
factored into pure functions and tested here without any fitting.
"""

import numpy as np
import pandas as pd

from model.retrain import (
    DEGRADED_EXIT,
    check_degradation,
    decide_promotion,
    drift_stats,
    resolve_outcome,
    rolling_window,
)


class TestDegradationGate:
    def test_within_threshold_is_not_degraded(self):
        assert check_degradation(0.11, baseline_rmse=0.10, multiplier=2.0) is False

    def test_beyond_threshold_is_degraded(self):
        assert check_degradation(0.25, baseline_rmse=0.10, multiplier=2.0) is True

    def test_nan_rmse_is_not_flagged(self):
        assert check_degradation(float("nan"), 0.10, 2.0) is False


class TestPromotion:
    def test_healthy_model_promotes(self):
        assert decide_promotion(degraded=False, accept_degraded=False) is True

    def test_degraded_model_not_promoted_by_default(self):
        assert decide_promotion(degraded=True, accept_degraded=False) is False

    def test_degraded_model_promoted_when_forced(self):
        assert decide_promotion(degraded=True, accept_degraded=True) is True


class TestResolveOutcome:
    def test_healthy_run_exits_zero_no_alert(self):
        degraded, promote, code, alert = resolve_outcome(0.11, 0.10, 2.0, accept_degraded=False)
        assert (degraded, promote, code, alert) == (False, True, 0, None)

    def test_degraded_run_alerts_and_exits_nonzero_without_promotion(self):
        degraded, promote, code, alert = resolve_outcome(0.30, 0.10, 2.0, accept_degraded=False)
        assert degraded is True and promote is False
        assert code == DEGRADED_EXIT and code != 0
        assert "DEGRADATION ALERT" in alert and "NOT promoted" in alert

    def test_accept_degraded_promotes_but_still_alerts(self):
        degraded, promote, code, alert = resolve_outcome(0.30, 0.10, 2.0, accept_degraded=True)
        assert degraded is True and promote is True
        assert code == 0
        assert "DEGRADATION ALERT" in alert and "Promoted anyway" in alert


class TestWindowAndDrift:
    def _panel(self, n_months=48):
        months = pd.date_range("2022-01-01", periods=n_months, freq="MS")
        rows = []
        for m in months:
            for z in ("00001", "00002"):
                rows.append({"zip": z, "month": m, "f": 1.0, "target": 0.01})
        df = pd.DataFrame(rows)
        df.loc[df["month"] >= months[-3], "target"] = np.nan  # recent live rows
        return df, months

    def test_rolling_window_keeps_only_recent_labeled_months(self):
        df, months = self._panel()
        win = rolling_window(df, months=36)
        assert win["target"].notna().all(), "only labeled rows"
        assert win["month"].nunique() == 36
        assert win["month"].max() == months[-4], "most recent labeled month (last 3 are live)"

    def test_drift_stats_flags_the_shifted_feature(self):
        ref = pd.DataFrame({"f": np.zeros(100), "g": np.zeros(100)})
        ref["f"] = np.linspace(-1, 1, 100)  # std ~ 0.58
        ref["g"] = np.linspace(-1, 1, 100)
        window = pd.DataFrame({"f": np.full(20, 2.0), "g": np.zeros(20)})  # f shifted hard, g not
        z = drift_stats(window, ref, ["f", "g"], top=2)
        assert z.index[0] == "f", "the shifted feature ranks first"
        assert z["f"] > z["g"]
