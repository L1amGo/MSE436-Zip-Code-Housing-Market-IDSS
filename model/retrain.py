"""Stage: retrain — monthly refresh cycle with monitoring and versioning (task M7).

Re-runs pipeline download -> split, retrains point + quantile models on the
rolling 36-month window, recomputes monitoring metrics and drift stats, appends
to reports/retrain_log.md, versions the artifact, and raises a DEGRADATION
ALERT (nonzero exit) if fresh RMSE exceeds degradation_rmse_multiplier x
baseline.
"""


def run(config: dict) -> None:
    raise NotImplementedError("retrain — implemented in spec_model.md task M7")
