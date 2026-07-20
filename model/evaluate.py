"""Stage: evaluate — baselines, CV comparison, calibration, holdout (tasks M1/M2/M3/M4).

Baselines (M1): naive_zero, naive_momentum, linear (Ridge), scored with the
gapped rolling CV on train only. Later flags: --baselines-only, --intervals,
--holdout. The 6-month holdout is touched exactly once, in M4.
"""


def run(config: dict) -> None:
    raise NotImplementedError("evaluate — baselines in M1; comparison/holdout in M2/M4")
