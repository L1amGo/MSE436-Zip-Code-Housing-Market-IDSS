"""Scenario engine (task M6): re-score all zips under hypothetical macro shifts.

`apply_scenario(features, overrides)` shifts the configured macro columns and
consistently updates their derived 3-month deltas, then callers score the result
with `model.predict.score`. `score_scenario` does both in one call.

Consistency rule: a scenario represents "what if the CURRENT macro level were
different." Since a 3-month delta is (current - value 3 months ago) and history
is fixed, shifting a level by Δ shifts its delta by the same Δ. Overriding a
delta column directly just adds to it.

No Streamlit or UI imports — plain DataFrames in, predictions out — so the
dashboard (next spec) can call these directly.
"""

import pandas as pd

from model.io import model_config
from model.predict import score
from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("scenario")

# Level -> its featurize-derived 3-month delta (mirrors pipeline/featurize.py).
MACRO_DELTA_LINKS = {
    "MORTGAGE30US": "mortgage_delta_3m",
    "UNRATE": "unrate_delta_3m",
}


def apply_scenario(features: pd.DataFrame, overrides: dict[str, float], config: dict) -> pd.DataFrame:
    """Return a copy of `features` with macro overrides applied.

    `overrides` maps a macro column to an additive shift in that column's units
    (e.g. {"MORTGAGE30US": 0.5} = +50 bps). Only columns in
    `model.scenario_features` may be overridden. Shifting a level also shifts its
    linked 3-month delta so the two stay mutually consistent.
    """
    allowed = set(model_config(config)["scenario_features"])
    df = features.copy()
    for col, shift in overrides.items():
        if col not in allowed:
            raise RuntimeError(
                f"'{col}' is not an overridable scenario feature. Allowed: {sorted(allowed)}."
            )
        if col not in df.columns:
            raise RuntimeError(f"scenario column '{col}' not present in features.")
        df[col] = df[col] + shift
        link = MACRO_DELTA_LINKS.get(col)
        if link and link in df.columns and link != col:
            df[link] = df[link] + shift
    return df


def score_scenario(features: pd.DataFrame, overrides: dict[str, float], config: dict) -> pd.DataFrame:
    """apply_scenario then score — batch predictions for every input zip."""
    return score(apply_scenario(features, overrides, config), config)


def live_features(config: dict) -> pd.DataFrame:
    """The single most-recent month across all zips — the 'score everything now' set."""
    path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} missing. Run `python -m pipeline all` first.")
    feats = pd.read_parquet(path)
    latest = feats["month"].max()
    return feats[feats["month"] == latest].copy()


def run_benchmark(config: dict) -> float:
    """Time a full scenario -> allocation cycle over the whole zip universe.

    The proposal's interactivity claim depends on this being snappy; the spec
    target is <= 2s. Models are warm-loaded first so the measured span is the
    re-score + rank + filter + allocate the user actually waits on.
    """
    import time

    from model import decide

    mcfg = model_config(config)
    feats = live_features(config)
    log.info("scenario-bench: %d zips at %s", len(feats), feats["month"].max().strftime("%Y-%m"))

    # Warm the model cache (excluded from the timed span).
    score(feats.head(1), config)

    overrides = {"MORTGAGE30US": 0.5}  # +50 bps
    t = time.time()
    preds = score_scenario(feats, overrides, config)
    ranked = decide.rank(preds, mcfg["risk_lambda_default"], config)
    kept = decide.filter(ranked, mcfg["default_min_roi"], mcfg["default_max_downside"], ci_level=80)
    alloc = decide.allocate(kept, budget=1_000_000)
    elapsed = time.time() - t

    log.info(
        "scenario -> allocation for %d zips in %.3fs (%d qualified, $%.0f allocated); target <= 2s",
        len(feats), elapsed, len(kept), alloc["allocation"].sum(),
    )
    if elapsed > 2.0:
        log.warning("scenario-bench exceeded the 2s interactivity target (%.3fs)", elapsed)
    return elapsed


def run(config: dict) -> None:
    run_benchmark(config)
