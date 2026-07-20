"""Decision layer: risk-adjusted ranking, filtering, budget allocation (task M6).

The three functions the dashboard calls, all pure (plain DataFrames/params in
and out, no Streamlit):

  rank(predictions, risk_lambda):
      risk-adjusted score = p50 - risk_lambda * ci80_width, ranked descending.
      The penalty band is ALWAYS p10-p90 (ci80_width) regardless of the display
      CI level, so rankings stay comparable across toggle states.

  filter(ranked, min_roi, max_downside, ci_level, exclude_zips):
      keep zips with p50 >= min_roi and lower-bound >= -max_downside, where the
      lower bound is p10 at ci_level 80 and p05 at the stricter level 90 (a wider
      band -> lower lower-bound -> stricter eligibility). Watch-listed zips in
      exclude_zips are dropped from the candidate set.

  allocate(filtered, budget):
      budget split proportional to the (non-negative) risk-adjusted score; shares
      sum to the budget. If every qualifier scores <= 0, falls back to an equal
      split so the budget is still fully allocated.
"""

import pandas as pd

# Lower-bound quantile column for each selectable CI level (M3 band-label decision).
_LOWER_BOUND_COL = {80: "p10", 90: "p05"}


def rank(predictions: pd.DataFrame, risk_lambda: float, config: dict | None = None) -> pd.DataFrame:
    """Add a risk-adjusted `score` and return rows sorted best-first.

    score = p50 - risk_lambda * ci80_width. Ties broken by p50 then zip for
    determinism. `config` is accepted for signature symmetry but unused.
    """
    for col in ("p50", "ci80_width"):
        if col not in predictions.columns:
            raise RuntimeError(f"predictions missing required column '{col}' (run predict.score).")
    out = predictions.copy()
    out["score"] = out["p50"] - float(risk_lambda) * out["ci80_width"]
    sort_cols = ["score", "p50"] + (["zip"] if "zip" in out.columns else [])
    return out.sort_values(sort_cols, ascending=[False, False, *([True] if "zip" in out.columns else [])]).reset_index(drop=True)


def filter(
    ranked: pd.DataFrame,
    min_roi: float,
    max_downside: float,
    ci_level: int,
    exclude_zips=None,
) -> pd.DataFrame:
    """Eligibility filter. Stricter ci_level widens the band and shrinks the set."""
    if ci_level not in _LOWER_BOUND_COL:
        raise RuntimeError(f"ci_level must be one of {sorted(_LOWER_BOUND_COL)}, got {ci_level}.")
    lb_col = _LOWER_BOUND_COL[ci_level]
    if lb_col not in ranked.columns:
        raise RuntimeError(f"ranked table missing lower-bound column '{lb_col}'.")
    keep = (ranked["p50"] >= min_roi) & (ranked[lb_col] >= -max_downside)
    out = ranked[keep]
    if exclude_zips:
        excluded = {str(z) for z in exclude_zips}
        out = out[~out["zip"].astype(str).isin(excluded)]
    return out.reset_index(drop=True)


def allocate(filtered: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Split `budget` across qualifiers proportional to non-negative score.

    Returns zip + score + weight (fraction) + allocation (dollars). Weights sum
    to 1 and allocations sum to `budget` (up to float error). Empty input ->
    empty allocation.
    """
    cols = [c for c in ("zip", "score") if c in filtered.columns]
    out = filtered[cols].copy() if cols else filtered.copy()
    if out.empty:
        out["weight"] = []
        out["allocation"] = []
        return out
    weight = out["score"].clip(lower=0.0)
    total = float(weight.sum())
    if total <= 0.0:  # nobody has positive expected upside net of risk -> equal split
        weight = pd.Series(1.0, index=out.index)
        total = float(weight.sum())
    out["weight"] = (weight / total).to_numpy()
    out["allocation"] = (out["weight"] * float(budget)).to_numpy()
    return out.reset_index(drop=True)
