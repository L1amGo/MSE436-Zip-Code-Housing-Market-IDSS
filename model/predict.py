"""Predictions with confidence intervals (task M3).

`score(features, config)` returns, per input row, the point prediction (from the
M2 mean model) and the quantile columns p05/p10/p50/p90/p95 from the multi-
quantile model, plus band widths ci80_width = p90 - p10 and ci90_width =
p95 - p05. Quantile crossing (a higher quantile predicting below a lower one —
possible because the quantiles share trees but are not jointly constrained) is
repaired by sorting each row's quantile predictions ascending, so
p05 <= p10 <= p50 <= p90 <= p95 always holds.

Band-label note (M3 decision): the wider p05-p95 band is labelled **90%** (its
true nominal coverage), not 95%. See config `model.ci_levels` and the model card.

Library module — no CLI stage of its own; imported by evaluate (calibration),
the decision layer (M6), and the future dashboard.
"""

import joblib
import numpy as np
import pandas as pd

from model.io import model_config, models_dir

QUANTILE_COLS = ["p05", "p10", "p50", "p90", "p95"]
# Column index of each band's bounds within the sorted quantile matrix.
_CI80 = ("p10", "p90")
_CI90 = ("p05", "p95")


def _quantile_label(q: float) -> str:
    """0.05 -> 'p05', 0.5 -> 'p50', 0.9 -> 'p90' (two-digit percent)."""
    return f"p{round(q * 100):02d}"


def enforce_monotone(qpred: np.ndarray) -> np.ndarray:
    """Sort each row ascending so quantile predictions never cross."""
    return np.sort(np.asarray(qpred, dtype=float), axis=1)


def assemble_predictions(
    point: np.ndarray, qpred: np.ndarray, quantile_labels: list[str], index=None
) -> pd.DataFrame:
    """Build the prediction frame from a point vector and a (rows, quantiles) matrix.

    Pure and model-free so it is unit-testable with synthetic arrays. Enforces
    monotonicity, then derives the two band widths.
    """
    qpred = enforce_monotone(qpred)
    if qpred.shape[1] != len(quantile_labels):
        raise ValueError(
            f"quantile matrix has {qpred.shape[1]} columns but {len(quantile_labels)} labels"
        )
    out = pd.DataFrame(qpred, columns=quantile_labels, index=index)
    out.insert(0, "point", np.asarray(point, dtype=float))
    for band, (lo, hi) in {"ci80_width": _CI80, "ci90_width": _CI90}.items():
        if lo in out.columns and hi in out.columns:
            out[band] = out[hi] - out[lo]
    return out


def _load(name: str, config: dict):
    path = models_dir(config) / name
    if not path.exists():
        raise RuntimeError(
            f"{path.name} missing. Run `python -m model train` first to fit and save it."
        )
    return joblib.load(path)


def score(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Point + quantile predictions for every row of `features`.

    Returns a frame indexed like `features` with columns: zip, month (carried
    through for joining), point, p05, p10, p50, p90, p95, ci80_width, ci90_width.
    """
    point_art = _load("xgb_point.joblib", config)
    quant_art = _load("xgb_quantiles.joblib", config)
    feat_cols = point_art["feature_cols"]

    quantiles = quant_art["quantiles"]
    labels = [_quantile_label(q) for q in quantiles]

    X = features[feat_cols]
    point = point_art["model"].predict(X)
    qpred = np.asarray(quant_art["model"].predict(X), dtype=float)
    if qpred.ndim == 1:  # single-quantile fallback
        qpred = qpred.reshape(-1, 1)

    preds = assemble_predictions(point, qpred, labels, index=features.index)
    for key in ("zip", "month"):
        if key in features.columns:
            preds.insert(0, key, features[key].values)
    return preds
