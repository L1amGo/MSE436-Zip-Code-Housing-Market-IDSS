"""Stage: evaluate — baselines, CV comparison, calibration, holdout (tasks M1/M2/M3/M4).

M1 implements three reference models, all scored with the SAME gapped rolling CV
on the train split only (invariant 1 — the 6-month holdout is never touched here):

  - naive_zero:     predict 0% change
  - naive_momentum: predict the zip's trailing 3-month % change (price_mom_3m)
  - linear:         Ridge regression on the full feature set

Flagged rows (config `model.exclude_flags` = low_volume, target_outlier) are
excluded from both fitting and metrics (invariant 5). Metrics per fold and
pooled: RMSE, MAE, directional accuracy (sign agreement), and rank correlation
(Spearman between predicted and realized change across zips within a month —
the metric closest to the manager's actual use, since decisions act on
rankings). Results are written to reports/baselines.md.

Later flags (--intervals, --holdout) and the model comparison land in M2/M3/M4.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.io import model_config, reports_dir
from pipeline.io_utils import REPO_ROOT, get_logger
from pipeline.split import rolling_cv

log = get_logger("evaluate")

# Columns that are keys, labels, flags, or the split tag — never model inputs.
NON_FEATURE_COLS = {"zip", "month", "target", "target_outlier", "low_volume", "split"}

PredictFn = Callable[[pd.DataFrame, pd.DataFrame, list[str], dict], np.ndarray]


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Numeric feature columns (everything that isn't a key, label, flag, or split)."""
    return [
        c
        for c in features.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(features[c])
    ]


def _drop_flagged(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop rows carrying any configured exclusion flag (low_volume, target_outlier)."""
    mask = pd.Series(False, index=df.index)
    for flag in model_config(config)["exclude_flags"]:
        if flag in df.columns:
            mask |= df[flag].fillna(False).astype(bool)
    return df[~mask]


def prepared_folds(features: pd.DataFrame, config: dict) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """(train_df, val_df) per fold from the gapped rolling CV, flagged rows removed.

    rolling_cv already restricts to the train split, so the test holdout is never
    touched; this wrapper additionally drops flagged rows from both sides.
    """
    n_folds = model_config(config)["cv_folds"]
    folds = rolling_cv(features, config, n_folds=n_folds)
    prepared = []
    for train_idx, val_idx in folds:
        train_df = _drop_flagged(features.loc[train_idx], config)
        val_df = _drop_flagged(features.loc[val_idx], config)
        if train_df.empty or val_df.empty:
            raise RuntimeError("A CV fold is empty after flag exclusion — check the panel size.")
        prepared.append((train_df, val_df))
    return prepared


def _fold_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    """RMSE, MAE, directional accuracy, and Spearman rank corr for one month's rows.

    (Each rolling-CV fold validates on a single month, so the rank correlation is
    computed across the zips of that month — exactly the intended metric.)
    """
    resid = pred - y
    # Spearman is undefined when either side is constant (e.g. naive_zero) — report
    # NaN directly rather than triggering scipy's ConstantInputWarning.
    if np.ptp(pred) == 0 or np.ptp(y) == 0:
        rank_corr = float("nan")
    else:
        rank_corr = pd.Series(pred).corr(pd.Series(y), method="spearman")
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "dir_acc": float(np.mean(np.sign(pred) == np.sign(y))),
        "rank_corr": float(rank_corr) if pd.notna(rank_corr) else float("nan"),
        "n": int(len(y)),
    }


def cv_evaluate(features: pd.DataFrame, config: dict, predict_fn: PredictFn) -> dict:
    """Run one predictor through every prepared fold; return per-fold + pooled metrics."""
    feat_cols = feature_columns(features)
    per_fold, all_pred, all_y, monthly_rank = [], [], [], []
    for train_df, val_df in prepared_folds(features, config):
        pred = np.asarray(predict_fn(train_df, val_df, feat_cols, config), dtype=float)
        y = val_df["target"].to_numpy(dtype=float)
        m = _fold_metrics(pred, y)
        m["month"] = pd.Timestamp(val_df["month"].iloc[0]).strftime("%Y-%m")
        per_fold.append(m)
        all_pred.append(pred)
        all_y.append(y)
        if pd.notna(m["rank_corr"]):
            monthly_rank.append(m["rank_corr"])
    pred_all, y_all = np.concatenate(all_pred), np.concatenate(all_y)
    resid = pred_all - y_all
    pooled = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "dir_acc": float(np.mean(np.sign(pred_all) == np.sign(y_all))),
        "rank_corr": float(np.mean(monthly_rank)) if monthly_rank else float("nan"),
        "n": int(len(y_all)),
    }
    return {"per_fold": per_fold, "pooled": pooled}


# --- baseline predictors -------------------------------------------------------

def _predict_zero(train_df, val_df, feat_cols, config) -> np.ndarray:
    return np.zeros(len(val_df))


def _predict_momentum(train_df, val_df, feat_cols, config) -> np.ndarray:
    # Trailing 3-month price change; if unavailable (early history) assume no change.
    return val_df["price_mom_3m"].fillna(0.0).to_numpy(dtype=float)


def _predict_ridge(train_df, val_df, feat_cols, config) -> np.ndarray:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=model_config(config)["random_seed"])),
        ]
    )
    pipe.fit(train_df[feat_cols], train_df["target"])
    return pipe.predict(val_df[feat_cols])


BASELINES: dict[str, PredictFn] = {
    "naive_zero": _predict_zero,
    "naive_momentum": _predict_momentum,
    "linear": _predict_ridge,
}


# --- reporting -----------------------------------------------------------------

def _fmt(x: float, pct: bool = False) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.4f}"


def _write_report(results: dict[str, dict], config: dict, path) -> None:
    lines = [
        "# Baselines",
        "",
        f"_Generated by `python -m model evaluate --baselines-only`. "
        f"Gapped rolling CV on the train split only, "
        f"{model_config(config)['cv_folds']} folds, "
        f"{config['params']['cv_gap_months']}-month gap, "
        f"{config['params']['train_window_months']}-month train window. "
        f"Flagged rows ({', '.join(model_config(config)['exclude_flags'])}) excluded._",
        "",
        "## Pooled metrics (across all validation months)",
        "",
        "| model | RMSE | MAE | Directional acc | Rank corr (Spearman) |",
        "|---|---|---|---|---|",
    ]
    for name, res in results.items():
        p = res["pooled"]
        lines.append(
            f"| `{name}` | {_fmt(p['rmse'])} | {_fmt(p['mae'])} | "
            f"{_fmt(p['dir_acc'], pct=True)} | {_fmt(p['rank_corr'])} |"
        )
    lines += [
        "",
        "RMSE/MAE are in fraction-of-price units (0.04 = 4 percentage points of "
        "3-month price change). Directional accuracy is sign agreement between "
        "predicted and realized change. Rank correlation is the average per-month "
        "Spearman between predicted and realized change across zips — the metric "
        "closest to the manager's ranking decision.",
        "",
        "## Per-fold breakdown",
        "",
    ]
    for name, res in results.items():
        lines += [
            f"### `{name}`",
            "",
            "| fold month | n | RMSE | MAE | Directional acc | Rank corr |",
            "|---|---|---|---|---|---|",
        ]
        for m in res["per_fold"]:
            lines.append(
                f"| {m['month']} | {m['n']:,} | {_fmt(m['rmse'])} | {_fmt(m['mae'])} | "
                f"{_fmt(m['dir_acc'], pct=True)} | {_fmt(m['rank_corr'])} |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "<!-- TEAM: write one paragraph in your own words interpreting the table above. "
        "Which baseline is hardest to beat, and on which metric? What does it imply for "
        "the XGBoost model in M2 (e.g., the bar it must clear on rank correlation)? "
        "Replace this comment. -->",
        "",
        "_Interpretation pending — to be written by the team (see comment above)._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path.relative_to(REPO_ROOT))


def run_baselines(config: dict) -> dict[str, dict]:
    """Load features and score all three baselines through the shared CV harness."""
    features_path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not features_path.exists():
        raise RuntimeError(
            f"{features_path} missing. Run `python -m pipeline all` first."
        )
    features = pd.read_parquet(features_path)
    if "split" not in features.columns:
        raise RuntimeError("features.parquet has no `split` column. Run `python -m pipeline split`.")
    log.info("features: %d rows, %d train rows", len(features), int((features["split"] == "train").sum()))

    results = {}
    for name, fn in BASELINES.items():
        res = cv_evaluate(features, config, fn)
        p = res["pooled"]
        log.info(
            "%-15s pooled: RMSE=%.4f MAE=%.4f dir_acc=%.1f%% rank_corr=%s (n=%d)",
            name, p["rmse"], p["mae"], 100 * p["dir_acc"],
            f"{p['rank_corr']:.4f}" if np.isfinite(p["rank_corr"]) else "—", p["n"],
        )
        results[name] = res
    _write_report(results, config, reports_dir(config) / "baselines.md")
    return results


def run(config: dict, baselines_only: bool = False) -> None:
    # M1 only implements baselines; model comparison (M2) and holdout (M4) extend this.
    run_baselines(config)
    if not baselines_only:
        log.info("(model comparison and holdout evaluation arrive in spec_model.md M2/M4)")
