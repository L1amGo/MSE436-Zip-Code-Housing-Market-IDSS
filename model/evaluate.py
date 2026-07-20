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

import json
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.io import model_config, models_dir, reports_dir
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


def cv_evaluate(
    features: pd.DataFrame,
    config: dict,
    predict_fn: PredictFn,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> dict:
    """Run one predictor through every prepared fold; return per-fold + pooled metrics.

    Pass `folds` (from prepared_folds) to reuse one flag-filtered split across many
    predictors — the grid search does this so the panel is sliced only once.
    """
    feat_cols = feature_columns(features)
    if folds is None:
        folds = prepared_folds(features, config)
    per_fold, all_pred, all_y, monthly_rank = [], [], [], []
    for train_df, val_df in folds:
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


# --- model comparison report (M2) ---------------------------------------------

# Fixed display order: dumb -> trend -> simple model -> ensembles.
MODEL_ORDER = ["naive_zero", "naive_momentum", "linear", "random_forest", "lightgbm", "xgboost"]
MODEL_LABELS = {
    "naive_zero": "naive_zero",
    "naive_momentum": "naive_momentum",
    "linear": "linear (Ridge)",
    "random_forest": "RandomForest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
TRADEOFFS_MARKER = "## Trade-offs"

_DEFAULT_TRADEOFFS = f"""{TRADEOFFS_MARKER}: XGBoost for this task (DRAFT — team must edit)

<!-- TEAM: these are starter claims, not final copy. Review each one against THIS
task (zip-level 3-month price-change ranking on a smoothed Redfin panel) and
rewrite in your own words. Keep the ones you can defend in Q&A; cut or correct
the rest. Advantages AND disadvantages must both be argued. -->

**Advantages for this task**
- Handles mixed-scale tabular features (dollar prices, ratios, national macro
  levels) without manual scaling.
- Deals with missing values natively (momentum lags are structurally missing
  early in each zip's history), so no imputation choice biases the model.
- Strong on medium-sized structured panels like this one; captures non-linear
  interactions a linear model cannot.

**Disadvantages for this task**
- Opaque without post-hoc tools — mitigated by the SHAP layer in M5, but that is
  extra machinery a linear model would not need.
- Can overfit thin/low-volume zips whose medians are noisy (the low_volume flag
  exists precisely because of this).
- No native uncertainty estimate — confidence bands require the separate
  quantile models trained in M3.
"""


def _pooled_row(name: str, res: dict, best_rank: float) -> str:
    p = res["pooled"]
    star = " ★" if np.isfinite(p["rank_corr"]) and abs(p["rank_corr"] - best_rank) < 1e-12 else ""
    return (
        f"| {MODEL_LABELS.get(name, name)} | {_fmt(p['rmse'])} | {_fmt(p['mae'])} | "
        f"{_fmt(p['dir_acc'], pct=True)} | {_fmt(p['rank_corr'])}{star} |"
    )


def _comparison_head(cv_results: dict, config: dict) -> str:
    models = cv_results["models"]
    ranks = [
        models[m]["pooled"]["rank_corr"]
        for m in MODEL_ORDER
        if m in models and np.isfinite(models[m]["pooled"]["rank_corr"])
    ]
    best_rank = max(ranks) if ranks else float("nan")
    cv = cv_results["cv"]
    lines = [
        "# Model comparison",
        "",
        "> **DRAFT — team must edit before using in slides.**",
        "",
        f"_Generated by `python -m model train`. Gapped rolling CV on the train split "
        f"only ({cv['folds']} folds, {cv['gap']}-month gap, {cv['window']}-month window); "
        f"flagged rows excluded; seed {cv_results['seed']}. Model selected on pooled "
        f"**{cv_results['selection_metric']}** (★). Same seed reproduces these numbers._",
        "",
        "## Cross-validation metrics (pooled across validation months)",
        "",
        "| model | RMSE | MAE | Directional acc | Rank corr (Spearman) |",
        "|---|---|---|---|---|",
        *[_pooled_row(m, models[m], best_rank) for m in MODEL_ORDER if m in models],
        "",
        "★ = best pooled rank correlation (the selection metric — closest to the "
        "manager's ranking decision).",
        "",
        "## Chosen hyperparameters",
        "",
        f"- **XGBoost (selected):** `{models['xgboost']['params']}`",
        f"- **RandomForest (comparison):** `{models['random_forest'].get('params', {})}`",
        f"- **LightGBM (comparison):** `{models['lightgbm'].get('params', {})}`",
        "",
        "_Only XGBoost is grid-searched. RandomForest and LightGBM use fixed "
        "sensible-defaults configs as reference points; RandomForest is kept "
        "lightweight (row-subsampled trees) to fit the training-time budget, so "
        "treat its row as a defaults baseline, not a fully tuned competitor._",
        "",
        "## XGBoost grid search (every combo's CV score)",
        "",
        "| max_depth | n_estimators | learning_rate | Rank corr | RMSE |",
        "|---|---|---|---|---|",
    ]
    for g in cv_results["xgb_grid"]:
        pr = g["params"]
        lines.append(
            f"| {pr['max_depth']} | {pr['n_estimators']} | {pr['learning_rate']} | "
            f"{_fmt(g['pooled']['rank_corr'])} | {_fmt(g['pooled']['rmse'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_comparison_report(cv_results: dict, config: dict) -> None:
    """Write reports/model_comparison.md, regenerating only the auto CV content.

    Everything from the first preserved marker down is kept verbatim: the M3
    calibration section (if present) and the team-edited Trade-offs section both
    survive a `model train` re-run. Only the CV table / params / grid up top are
    rebuilt.
    """
    path = reports_dir(config) / "model_comparison.md"
    tail = "\n" + _DEFAULT_TRADEOFFS
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        markers = [existing.find(m) for m in (CALIBRATION_MARKER, TRADEOFFS_MARKER)]
        present = [i for i in markers if i != -1]
        if present:
            tail = existing[min(present):]
    path.write_text(_comparison_head(cv_results, config) + "\n" + tail, encoding="utf-8")
    log.info("wrote %s", path.relative_to(REPO_ROOT))


def run_comparison(config: dict) -> None:
    """Rebuild the comparison report from the CV results train.py cached."""
    cv_path = models_dir(config) / "cv_results.json"
    if not cv_path.exists():
        raise RuntimeError(
            f"{cv_path} missing. Run `python -m model train` first to produce CV results."
        )
    cv_results = json.loads(cv_path.read_text(encoding="utf-8"))
    for name in MODEL_ORDER:
        if name in cv_results["models"]:
            p = cv_results["models"][name]["pooled"]
            log.info(
                "%-15s RMSE=%.4f rank_corr=%s",
                name, p["rmse"],
                f"{p['rank_corr']:.4f}" if np.isfinite(p["rank_corr"]) else "—",
            )
    write_comparison_report(cv_results, config)


# --- interval calibration (M3) ------------------------------------------------

# Nominal coverage of each band, given the M3 band-label decision (p05/p95 -> 90%).
BANDS = {"80% (p10-p90)": ("p10", "p90", 0.80), "90% (p05-p95)": ("p05", "p95", 0.90)}
CALIBRATION_MARKER = "## Interval calibration"


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of realized targets falling inside [lo, hi] (bounds inclusive)."""
    return float(np.mean((y >= lo) & (y <= hi)))


def calibrate_intervals(features: pd.DataFrame, config: dict) -> dict:
    """Empirical coverage of each band on train-only gapped CV.

    For every fold a fresh multi-quantile model is fit on the fold's train rows
    and scored on its validation month; coverage is the share of realized targets
    inside each band. Quantiles are monotonised first (same fix as predict.py), so
    reported coverage matches what the dashboard will actually show.
    """
    from model.predict import QUANTILE_COLS, enforce_monotone
    from model.train import fit_quantile_model, quantile_params

    mcfg = model_config(config)
    seed, quantiles = mcfg["random_seed"], mcfg["quantiles"]
    params = quantile_params(_selected_params(config), config)
    feat_cols = feature_columns(features)
    folds = prepared_folds(features, config)

    per_fold = []
    for train_df, val_df in folds:
        qmodel = fit_quantile_model(train_df, feat_cols, params, quantiles, seed)
        qpred = enforce_monotone(qmodel.predict(val_df[feat_cols]))
        qdf = pd.DataFrame(qpred, columns=QUANTILE_COLS)
        y = val_df["target"].to_numpy(dtype=float)
        row = {"month": pd.Timestamp(val_df["month"].iloc[0]).strftime("%Y-%m"), "n": int(len(y))}
        for label, (lo, hi, _nom) in BANDS.items():
            row[label] = _coverage(y, qdf[lo].to_numpy(), qdf[hi].to_numpy())
        per_fold.append(row)

    pooled = {
        label: float(np.mean([f[label] for f in per_fold])) for label in BANDS
    }
    return {"per_fold": per_fold, "pooled": pooled, "params": params}


def _selected_params(config: dict) -> dict:
    """Selected XGBoost params from the cached M2 CV results (fail loudly if absent)."""
    cv_path = models_dir(config) / "cv_results.json"
    if not cv_path.exists():
        raise RuntimeError(
            f"{cv_path} missing. Run `python -m model train` first (M2) to select params."
        )
    return json.loads(cv_path.read_text(encoding="utf-8"))["models"]["xgboost"]["params"]


def _calibration_section(cal: dict) -> str:
    lines = [
        CALIBRATION_MARKER,
        "",
        "Empirical coverage of each confidence band on the train-only gapped CV "
        "(share of realized 3-month changes that fell inside the band). Nominal "
        "coverage is the band's label; a well-calibrated band matches it.",
        "",
        "**Band-label decision (M3):** the wider p05-p95 band is labelled **90%** "
        "(its true nominal coverage), not 95%. We kept p05/p95 rather than training "
        "p025/p975 because the less-extreme tails calibrate more reliably on noisy "
        "zip-level data. Dashboard and slide labels must read **80% / 90%**.",
        "",
        f"_Band models: multi-quantile XGBoost with the M2-selected depth/learning-rate "
        f"but {cal['params'].get('n_estimators')} trees (config `model.quantile_n_estimators`) "
        f"— fewer than the point model, since the bands feed relative risk ranking rather "
        f"than the headline point accuracy._",
        "",
        "| band | nominal | empirical (pooled) | gap |",
        "|---|---|---|---|",
    ]
    for label, (_lo, _hi, nom) in BANDS.items():
        emp = cal["pooled"][label]
        lines.append(f"| {label} | {nom:.0%} | {emp:.1%} | {emp - nom:+.1%} |")
    worst = max(abs(cal["pooled"][l] - BANDS[l][2]) for l in BANDS)
    note = (
        "Both bands calibrate within ±10 points of nominal."
        if worst <= 0.10
        else "At least one band is miscalibrated by more than 10 points — see the gap "
        "column. On thin/low-volume zips the quantile spread understates true "
        "dispersion, so treat the band as indicative, not exact; the decision layer "
        "(M6) uses band width for relative risk ranking, which tolerates a level bias."
    )
    lines += ["", note, "", "### Per-fold coverage", "",
              "| fold month | n | " + " | ".join(BANDS) + " |",
              "|---|---|" + "|".join(["---"] * len(BANDS)) + "|"]
    for f in cal["per_fold"]:
        lines.append(
            f"| {f['month']} | {f['n']:,} | "
            + " | ".join(f"{f[label]:.1%}" for label in BANDS)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def run_intervals(config: dict) -> dict:
    """Compute band calibration and append it to reports/model_comparison.md."""
    features_path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not features_path.exists():
        raise RuntimeError(f"{features_path} missing. Run `python -m pipeline all` first.")
    features = pd.read_parquet(features_path)
    cal = calibrate_intervals(features, config)
    for label in BANDS:
        log.info("band %-14s pooled coverage=%.1f%% (nominal %.0f%%)",
                 label, 100 * cal["pooled"][label], 100 * BANDS[label][2])

    # Insert/replace the calibration section just before the Trade-offs section so
    # team edits to Trade-offs are preserved and the auto content stays together.
    path = reports_dir(config) / "model_comparison.md"
    if not path.exists():
        raise RuntimeError(f"{path} missing. Run `python -m model train` (M2) first.")
    text = path.read_text(encoding="utf-8")
    section = _calibration_section(cal) + "\n"
    if CALIBRATION_MARKER in text:  # replace existing calibration block
        head = text[: text.index(CALIBRATION_MARKER)]
        rest = text[text.index(CALIBRATION_MARKER):]
        tail = rest[rest.index(TRADEOFFS_MARKER):] if TRADEOFFS_MARKER in rest else ""
        text = head + section + tail
    elif TRADEOFFS_MARKER in text:  # insert above Trade-offs
        idx = text.index(TRADEOFFS_MARKER)
        text = text[:idx] + section + "\n" + text[idx:]
    else:
        text = text + "\n" + section
    path.write_text(text, encoding="utf-8")
    log.info("wrote calibration section to %s", path.relative_to(REPO_ROOT))
    return cal


# --- holdout evaluation (M4): touch the 6-month test split exactly once --------

HOLDOUT_MODEL_ORDER = MODEL_ORDER  # same display order as the CV comparison


def _holdout_predictors(config: dict) -> dict:
    """All models as (train_df, test_df, feat_cols, config) -> predictions callables.

    Ensembles are refit on the full train split here (deterministic, same seed +
    params as CV), then scored once on the holdout.
    """
    from model.train import _lgbm_factory, _rf_factory, _xgb_factory

    seed = model_config(config)["random_seed"]
    return {
        "naive_zero": _predict_zero,
        "naive_momentum": _predict_momentum,
        "linear": _predict_ridge,
        "random_forest": _rf_factory(seed),
        "lightgbm": _lgbm_factory(seed),
        "xgboost": _xgb_factory(_selected_params(config), seed),
    }


def _pooled_and_monthly(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """Pooled + per-month metrics for one model's holdout predictions."""
    df = test[["month", "target"]].copy()
    df["pred"] = pred
    per_month, monthly_rank = [], []
    for month, g in df.groupby("month"):
        m = _fold_metrics(g["pred"].to_numpy(float), g["target"].to_numpy(float))
        m["month"] = pd.Timestamp(month).strftime("%Y-%m")
        per_month.append(m)
        if pd.notna(m["rank_corr"]):
            monthly_rank.append(m["rank_corr"])
    resid = df["pred"].to_numpy(float) - df["target"].to_numpy(float)
    pooled = {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "dir_acc": float(np.mean(np.sign(df["pred"]) == np.sign(df["target"]))),
        "rank_corr": float(np.mean(monthly_rank)) if monthly_rank else float("nan"),
        "n": int(len(df)),
    }
    return {"pooled": pooled, "per_month": per_month}


def _holdout_report(results: dict, test_months, headline: str, fig_paths: list[str], config: dict) -> str:
    lines = [
        "# Holdout results",
        "",
        f"_Generated by `python -m model evaluate --holdout`. The 6-month temporal "
        f"holdout ({test_months[0]} to {test_months[-1]}) is scored exactly once here; "
        f"every model is refit on the full train split with its CV-chosen config, seed "
        f"{model_config(config)['random_seed']}. Flagged rows excluded from headline "
        f"metrics (invariant 5)._",
        "",
        "## Headline: did the model beat the naive baseline?",
        "",
        headline,
        "",
        "## Pooled holdout metrics",
        "",
        "| model | RMSE | MAE | Directional acc | Rank corr |",
        "|---|---|---|---|---|",
    ]
    ranks = [results[m]["pooled"]["rank_corr"] for m in HOLDOUT_MODEL_ORDER
             if m in results and np.isfinite(results[m]["pooled"]["rank_corr"])]
    best = max(ranks) if ranks else float("nan")
    for name in HOLDOUT_MODEL_ORDER:
        if name not in results:
            continue
        p = results[name]["pooled"]
        star = " ★" if np.isfinite(p["rank_corr"]) and abs(p["rank_corr"] - best) < 1e-12 else ""
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {_fmt(p['rmse'])} | {_fmt(p['mae'])} | "
            f"{_fmt(p['dir_acc'], pct=True)} | {_fmt(p['rank_corr'])}{star} |"
        )
    lines += ["", "## Monthly rank correlation (is performance stable?)", "",
              "| month | " + " | ".join(MODEL_LABELS.get(m, m) for m in HOLDOUT_MODEL_ORDER if m in results) + " |",
              "|---|" + "|".join(["---"] * len(results)) + "|"]
    for i, month in enumerate([pd.Timestamp(m).strftime("%Y-%m") for m in test_months]):
        row = [month]
        for name in HOLDOUT_MODEL_ORDER:
            if name in results:
                row.append(_fmt(results[name]["per_month"][i]["rank_corr"]))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Figures", ""]
    for fp in fig_paths:
        lines.append(f"![{fp}]({fp.split('/')[-1]})")
    lines += ["", "_Test split is frozen after this task; later tasks must not recompute "
              "holdout metrics with different models (spec_model.md M4)._", ""]
    return "\n".join(lines)


def run_holdout(config: dict) -> dict:
    """Refit every model on full train, score once on the 6-month holdout, write report + figures."""
    from model import figures

    features_path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not features_path.exists():
        raise RuntimeError(f"{features_path} missing. Run `python -m pipeline all` first.")
    features = pd.read_parquet(features_path)
    train_full = _drop_flagged(features[features["split"] == "train"], config)
    test = _drop_flagged(features[features["split"] == "test"], config)
    if test.empty:
        raise RuntimeError("Holdout (test split) is empty after flag exclusion.")
    feat_cols = feature_columns(features)
    test_months = sorted(test["month"].unique())
    log.info("holdout: %d test rows over %d months (%s..%s); %d train rows",
             len(test), len(test_months),
             pd.Timestamp(test_months[0]).strftime("%Y-%m"),
             pd.Timestamp(test_months[-1]).strftime("%Y-%m"), len(train_full))

    results, preds = {}, {}
    for name, fn in _holdout_predictors(config).items():
        pred = np.asarray(fn(train_full, test, feat_cols, config), dtype=float)
        preds[name] = pred
        results[name] = _pooled_and_monthly(test, pred)
        log.info("  %-15s holdout rank_corr=%s rmse=%.4f", name,
                 f"{results[name]['pooled']['rank_corr']:.4f}"
                 if np.isfinite(results[name]["pooled"]["rank_corr"]) else "—",
                 results[name]["pooled"]["rmse"])

    xgb_r = results["xgboost"]["pooled"]["rank_corr"]
    mom_r = results["naive_momentum"]["pooled"]["rank_corr"]
    delta = xgb_r - mom_r
    verb = "beat" if delta > 0 else "did NOT beat"
    headline = (
        f"On the holdout, **XGBoost {verb} the naive-momentum baseline** on rank "
        f"correlation: {xgb_r:.4f} vs {mom_r:.4f} (a {delta:+.4f} difference). "
        f"Rank correlation is the decision-relevant metric (the manager acts on the "
        f"ranking of zips). "
        + (
            "XGBoost also has the lowest holdout RMSE among all models."
            if results["xgboost"]["pooled"]["rmse"] == min(
                results[m]["pooled"]["rmse"] for m in results
            )
            else "Note another model has a lower holdout RMSE — see the table."
        )
    )

    months_str = [pd.Timestamp(m).strftime("%Y-%m") for m in test_months]
    fig_paths = [
        figures.monthly_rank_corr(
            months_str,
            {
                "XGBoost": [results["xgboost"]["per_month"][i]["rank_corr"] for i in range(len(test_months))],
                "naive_momentum": [results["naive_momentum"]["per_month"][i]["rank_corr"] for i in range(len(test_months))],
            },
            config,
        ),
        figures.predicted_vs_realized(preds["xgboost"], test["target"].to_numpy(float), config),
    ]

    report = _holdout_report(results, test_months, headline, fig_paths, config)
    path = reports_dir(config) / "holdout_results.md"
    path.write_text(report, encoding="utf-8")
    log.info("wrote %s", path.relative_to(REPO_ROOT))
    return results


def run(config: dict, baselines_only: bool = False, intervals: bool = False, holdout: bool = False) -> None:
    if baselines_only:
        run_baselines(config)
    elif intervals:
        run_intervals(config)
    elif holdout:
        run_holdout(config)
    else:
        run_comparison(config)
