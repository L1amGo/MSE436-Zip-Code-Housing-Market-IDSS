"""Stage: retrain — monthly refresh cycle with monitoring and versioning (task M7).

`python -m model retrain` runs the full operational cycle:
  1. refresh data (pipeline download -> ... -> split; download is cached)
  2. retrain point + quantile models on the rolling 36-month window
  3. monitoring metrics on the most recent labeled month (gapped, out-of-sample)
  4. feature-drift stats (standardized mean shift vs the training distribution)
  5. version the artifacts (timestamped files + a `latest` pointer) and append a
     dated entry to reports/retrain_log.md

Degradation gate: if fresh RMSE > model.degradation_rmse_multiplier x the
baseline RMSE, print a DEGRADATION ALERT, log it, and exit nonzero WITHOUT
promoting `latest` (a scheduler then registers the run as failed). The degraded
artifact is still saved for inspection; `--accept-degraded` promotes it anyway.

The degradation/promotion logic is factored into pure functions so it is unit-
tested without a multi-minute retrain.
"""

import datetime
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from model.evaluate import _drop_flagged, _fold_metrics, _selected_params, feature_columns
from model.io import model_config, models_dir, reports_dir
from model.train import fit_quantile_model, quantile_params
from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("retrain")

DEGRADED_EXIT = 3  # distinct nonzero code so a scheduler can tell degradation from a crash


# --- pure, unit-testable decision logic ---------------------------------------

def check_degradation(fresh_rmse: float, baseline_rmse: float, multiplier: float) -> bool:
    """True if the fresh model's RMSE has blown past the degradation threshold."""
    return bool(np.isfinite(fresh_rmse) and fresh_rmse > multiplier * baseline_rmse)


def decide_promotion(degraded: bool, accept_degraded: bool) -> bool:
    """Promote `latest` only if not degraded, or the operator forces it."""
    return (not degraded) or accept_degraded


def resolve_outcome(fresh_rmse, baseline_rmse, multiplier, accept_degraded):
    """(degraded, promote, exit_code, alert) for a retrain run — no side effects."""
    degraded = check_degradation(fresh_rmse, baseline_rmse, multiplier)
    promote = decide_promotion(degraded, accept_degraded)
    exit_code = DEGRADED_EXIT if (degraded and not promote) else 0
    alert = None
    if degraded:
        alert = (
            f"DEGRADATION ALERT: fresh RMSE {fresh_rmse:.4f} exceeds "
            f"{multiplier:.1f}x baseline {baseline_rmse:.4f} "
            f"(= {multiplier * baseline_rmse:.4f}). "
            + ("Promoted anyway via --accept-degraded." if promote
               else "latest NOT promoted; previous model stays live.")
        )
    return degraded, promote, exit_code, alert


# --- pipeline refresh, training, monitoring, drift ----------------------------

def refresh_data(config: dict) -> None:
    """Re-run pipeline download -> split for fresh data (download is cached)."""
    from pipeline import clean, download, featurize, join, split

    for stage in (download, clean, join, featurize, split):
        stage.run(config)


def _load_features(config: dict) -> pd.DataFrame:
    path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    return pd.read_parquet(path)


def rolling_window(features: pd.DataFrame, months: int) -> pd.DataFrame:
    """Most recent `months` labeled months across all zips (the production window)."""
    labeled = features[features["target"].notna()]
    recent = sorted(labeled["month"].unique())[-months:]
    return labeled[labeled["month"].isin(recent)]


def _baseline_rmse(config: dict) -> float:
    """Reference RMSE for the degradation gate: the M2 CV-selected XGBoost pooled RMSE."""
    cv_path = models_dir(config) / "cv_results.json"
    if not cv_path.exists():
        raise RuntimeError(f"{cv_path} missing. Run `python -m model train` (M2) first.")
    return float(json.loads(cv_path.read_text(encoding="utf-8"))["models"]["xgboost"]["pooled"]["rmse"])


def drift_stats(window: pd.DataFrame, reference: pd.DataFrame, feat_cols: list[str], top: int = 8) -> pd.Series:
    """Standardized mean shift |(mean_window - mean_ref)/std_ref| per feature, top-N."""
    ref_mean, ref_std = reference[feat_cols].mean(), reference[feat_cols].std()
    win_mean = window[feat_cols].mean()
    z = ((win_mean - ref_mean) / ref_std.replace(0, np.nan)).abs()
    return z.sort_values(ascending=False).head(top)


def _monitor_metrics(window: pd.DataFrame, feat_cols: list[str], params: dict, seed: int, gap: int) -> dict:
    """Out-of-sample metrics on the most recent labeled month (gapped from training)."""
    months = sorted(window["month"].unique())
    val_month = months[-1]
    train_end = pd.Timestamp(val_month) - pd.DateOffset(months=gap)
    mon_train = window[window["month"] <= train_end]
    mon_val = window[window["month"] == val_month]
    if mon_train.empty or mon_val.empty:
        raise RuntimeError("Monitoring split is empty — window too short for the CV gap.")
    m = XGBRegressor(**params, random_state=seed, tree_method="hist", n_jobs=-1,
                     objective="reg:squarederror")
    m.fit(mon_train[feat_cols], mon_train["target"])
    pred = m.predict(mon_val[feat_cols])
    metrics = _fold_metrics(pred, mon_val["target"].to_numpy(float))
    metrics["month"] = pd.Timestamp(val_month).strftime("%Y-%m")
    return metrics


def _version_and_promote(config, point_model, qmodel, meta: dict, promote: bool) -> dict:
    """Save timestamped artifacts; if promoting, update the canonical `latest` files + pointer."""
    mdir = models_dir(config)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    point_path = mdir / f"xgb_point_{ts}.joblib"
    quant_path = mdir / f"xgb_quantiles_{ts}.joblib"
    joblib.dump({**meta["point"], "model": point_model}, point_path)
    joblib.dump({**meta["quant"], "model": qmodel}, quant_path)

    latest_path = mdir / "latest.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else {"promoted": None, "history": []}
    latest["history"].append({"timestamp": ts, "promoted": promote, "monitor": meta["monitor"]})
    if promote:
        shutil.copyfile(point_path, mdir / "xgb_point.joblib")
        shutil.copyfile(quant_path, mdir / "xgb_quantiles.joblib")
        latest["promoted"] = ts
    latest_path.write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return {"timestamp": ts, "point": point_path.name, "quant": quant_path.name}


def _append_log(config, entry: str) -> None:
    path = reports_dir(config) / "retrain_log.md"
    header = "# Retrain log\n\nOne dated entry per `python -m model retrain` run.\n"
    text = path.read_text(encoding="utf-8") if path.exists() else header
    path.write_text(text + entry, encoding="utf-8")
    log.info("appended entry to %s", path.relative_to(REPO_ROOT))


def _log_entry(ts, window_months, monitor, baseline_rmse, degraded, promote, drift, versions, alert) -> str:
    lines = [
        f"\n## {ts}",
        "",
        f"- Window: most recent {window_months} labeled months, "
        f"monitor month **{monitor['month']}** (out-of-sample, gapped).",
        f"- Monitoring metrics: RMSE **{monitor['rmse']:.4f}**, MAE {monitor['mae']:.4f}, "
        f"directional acc {100 * monitor['dir_acc']:.1f}%, rank corr "
        + (f"{monitor['rank_corr']:.4f}" if np.isfinite(monitor['rank_corr']) else "—") + ".",
        f"- Degradation gate: baseline RMSE {baseline_rmse:.4f}; "
        f"status **{'DEGRADED' if degraded else 'OK'}**; "
        f"`latest` {'promoted' if promote else 'NOT promoted'} "
        f"(artifacts `{versions['point']}`, `{versions['quant']}`).",
        "- Top feature drift (|standardized mean shift| vs training distribution):",
        "",
        "  | feature | z-shift |",
        "  |---|---|",
        *[f"  | `{f}` | {v:.2f} |" for f, v in drift.items()],
        "",
    ]
    if alert:
        lines += [f"> **{alert}**", ""]
    return "\n".join(lines)


def run(config: dict, accept_degraded: bool = False) -> int:
    """Full retrain cycle. Returns an exit code (0 ok, 3 degraded-and-not-promoted)."""
    mcfg = model_config(config)
    seed = mcfg["random_seed"]
    window_months = config["params"]["train_window_months"]
    gap = config["params"]["cv_gap_months"]
    quantiles = mcfg["quantiles"]

    log.info("retrain: refreshing data (download -> split) ...")
    refresh_data(config)
    features = _load_features(config)
    feat_cols = feature_columns(features)
    params = _selected_params(config)
    qparams = quantile_params(params, config)

    window = _drop_flagged(rolling_window(features, window_months), config)
    log.info("retrain window: %d rows over %d months", len(window), window["month"].nunique())

    monitor = _monitor_metrics(window, feat_cols, params, seed, gap)
    baseline_rmse = _baseline_rmse(config)
    log.info("monitor month %s: RMSE=%.4f (baseline %.4f)", monitor["month"], monitor["rmse"], baseline_rmse)

    degraded, promote, exit_code, alert = resolve_outcome(
        monitor["rmse"], baseline_rmse, mcfg["degradation_rmse_multiplier"], accept_degraded
    )

    # Fit the deployable point + quantile models on the full window.
    log.info("fitting deployable point + quantile models on the window ...")
    point = XGBRegressor(**params, random_state=seed, tree_method="hist", n_jobs=-1,
                         objective="reg:squarederror")
    point.fit(window[feat_cols], window["target"])
    qmodel = fit_quantile_model(window, feat_cols, qparams, quantiles, seed)

    drift = drift_stats(window, features[features["split"] == "train"], feat_cols)
    meta = {
        "point": {"params": params, "feature_cols": feat_cols, "seed": seed,
                  "trained_on": {"rows": int(len(window)), "window_months": window_months}},
        "quant": {"params": qparams, "quantiles": quantiles, "feature_cols": feat_cols, "seed": seed,
                  "trained_on": {"rows": int(len(window)), "window_months": window_months}},
        "monitor": monitor,
    }
    versions = _version_and_promote(config, point, qmodel, meta, promote)
    _append_log(config, _log_entry(versions["timestamp"], window_months, monitor, baseline_rmse,
                                   degraded, promote, drift, versions, alert))

    if alert:
        log.warning(alert)
        print(alert)
    log.info("retrain complete: status=%s promoted=%s exit=%d",
             "DEGRADED" if degraded else "OK", promote, exit_code)
    return exit_code
