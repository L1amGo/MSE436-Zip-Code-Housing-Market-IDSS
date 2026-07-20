"""Stage: train — XGBoost regressor selected by gapped rolling CV, vs RF/LightGBM (M2).

Grid-searches xgb_param_grid (train split only, gapped rolling CV, flagged rows
excluded — invariants 1 and 5), selecting on pooled rank correlation (the
decision-relevant metric per reports/baselines.md). RandomForest and LightGBM —
the comparison models named in the proposal — are scored through the SAME
harness with modest fixed configs. The best XGBoost is refit on the full train
split and saved to models/ with its chosen params; all CV metrics are cached to
models/cv_results.json and rendered into reports/model_comparison.md.

Determinism: fixed seeds everywhere, so the same data + config reproduce
identical CV metrics.
"""

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from model.evaluate import (
    BASELINES,
    _drop_flagged,
    cv_evaluate,
    feature_columns,
    prepared_folds,
    write_comparison_report,
)
from model.io import model_config, models_dir
from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("train")

SELECTION_METRIC = "rank_corr"  # maximize; the manager acts on rankings

# Modest fixed configs for the two comparison ensembles (not grid-searched — they
# exist to justify XGBoost, not to be tuned). RandomForest is deliberately kept
# lightweight (fewer/shallower trees, and each tree sees a 40% row subsample via
# max_samples) because a full-depth 200-tree forest on a ~430k-row-per-fold panel
# runs for tens of minutes and blows the training-time budget; this config is a
# fair "sensible-defaults" reference and finishes in a couple of minutes.
RF_PARAMS = {
    "n_estimators": 120,
    "max_depth": 12,
    "min_samples_leaf": 100,
    "max_samples": 0.4,
    "bootstrap": True,
}
LGBM_PARAMS = {"n_estimators": 400, "num_leaves": 63, "learning_rate": 0.05}


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a {param: [values]} grid into a list of param dicts."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def select_best(grid_results: list[dict], metric: str = SELECTION_METRIC) -> dict:
    """Pick the grid entry with the highest finite pooled `metric` (deterministic:
    ties broken by original grid order)."""

    def key(entry: dict) -> float:
        v = entry["pooled"][metric]
        return v if np.isfinite(v) else -np.inf

    return max(grid_results, key=key)


def quantile_params(selected_params: dict, config: dict) -> dict:
    """Band-model params: the selected point-model params with the tree count swapped
    for the (smaller) quantile-specific `model.quantile_n_estimators`."""
    return {**selected_params, "n_estimators": model_config(config)["quantile_n_estimators"]}


def fit_quantile_model(train_df, feat_cols, params: dict, quantiles: list[float], seed: int):
    """One multi-quantile XGBoost (reg:quantileerror) fit for all configured quantiles.

    xgboost >= 2.0 fits every quantile in a single model whose predict() returns a
    (n_rows, n_quantiles) matrix — far cheaper than one model per quantile, and the
    shared trees keep the quantiles better-behaved. Row order of the output columns
    matches `quantiles`.
    """
    m = XGBRegressor(
        **params,
        objective="reg:quantileerror",
        quantile_alpha=np.array(quantiles, dtype=float),
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
    )
    m.fit(train_df[feat_cols], train_df["target"])
    return m


def _xgb_factory(params: dict, seed: int):
    def predict(train_df, val_df, feat_cols, config) -> np.ndarray:
        m = XGBRegressor(
            **params,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            objective="reg:squarederror",
        )
        m.fit(train_df[feat_cols], train_df["target"])
        return m.predict(val_df[feat_cols])

    return predict


def _rf_factory(seed: int):
    def predict(train_df, val_df, feat_cols, config) -> np.ndarray:
        # RandomForest can't ingest NaN; impute (trees split on the median-filled value).
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(**RF_PARAMS, random_state=seed, n_jobs=-1)),
            ]
        )
        pipe.fit(train_df[feat_cols], train_df["target"])
        return pipe.predict(val_df[feat_cols])

    return predict


def _lgbm_factory(seed: int):
    def predict(train_df, val_df, feat_cols, config) -> np.ndarray:
        m = LGBMRegressor(
            **LGBM_PARAMS,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        m.fit(train_df[feat_cols], train_df["target"])
        return m.predict(val_df[feat_cols])

    return predict


def _load_features(config: dict) -> pd.DataFrame:
    path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not path.exists():
        raise RuntimeError(f"{path} missing. Run `python -m pipeline all` first.")
    features = pd.read_parquet(path)
    if "split" not in features.columns:
        raise RuntimeError("features.parquet has no `split` column. Run `python -m pipeline split`.")
    return features


def run(config: dict) -> None:
    mcfg = model_config(config)
    seed = mcfg["random_seed"]
    features = _load_features(config)
    feat_cols = feature_columns(features)
    folds = prepared_folds(features, config)  # slice the panel once; reused by every model
    log.info(
        "train: %d rows, %d features, %d CV folds (val months %s)",
        len(features), len(feat_cols), len(folds),
        ", ".join(pd.Timestamp(v["month"].iloc[0]).strftime("%Y-%m") for _, v in folds),
    )

    models: dict[str, dict] = {}

    # Baselines + comparison ensembles, all through the identical CV harness.
    for name, fn in BASELINES.items():
        models[name] = cv_evaluate(features, config, fn, folds)
    log.info("baselines scored; fitting comparison ensembles ...")
    models["random_forest"] = {**cv_evaluate(features, config, _rf_factory(seed), folds), "params": RF_PARAMS}
    log.info("  RandomForest done")
    models["lightgbm"] = {**cv_evaluate(features, config, _lgbm_factory(seed), folds), "params": LGBM_PARAMS}
    log.info("  LightGBM done")

    # XGBoost grid search.
    grid = expand_grid(mcfg["xgb_param_grid"])
    log.info("XGBoost grid search: %d combos x %d folds", len(grid), len(folds))
    xgb_grid = []
    for i, params in enumerate(grid, 1):
        res = cv_evaluate(features, config, _xgb_factory(params, seed), folds)
        p = res["pooled"]
        log.info(
            "  [%2d/%d] %s -> rank_corr=%.4f rmse=%.4f",
            i, len(grid), params, p["rank_corr"], p["rmse"],
        )
        xgb_grid.append({"params": params, "pooled": p, "per_fold": res["per_fold"]})

    best = select_best(xgb_grid)
    log.info("selected XGBoost params (max %s): %s -> %.4f",
             SELECTION_METRIC, best["params"], best["pooled"][SELECTION_METRIC])
    models["xgboost"] = {"pooled": best["pooled"], "per_fold": best["per_fold"], "params": best["params"]}

    # Refit the selected model on the FULL train split (flagged rows excluded) and save.
    train_full = _drop_flagged(features[features["split"] == "train"], config)
    final = XGBRegressor(
        **best["params"], random_state=seed, tree_method="hist", n_jobs=-1,
        objective="reg:squarederror",
    )
    final.fit(train_full[feat_cols], train_full["target"])
    artifact = {
        "model": final,
        "params": best["params"],
        "feature_cols": feat_cols,
        "seed": seed,
        "cv_pooled": best["pooled"],
        "trained_on": {"rows": int(len(train_full)), "split": "train (flagged excluded)"},
    }
    mdir = models_dir(config)
    joblib.dump(artifact, mdir / "xgb_point.joblib")
    log.info("saved models/xgb_point.joblib (fit on %d train rows)", len(train_full))

    # Quantile models (M3): same selected params + full train split, one multi-quantile
    # fit for the confidence bands. predict.py loads this alongside the point model.
    quantiles = mcfg["quantiles"]
    qparams = quantile_params(best["params"], config)
    qmodel = fit_quantile_model(train_full, feat_cols, qparams, quantiles, seed)
    joblib.dump(
        {
            "model": qmodel,
            "quantiles": quantiles,
            "params": qparams,
            "feature_cols": feat_cols,
            "seed": seed,
            "trained_on": {"rows": int(len(train_full)), "split": "train (flagged excluded)"},
        },
        mdir / "xgb_quantiles.joblib",
    )
    log.info("saved models/xgb_quantiles.joblib (quantiles %s)", quantiles)

    # Cache all CV results for the evaluate stage + write the comparison report.
    cv_results = {
        "seed": seed,
        "selection_metric": SELECTION_METRIC,
        "cv": {
            "folds": len(folds),
            "gap": config["params"]["cv_gap_months"],
            "window": config["params"]["train_window_months"],
        },
        "models": models,
        "xgb_grid": [{"params": g["params"], "pooled": g["pooled"]} for g in xgb_grid],
    }
    (mdir / "cv_results.json").write_text(json.dumps(cv_results, indent=2), encoding="utf-8")
    write_comparison_report(cv_results, config)
