"""M2 unit tests for the pure grid-expansion and selection helpers (no model fitting)."""

import math

from model.train import expand_grid, select_best


def test_expand_grid_is_cartesian_product():
    grid = {"max_depth": [3, 5], "n_estimators": [200, 400], "learning_rate": [0.1]}
    combos = expand_grid(grid)
    assert len(combos) == 4  # 2 x 2 x 1
    assert {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.1} in combos
    assert {"max_depth": 5, "n_estimators": 400, "learning_rate": 0.1} in combos


def test_select_best_maximizes_the_metric():
    results = [
        {"params": {"a": 1}, "pooled": {"rank_corr": 0.40}},
        {"params": {"a": 2}, "pooled": {"rank_corr": 0.55}},
        {"params": {"a": 3}, "pooled": {"rank_corr": 0.50}},
    ]
    assert select_best(results)["params"] == {"a": 2}


def test_select_best_ignores_nan_scores():
    results = [
        {"params": {"a": 1}, "pooled": {"rank_corr": float("nan")}},
        {"params": {"a": 2}, "pooled": {"rank_corr": 0.30}},
    ]
    assert select_best(results)["params"] == {"a": 2}


def test_select_best_breaks_ties_by_grid_order():
    results = [
        {"params": {"a": 1}, "pooled": {"rank_corr": 0.50}},
        {"params": {"a": 2}, "pooled": {"rank_corr": 0.50}},
    ]
    # max() keeps the first max on ties -> deterministic selection.
    assert select_best(results)["params"] == {"a": 1}
