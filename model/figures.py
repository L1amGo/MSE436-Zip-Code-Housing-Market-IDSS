"""Matplotlib figures for the reports (M4 holdout, M5 SHAP).

Uses the non-interactive Agg backend so figures render headless (CI, no display).
Every figure labels both axes and carries a title — the rubric penalizes
unlabeled figures. Saved as PNG into reports/figures/.
"""

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from model.io import figures_dir  # noqa: E402
from pipeline.io_utils import REPO_ROOT, get_logger  # noqa: E402

log = get_logger("figures")


def _save(fig, config: dict, name: str) -> str:
    path = figures_dir(config) / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.relative_to(REPO_ROOT))
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def monthly_rank_corr(months, series: dict[str, list[float]], config: dict) -> str:
    """Grouped bar chart of per-month rank correlation for several models."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(months))
    width = 0.8 / max(len(series), 1)
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Holdout month")
    ax.set_ylabel("Rank correlation (Spearman, across zips)")
    ax.set_title("Per-month ranking quality on the 6-month holdout")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend(fontsize=8)
    return _save(fig, config, "holdout_rank_corr_by_month.png")


def predicted_vs_realized(pred: np.ndarray, realized: np.ndarray, config: dict, model_name="XGBoost") -> str:
    """Scatter of predicted vs realized 3-month change on the holdout, with y=x."""
    rng = np.random.default_rng(0)
    if len(pred) > 8000:  # keep the PNG light and readable
        idx = rng.choice(len(pred), 8000, replace=False)
        pred, realized = pred[idx], realized[idx]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(pred, realized, s=4, alpha=0.2, edgecolors="none")
    lim = [min(pred.min(), realized.min()), max(pred.max(), realized.max())]
    ax.plot(lim, lim, color="crimson", linewidth=1, label="perfect (y = x)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Predicted 3-month change ({model_name})")
    ax.set_ylabel("Realized 3-month change")
    ax.set_title(f"{model_name}: predicted vs realized (holdout)")
    ax.legend(fontsize=8)
    return _save(fig, config, "holdout_pred_vs_realized.png")
