"""Shared model-layer IO: config access and artifact/report paths.

Reuses the pipeline's config loader so there is one source of truth for
config.yaml. Model binaries live in `models/` (gitignored except .gitkeep);
grader-facing markdown/figures live in `reports/` (tracked).
"""

from pathlib import Path

from pipeline.io_utils import REPO_ROOT, load_config  # re-exported for the model layer

__all__ = ["REPO_ROOT", "load_config", "model_config", "models_dir", "reports_dir", "figures_dir"]


def model_config(config: dict) -> dict:
    """Return the `model:` config section, or fail loudly if it is missing."""
    section = config.get("model")
    if not section:
        raise RuntimeError(
            "config.yaml has no `model:` section. It is added in spec_model.md task M0."
        )
    return section


def models_dir(config: dict) -> Path:
    d = REPO_ROOT / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir(config: dict) -> Path:
    d = REPO_ROOT / config["paths"]["reports"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(config: dict) -> Path:
    d = reports_dir(config) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d
