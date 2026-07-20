"""Cached loading of the artifacts the dashboard reads.

Two rules this module exists to enforce:

  1. Loading happens once. Models are `@st.cache_resource` (one live object per
     process); frames are `@st.cache_data` (copied per caller, so a widget
     mutating a table can't corrupt the cache).
  2. A missing artifact is a *message*, not a traceback. `missing_artifacts()`
     reports what is absent and the exact command that produces it, and
     `app/main.py` renders that instead of the dashboard.

No modelling here — this module locates and caches files. Scoring is
`model.predict.score`, via `model.scenario`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.io_utils import REPO_ROOT, load_config

# Artifacts the dashboard cannot start without, and the command that builds each.
POINT_MODEL = "xgb_point.joblib"
QUANTILE_MODEL = "xgb_quantiles.joblib"
FEATURES_FILE = "features.parquet"
ZIP_METRO_FILE = "zip_metro.parquet"


@dataclass(frozen=True)
class MissingArtifact:
    """One absent input, phrased for the person looking at the screen."""

    what: str
    path: str
    command: str


@st.cache_data(show_spinner=False)
def config() -> dict:
    """Parsed `config.yaml`. Every tunable the dashboard reads comes from here."""
    return load_config()


def _processed_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["paths"]["processed"]


def _models_dir(cfg: dict) -> Path:
    # Mirrors model.io.models_dir without importing the model package's own
    # loader, so a missing model file surfaces here as a message rather than an
    # exception raised deep inside predict._load.
    return REPO_ROOT / "models"


def missing_artifacts(cfg: dict) -> list[MissingArtifact]:
    """Everything the dashboard needs that isn't on disk yet, in build order."""
    models = _models_dir(cfg)
    processed = _processed_dir(cfg)
    required = [
        (
            processed / FEATURES_FILE,
            "Feature matrix (the zip-month panel the models score)",
            "python -m pipeline all",
        ),
        (
            models / POINT_MODEL,
            "Point model (the p50 prediction)",
            "python -m model train",
        ),
        (
            models / QUANTILE_MODEL,
            "Quantile models (the 80% / 90% confidence bands)",
            "python -m model train",
        ),
    ]
    return [
        MissingArtifact(what=what, path=str(path.relative_to(REPO_ROOT)), command=cmd)
        for path, what, cmd in required
        if not path.exists()
    ]


@st.cache_resource(show_spinner="Loading models…")
def warm_models() -> str:
    """Load both model artifacts once per process and report the as-of month.

    Scoring a single row forces joblib to deserialize the boosters now, so the
    first control change the user makes is already warm and the measured
    re-rank time reflects steady state rather than model load.
    """
    from model.predict import score

    cfg = config()
    feats = live_features()
    score(feats.head(1), cfg)
    return "warm"


@st.cache_data(show_spinner="Loading the latest month of features…")
def live_features() -> pd.DataFrame:
    """All zips at the most recent month — the universe every control re-scores."""
    from model.scenario import live_features as _live

    return _live(config())


@st.cache_data(show_spinner=False)
def zip_metro() -> pd.DataFrame:
    """zip -> metro lookup written by `pipeline clean`.

    Presentation-layer only: metro never enters the feature matrix. Returns an
    empty frame when the lookup is absent (older data builds), which callers
    treat as "metro filtering unavailable" rather than an error.
    """
    path = _processed_dir(config()) / ZIP_METRO_FILE
    if not path.exists():
        return pd.DataFrame({"zip": pd.Series(dtype="object"), "metro": pd.Series(dtype="object")})
    return pd.read_parquet(path)


def universe_summary(feats: pd.DataFrame, metros: pd.DataFrame) -> dict:
    """Counts the data slide quotes: rows, zips, as-of month, metro coverage."""
    as_of = feats["month"].max()
    matched = feats["zip"].isin(set(metros["zip"])).sum() if len(metros) else 0
    return {
        "rows": len(feats),
        "zips": feats["zip"].nunique(),
        "as_of": as_of,
        "as_of_label": pd.Timestamp(as_of).strftime("%B %Y"),
        "metros": int(metros["metro"].nunique()) if len(metros) else 0,
        "metro_matched": int(matched),
        "metro_unmatched": int(len(feats) - matched),
    }
