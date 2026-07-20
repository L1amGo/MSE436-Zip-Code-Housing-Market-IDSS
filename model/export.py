"""Stage: export — write a ready-to-use ranked prediction table for the team.

`python -m model export` scores every zip in the most recent month with the
trained models, adds the risk-adjusted ranking score, and writes
`outputs/zip_predictions.csv` (+ a small parquet). This is the "just use the
output" artifact: teammates can open it directly — no model, no retrain — while
the importable API (predict/scenario/decide) stays available for the dashboard.
"""

import pandas as pd

from model import decide
from model.io import model_config
from model.scenario import live_features
from model.predict import score
from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("export")

OUTPUT_COLS = [
    "rank", "zip", "month", "point",
    "p05", "p10", "p50", "p90", "p95",
    "ci80_width", "ci90_width", "score",
]


def build_export(config: dict) -> pd.DataFrame:
    """Score the latest-month zip universe and attach the default risk-adjusted rank."""
    feats = live_features(config)
    preds = score(feats, config)
    ranked = decide.rank(preds, model_config(config)["risk_lambda_default"], config)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    cols = [c for c in OUTPUT_COLS if c in ranked.columns]
    return ranked[cols]


def run(config: dict) -> None:
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_export(config)
    month = pd.Timestamp(table["month"].iloc[0]).strftime("%Y-%m")

    csv_path = out_dir / "zip_predictions.csv"
    pq_path = out_dir / "zip_predictions.parquet"
    table.to_csv(csv_path, index=False)
    table.to_parquet(pq_path, index=False)
    log.info("wrote %s and %s (%d zips, month %s)", csv_path.name, pq_path.name, len(table), month)

    readme = out_dir / "README.md"
    readme.write_text(
        "# Prediction outputs\n\n"
        f"`zip_predictions.csv` / `.parquet` — every zip scored for the most recent "
        f"month (**{month}**), ranked best-first by risk-adjusted score.\n\n"
        "Columns: `rank`, `zip`, `month`, `point` (predicted 3-month % change), "
        "`p05`/`p10`/`p50`/`p90`/`p95` (quantiles), `ci80_width`/`ci90_width` "
        "(band widths; **bands are 80% and 90%**, not 95%), `score` "
        "(= p50 - risk_lambda x ci80_width, the ranking key).\n\n"
        "Regenerate with `python -m model export`. Interactive re-scoring "
        "(macro scenarios, custom budget/filters) uses the importable API — see CLAUDE.md.\n",
        encoding="utf-8",
    )
    log.info("wrote %s", readme.relative_to(REPO_ROOT))
