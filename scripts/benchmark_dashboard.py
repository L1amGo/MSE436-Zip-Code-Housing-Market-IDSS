"""Measure the dashboard's interactive re-rank time (task D5).

The proposal claims the system is interactive; this replaces that claim with a
number. The headline metric is the one the manager actually waits on: change the
rate scenario, and how long until the whole zip universe is re-scored, re-ranked,
re-filtered and re-allocated.

    python scripts/benchmark_dashboard.py

Writes reports/dashboard_benchmark.md. Deterministic apart from timing: same
data, same config, same seed.
"""

from __future__ import annotations

import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import controls as C  # noqa: E402
from app import geo  # noqa: E402
from model.scenario import live_features  # noqa: E402
from pipeline.io_utils import load_config  # noqa: E402

RUNS = 10
TARGET_SECONDS = 2.0
# Cycled so no single run is served by a warm identical computation.
SCENARIOS_BPS = [25, 50, 100, -25, -50]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def time_rerank(features, metros, config, runs: int = RUNS) -> list[float]:
    """Time `runs` full-universe re-ranks, each under a different rate scenario."""
    timings = []
    for i in range(runs):
        bps = SCENARIOS_BPS[i % len(SCENARIOS_BPS)]
        controls = C.Controls(
            metros=(),
            min_roi=config["model"]["default_min_roi"],
            max_downside=config["model"]["default_max_downside"],
            ci_level=config["dashboard"]["default_ci_level"],
            risk_tolerance=config["dashboard"]["default_risk_tolerance"],
            rate_bps=bps,
            budget=float(config["dashboard"]["default_budget"]),
        )
        start = time.perf_counter()
        C.evaluate(controls, features, metros, config)
        timings.append(time.perf_counter() - start)
    return timings


def time_map_build(features, metros, config) -> tuple[float, int] | None:
    """Time building the choropleth for the largest single metro."""
    if not geo.cache_path(config).exists() or metros.empty:
        return None
    from app.components import map as map_view
    from app.components import theme

    largest = metros["metro"].value_counts().index[0]
    controls = C.Controls((largest,), 0.0, 0.05, 80, "Moderate", 0, 1e6)
    decision = C.evaluate(controls, features, metros, config)
    geojson = geo.zip_geojson(config)

    start = time.perf_counter()
    frame = map_view.map_frame(decision, (largest,))
    subset = geo.subset_geojson(geojson, frame["zip"])
    map_view.choropleth(frame, subset, config, theme.LIGHT, 80)
    return time.perf_counter() - start, len(frame)


def time_drilldown(config, zip_code: str) -> tuple[float, float] | None:
    """Cold and warm explanation time for one zip."""
    try:
        from app.components import drilldown
    except Exception:
        return None

    features = pd.read_parquet(
        REPO_ROOT / config["paths"]["processed"] / "features.parquet",
        columns=["zip", "month", "split"],
    )
    month = features[features["split"] == "live"]["month"].max()

    start = time.perf_counter()
    drilldown.explain(zip_code, month, config)
    cold = time.perf_counter() - start

    start = time.perf_counter()
    drilldown.explain(zip_code, month, config)
    warm = time.perf_counter() - start
    return cold, warm


def render_report(
    timings: list[float], n_zips: int, as_of, config: dict,
    map_result, drill_result, qualified: int,
) -> str:
    median = statistics.median(timings)
    p95 = _quantile(timings, 0.95)
    verdict = (
        f"**PASS** — median {median:.3f}s is within the {TARGET_SECONDS:.0f}s target."
        if median <= TARGET_SECONDS
        else f"**MISS** — median {median:.3f}s exceeds the {TARGET_SECONDS:.0f}s target."
    )

    lines = [
        "# Dashboard performance benchmark",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
        "`python scripts/benchmark_dashboard.py`.",
        "",
        "## What is measured",
        "",
        "One **re-rank**: the work behind a single control change. The rate scenario",
        "is shifted, then every zip in the universe is re-scored by the point and",
        "quantile models, ranked by risk-adjusted score, filtered against the",
        "eligibility bars, and the budget re-allocated across the survivors.",
        "",
        "Model loading is excluded — it happens once at startup, not per",
        f"interaction. {RUNS} runs, cycling through {SCENARIOS_BPS} bps so no run is",
        "served by an identical warm computation.",
        "",
        "## Headline result",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| zips re-ranked per run | {n_zips:,} |",
        f"| qualifying at defaults | {qualified:,} |",
        f"| data as of | {pd.Timestamp(as_of):%Y-%m} |",
        f"| runs | {len(timings)} |",
        f"| **median** | **{median:.3f} s** |",
        f"| p95 | {p95:.3f} s |",
        f"| min / max | {min(timings):.3f} s / {max(timings):.3f} s |",
        f"| target | {TARGET_SECONDS:.0f} s |",
        "",
        verdict,
        "",
        f"That is roughly **{median / n_zips * 1e6:.0f} microseconds per zip**, which",
        "replaces the proposal's unmeasured \"~50 ms per zip\" estimate.",
        "",
        "## Per-run timings",
        "",
        "| run | scenario (bps) | seconds |",
        "|---|---|---|",
    ]
    for i, t in enumerate(timings, 1):
        lines.append(f"| {i} | {SCENARIOS_BPS[(i - 1) % len(SCENARIOS_BPS)]:+d} | {t:.3f} |")

    lines += ["", "## Supporting measurements", "", "| operation | seconds | note |", "|---|---|---|"]
    if map_result:
        elapsed, zips = map_result
        lines.append(
            f"| choropleth build (largest metro) | {elapsed:.3f} | {zips:,} zips, "
            "geometry subset to those drawn |"
        )
    else:
        lines.append("| choropleth build | n/a | geometry cache not built |")
    if drill_result:
        cold, warm = drill_result
        lines.append(
            f"| drill-down explanation (cold) | {cold:.3f} | first call loads the "
            "feature matrix and builds the explainer |"
        )
        lines.append(f"| drill-down explanation (warm) | {warm:.3f} | subsequent zips |")
    else:
        lines.append("| drill-down explanation | n/a | not measurable in this environment |")

    lines += [
        "",
        "## Environment",
        "",
        f"- Python {platform.python_version()} on {platform.system()} {platform.machine()}",
        f"- Seed {config['model']['random_seed']}; single process, no GPU",
        "",
        "## Caveats",
        "",
        "- Server-side timings. Browser paint is on top of these, and is the reason",
        f"  the map asks for a metro filter above "
        f"{config['dashboard']['map_max_zips']:,} zips.",
        "- Measured on one machine; treat as an order of magnitude, not a guarantee.",
        "- The re-rank covers the whole universe, so it does not shrink when the",
        "  eligibility filters do — filtering happens after scoring.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    config = load_config()
    features = live_features(config)
    metro_path = REPO_ROOT / config["paths"]["processed"] / "zip_metro.parquet"
    metros = pd.read_parquet(metro_path) if metro_path.exists() else pd.DataFrame(
        {"zip": pd.Series(dtype="object"), "metro": pd.Series(dtype="object")}
    )

    defaults = C.defaults(config)
    warm = C.evaluate(defaults, features.head(50), metros, config)  # load models
    print(f"warmed models ({warm.evaluated} zips)", flush=True)

    baseline = C.evaluate(defaults, features, metros, config)
    timings = time_rerank(features, metros, config)
    print(f"re-rank median {statistics.median(timings):.3f}s over {len(timings)} runs", flush=True)

    map_result = time_map_build(features, metros, config)
    top_zip = str(baseline.ranked["zip"].iloc[0]) if len(baseline.ranked) else "00000"
    drill_result = time_drilldown(config, top_zip)

    report = render_report(
        timings, len(features), features["month"].max(), config,
        map_result, drill_result, baseline.qualified,
    )
    out = REPO_ROOT / config["paths"]["reports"] / "dashboard_benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
