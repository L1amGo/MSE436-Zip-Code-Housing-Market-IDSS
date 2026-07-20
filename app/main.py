"""Zip-code housing IDSS — Streamlit entry point.

Run with:  streamlit run app/main.py

Layout only. Every figure on screen is produced by a `model/` call; this module
decides where it goes and what it is labelled.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# `streamlit run app/main.py` puts app/ on sys.path, not the repo root, so the
# `app.` and `model.` packages are invisible without this. Must come before any
# first-party import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from app import controls as C  # noqa: E402
from app import state
from app.components import drilldown
from app.components import map as map_view
from app.components import sidebar, table

PAGE_TITLE = "Zip-Code Housing IDSS"


def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🏘️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_missing(missing: list[state.MissingArtifact]) -> None:
    """The friendly stand-in for a traceback when an input isn't built yet."""
    st.title(PAGE_TITLE)
    st.error(
        f"The dashboard can't start: {len(missing)} required "
        f"{'artifact is' if len(missing) == 1 else 'artifacts are'} missing."
    )
    st.write("Build them by running these commands from the repository root, in order:")
    for item in missing:
        st.markdown(f"**{item.what}**")
        st.caption(f"expected at `{item.path}`")
        st.code(item.command, language="bash")
    st.info(
        "Rebuilding the feature matrix needs a free FRED API key in `.env` — "
        "see `.env.example` and the README. The trained models are committed, so "
        "`python -m model train` is only needed if `models/` has been cleared."
    )


def render_header(summary: dict) -> None:
    """Universe scope: what was scored, and as of when."""
    st.title(PAGE_TITLE)
    st.caption(
        "Risk-adjusted zip selection and budget allocation for the coming quarter, "
        "driven by a 3-month-ahead median sale price forecast."
    )

    cols = st.columns(4)
    cols[0].metric("Zips scored", f"{summary['zips']:,}")
    cols[1].metric("Rows in live slice", f"{summary['rows']:,}")
    cols[2].metric("Data as of", summary["as_of_label"])
    cols[3].metric("Metros covered", f"{summary['metros']:,}")

    if summary["metros"] == 0:
        st.warning(
            "Metro lookup unavailable — metro filtering is disabled. "
            "Re-run `python -m pipeline clean` to build `zip_metro.parquet`."
        )
    elif summary["metro_unmatched"]:
        # Coverage honesty (C3): unmatched zips are reported, never silently dropped.
        st.caption(
            f"{summary['metro_matched']:,} of {summary['rows']:,} zips matched to a metro; "
            f"{summary['metro_unmatched']:,} unmatched and shown as “Unknown”."
        )


def render_decision_summary(decision, user, config: dict, elapsed: float) -> None:
    """Header strip: what the current control state did to the decision."""
    lam = C.risk_lambda(user, config)
    scenario = "no change" if user.rate_bps == 0 else f"{user.rate_bps:+d} bps"

    cols = st.columns(4)
    cols[0].metric("Zips evaluated", f"{decision.evaluated:,}")
    cols[1].metric("Qualifying", f"{decision.qualified:,}")
    cols[2].metric("Budget deployed", f"${decision.deployed:,.0f}")
    cols[3].metric("Unallocated", f"${decision.unallocated:,.0f}")

    st.caption(
        f"Rate scenario: {scenario} · confidence band {user.ci_level}% · "
        f"risk tolerance {user.risk_tolerance} (λ = {lam:g}) · "
        f"re-ranked in {elapsed:.2f}s"
    )
    if decision.qualified == 0:
        st.warning(
            "No zip clears the current bars, so the whole budget is undeployed. "
            "Lower the minimum ROI, widen the acceptable downside, or select more metros."
        )


def main() -> None:
    configure_page()
    cfg = state.config()

    missing = state.missing_artifacts(cfg)
    if missing:
        render_missing(missing)
        return

    feats = state.live_features()
    metros = state.zip_metro()
    state.warm_models()

    render_header(state.universe_summary(feats, metros))

    user = sidebar.render(cfg, metros)

    started = time.perf_counter()
    decision = C.evaluate(user, feats, metros, cfg)
    elapsed = time.perf_counter() - started

    st.divider()
    render_decision_summary(decision, user, cfg, elapsed)

    st.divider()
    table.render(decision, user, cfg)

    st.divider()
    map_view.render(decision, user, cfg)

    st.divider()
    drilldown.render(decision, user, cfg, feats, metros)


main()
