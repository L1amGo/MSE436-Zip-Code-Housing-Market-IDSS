"""Sidebar parameter controls (task D1).

Widgets only. Each one reads its options and initial value from `config.yaml`
via `app.controls.defaults()`, and every value it collects is handed to
`controls.evaluate()` — no widget merely filters an already-rendered table.
"""

from __future__ import annotations

import streamlit as st

from app import controls as C

# Percent-denominated sliders: the model works in fractions, the manager thinks
# in percent, so convert at the widget boundary and nowhere else.
PCT = 100.0
_RESET_KEY = "reset_counter"


def _reset_button() -> None:
    """Bump a counter that is part of every widget key, so all widgets remount."""
    if st.button("Reset to defaults", use_container_width=True):
        st.session_state[_RESET_KEY] = st.session_state.get(_RESET_KEY, 0) + 1
        st.rerun()


def _key(name: str) -> str:
    return f"{name}__{st.session_state.get(_RESET_KEY, 0)}"


def _exclusions(defaults: C.Controls) -> tuple[str, ...]:
    """File uploader for the watch-list, with a legible error on malformed input."""
    upload = st.file_uploader(
        "Exclusions (CSV of zips)",
        type=["csv"],
        key=_key("exclusions"),
        help="Zips you will not buy into. Excluded zips are dropped from the "
        "candidate set, and their budget share is redistributed across the rest.",
    )
    if upload is None:
        return defaults.exclude_zips
    try:
        zips = C.parse_exclusions(upload)
    except C.ExclusionFormatError as exc:
        st.error(str(exc))
        return ()
    st.success(f"Excluding {len(zips):,} zip{'s' if len(zips) != 1 else ''}.")
    return tuple(zips)


def render(config: dict, metro_lookup) -> C.Controls:
    """Draw every control and return the resulting state."""
    dash = C.dashboard_config(config)
    d = C.defaults(config)

    with st.sidebar:
        st.header("Decision parameters")
        _reset_button()

        st.subheader("Universe")
        options = C.metro_options(metro_lookup)
        if options:
            metros = st.multiselect(
                "Target metros",
                options=options,
                default=list(d.metros),
                key=_key("metros"),
                help="Restricts which zips are scored and ranked. Empty = every metro.",
            )
        else:
            metros = []
            st.caption("Metro lookup unavailable — scoring the full universe.")

        st.subheader("Eligibility")
        min_roi = st.slider(
            "Min expected ROI (%)",
            min_value=-5.0,
            max_value=10.0,
            value=d.min_roi * PCT,
            step=0.25,
            key=_key("min_roi"),
            help="Drop zips whose predicted 3-month change (p50) falls below this.",
        ) / PCT
        max_downside = st.slider(
            "Max acceptable downside (%)",
            min_value=0.0,
            max_value=25.0,
            value=d.max_downside * PCT,
            step=0.5,
            key=_key("max_downside"),
            help="Drop zips whose lower confidence bound falls below this loss.",
        ) / PCT
        ci_level = st.radio(
            "Confidence level",
            options=config["model"]["ci_levels"],
            index=config["model"]["ci_levels"].index(d.ci_level),
            format_func=lambda x: f"{x}%",
            horizontal=True,
            key=_key("ci_level"),
            help="80% uses the p10–p90 band; 90% uses the wider p05–p95 band. "
            "A wider band means a lower downside bound, so 90% is the stricter test.",
        )

        st.subheader("Risk and scenario")
        tolerance = st.select_slider(
            "Risk tolerance",
            options=list(dash["risk_tolerance_lambda"].keys()),
            value=d.risk_tolerance,
            key=_key("risk_tolerance"),
            help="How hard to penalise uncertainty when ranking. Low tolerance "
            "penalises wide confidence bands most.",
        )
        st.caption(
            f"λ = {dash['risk_tolerance_lambda'][tolerance]:g} "
            "in score = p50 − λ · (p90 − p10)"
        )

        rate_bps = st.radio(
            "Mortgage rate scenario",
            options=dash["rate_scenario_bps"],
            index=dash["rate_scenario_bps"].index(d.rate_bps),
            format_func=lambda b: "No change" if b == 0 else f"{b:+d} bps",
            horizontal=True,
            key=_key("rate_bps"),
            help="Re-scores every zip with the 30-year mortgage rate shifted by "
            "this much. This changes the predictions themselves, not the display.",
        )

        st.subheader("Capital")
        budget = st.number_input(
            "Quarterly budget ($)",
            min_value=float(dash["budget_min"]),
            value=d.budget,
            step=float(dash["budget_step"]),
            format="%.0f",
            key=_key("budget"),
            help="Split across qualifying zips in proportion to risk-adjusted score.",
        )
        exclude_zips = _exclusions(d)

    return C.Controls(
        metros=tuple(metros),
        min_roi=min_roi,
        max_downside=max_downside,
        ci_level=int(ci_level),
        risk_tolerance=tolerance,
        rate_bps=int(rate_bps),
        budget=float(budget),
        exclude_zips=exclude_zips,
    )
