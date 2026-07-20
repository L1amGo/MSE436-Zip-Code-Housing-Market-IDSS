"""Ranked table and budget allocation view (task D2).

The main panel: which zips qualify, in what order, and how the quarter's capital
splits across them. The frame-building functions are pure so the allocation tests
can assert on them without a browser; only `render` touches Streamlit.

Display units: predictions and score are fractions in the model and percentages
on screen, converted once here at the boundary. Score is in percentage points
because it is a p50 net of a band penalty, both of which are fractions.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import theme

PCT = 100.0
TOP_N_CHART = 20

# Lower/upper quantile column for each selectable band (mirrors model.decide).
BAND_COLS = {80: ("p10", "p90"), 90: ("p05", "p95")}

SCORE_FORMULA = "score = p50 − λ · (p90 − p10)"

# Above this single-zip share the "allocation" is really a single bet, which the
# manager should be told rather than left to infer from a one-bar chart.
CONCENTRATION_WARN_SHARE = 25.0


def concentration_note(display: pd.DataFrame) -> str | None:
    """Warn when the budget collapses onto a handful of zips.

    Allocation weight is score clipped at zero, so a lambda large enough to push
    most scores negative silently concentrates the whole budget. That is a real
    property of the decision rule, not a rendering artefact — so it is surfaced.
    """
    if display.empty:
        return None
    top = float(display["share_pct"].max())
    if top < CONCENTRATION_WARN_SHARE:
        return None
    funded = int((display["allocation"] > 0).sum())
    return (
        f"Top zip takes {top:.1f}% of the budget and only {funded:,} of "
        f"{len(display):,} qualifying zips are funded. Allocation weight is the "
        f"risk-adjusted score clipped at zero, so a high λ concentrates capital. "
        f"Raise risk tolerance to spread it."
    )


def band_columns(ci_level: int) -> tuple[str, str]:
    if ci_level not in BAND_COLS:
        raise RuntimeError(f"ci_level must be one of {sorted(BAND_COLS)}, got {ci_level}.")
    return BAND_COLS[ci_level]


def build_display_table(qualifying: pd.DataFrame, ci_level: int) -> pd.DataFrame:
    """Build the columns shown on screen from the qualifying set, in display units.

    Percentages rather than fractions, dollars rather than weights, and the
    confidence band drawn from whichever pair of quantiles the CI toggle selects.
    """
    lo_col, hi_col = band_columns(ci_level)
    if qualifying.empty:
        return pd.DataFrame(
            columns=["rank", "zip", "metro", "pred_pct", "ci_lo_pct", "ci_hi_pct",
                     "score_pp", "allocation", "share_pct"]
        )

    out = pd.DataFrame(
        {
            "rank": qualifying["rank"].to_numpy(),
            "zip": qualifying["zip"].to_numpy(),
            "metro": qualifying["metro"].to_numpy(),
            "pred_pct": qualifying["p50"].to_numpy() * PCT,
            "ci_lo_pct": qualifying[lo_col].to_numpy() * PCT,
            "ci_hi_pct": qualifying[hi_col].to_numpy() * PCT,
            "score_pp": qualifying["score"].to_numpy() * PCT,
        }
    )
    out["allocation"] = qualifying["allocation"].fillna(0.0).to_numpy()
    out["share_pct"] = qualifying["weight"].fillna(0.0).to_numpy() * PCT
    return out


def allocation_csv(display: pd.DataFrame, ci_level: int) -> bytes:
    """The current allocation as a CSV download, with units named in the headers."""
    renamed = display.rename(
        columns={
            "rank": "rank",
            "zip": "zip",
            "metro": "metro",
            "pred_pct": "predicted_3mo_change_pct",
            "ci_lo_pct": f"ci{ci_level}_low_pct",
            "ci_hi_pct": f"ci{ci_level}_high_pct",
            "score_pp": "risk_adjusted_score_pp",
            "allocation": "allocated_usd",
            "share_pct": "allocated_share_pct",
        }
    )
    return renamed.to_csv(index=False).encode("utf-8")


def _column_config(ci_level: int) -> dict:
    return {
        "rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
        "zip": st.column_config.TextColumn("Zip", width="small"),
        "metro": st.column_config.TextColumn("Metro"),
        "pred_pct": st.column_config.NumberColumn(
            "Predicted 3-mo change (%)",
            help="Median forecast (p50) of the change in median sale price three months out.",
            format="%.2f%%",
        ),
        "ci_lo_pct": st.column_config.NumberColumn(f"{ci_level}% low (%)", format="%.2f%%"),
        "ci_hi_pct": st.column_config.NumberColumn(f"{ci_level}% high (%)", format="%.2f%%"),
        "score_pp": st.column_config.NumberColumn(
            "Score (pp)",
            help=SCORE_FORMULA + ", in percentage points",
            format="%.2f",
        ),
        "allocation": st.column_config.NumberColumn("Allocated ($)", format="$%,.0f"),
        "share_pct": st.column_config.NumberColumn("Share (%)", format="%.2f%%"),
    }


def allocation_chart(display: pd.DataFrame, pal: dict, top_n: int = TOP_N_CHART):
    """Horizontal bars of the largest allocations — magnitude across named zips.

    Single series, so no legend: the axis titles carry the measure and its units.
    """
    top = display.nlargest(top_n, "allocation").sort_values("allocation")
    labels = [f"{z} · {m}" if m else z for z, m in zip(top["zip"], top["metro"])]

    fig = go.Figure(
        go.Bar(
            x=top["allocation"],
            y=labels,
            orientation="h",
            marker=dict(color=pal["series"], cornerradius=4),
            customdata=top[["pred_pct", "share_pct"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Allocated: $%{x:,.0f}<br>"
                "Share of budget: %{customdata[1]:.2f}%<br>"
                "Predicted 3-mo change: %{customdata[0]:.2f}%"
                "<extra></extra>"
            ),
        )
    )
    theme.style(
        fig,
        pal,
        x_title="Allocated capital (USD)",
        y_title="Zip · metro",
        height=max(280, 26 * len(top) + 90),
    )
    fig.update_layout(bargap=0.35)
    return fig


def render(decision, controls, config: dict) -> None:
    """Draw the ranked table, the allocation chart, and the CSV download."""
    pal = theme.palette()
    display = build_display_table(decision.qualifying, controls.ci_level)

    st.subheader("Ranked buy list")
    if display.empty:
        st.info(
            "No zip qualifies under the current parameters, so there is nothing to "
            "rank or allocate. Loosen the eligibility bars in the sidebar."
        )
        return

    note = concentration_note(display)
    if note:
        st.warning(note)

    st.dataframe(
        display,
        column_config=_column_config(controls.ci_level),
        hide_index=True,
        use_container_width=True,
        height=min(560, 38 * len(display) + 40),
    )

    # The score formula is on screen, not just in the docs, because the ordering
    # is the recommendation — the manager should be able to see what produced it.
    lam = config["dashboard"]["risk_tolerance_lambda"][controls.risk_tolerance]
    gaps = len(display) and int(display["rank"].max()) > len(display)
    st.caption(
        f"**{SCORE_FORMULA}** — with λ = {lam:g} at “{controls.risk_tolerance}” risk "
        f"tolerance. The penalty always uses the 80% band (p90 − p10) so rankings stay "
        f"comparable when the displayed band changes; the table shows the "
        f"{controls.ci_level}% band."
        + (
            " Rank is position in the full scored universe, so gaps in the sequence "
            "are higher-ranked zips that failed the eligibility bars."
            if gaps
            else ""
        )
    )

    st.download_button(
        "Download this allocation (CSV)",
        data=allocation_csv(display, controls.ci_level),
        file_name=f"allocation_{controls.ci_level}ci_{controls.rate_bps:+d}bps.csv",
        mime="text/csv",
    )

    st.subheader(f"Budget allocation — top {min(TOP_N_CHART, len(display))} of {len(display):,} zips")
    st.plotly_chart(
        allocation_chart(display, pal),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    if len(display) > TOP_N_CHART:
        st.caption(
            f"Chart shows the {TOP_N_CHART} largest allocations; all {len(display):,} "
            f"qualifying zips are in the table and the CSV."
        )
