"""Zip-level choropleth of predicted appreciation (task D3).

Two traces, deliberately:

  * qualifying zips, coloured on a diverging scale centred at zero;
  * non-qualifying zips in flat grey.

The rejected zips stay on the map because the manager needs to see what the
filters threw away — a map of only the survivors makes the eligibility bars
invisible. The colour domain is fixed in config.yaml rather than fitted to the
data, so "how blue is this zip" means the same thing before and after a scenario
change; without that, every scenario would silently rescale its own colours and
the before/after comparison would be meaningless.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import geo
from app.components import theme

PCT = 100.0


def color_domain(config: dict) -> tuple[float, float]:
    lo, hi = config["dashboard"]["map_color_domain"]
    return float(lo), float(hi)


def diverging_scale(pal: dict) -> list:
    """Cool for gains, warm for losses, neutral grey at exactly zero."""
    return [
        [0.0, pal["negative"]],
        [0.5, pal["neutral"]],
        [1.0, pal["positive"]],
    ]


def map_frame(decision, metros: tuple[str, ...] = ()) -> pd.DataFrame:
    """Every evaluated zip with the fields the map and its hover need.

    Built from `decision.ranked` (all evaluated zips), with allocation joined
    from the qualifying set, so non-qualifiers survive with a null allocation
    and can be drawn greyed rather than omitted.
    """
    if decision.ranked is None or decision.ranked.empty:
        return pd.DataFrame(
            columns=["zip", "metro", "rank", "p50", "lo", "hi", "allocation", "qualifies"]
        )

    ranked = decision.ranked
    qualifying_zips = set(decision.qualifying["zip"]) if len(decision.qualifying) else set()

    out = ranked[["zip", "metro", "rank", "p50", "p05", "p10", "p90", "p95"]].copy()
    out["qualifies"] = out["zip"].isin(qualifying_zips)

    if len(decision.qualifying) and "allocation" in decision.qualifying:
        alloc = decision.qualifying[["zip", "allocation"]]
        out = out.merge(alloc, on="zip", how="left")
    else:
        out["allocation"] = pd.NA
    if metros:
        out = out[out["metro"].isin(metros)]
    return out.reset_index(drop=True)


def _hover(ci_level: int) -> str:
    return (
        "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
        "Predicted 3-mo change: %{z:.2f}%<br>"
        f"{ci_level}%% band: " + "%{customdata[2]:.2f}% to %{customdata[3]:.2f}%<br>"
        "Rank: %{customdata[4]}<br>"
        "Allocated: %{customdata[5]}"
        "<extra></extra>"
    )


def _customdata(frame: pd.DataFrame, ci_level: int):
    lo_col, hi_col = ("p10", "p90") if ci_level == 80 else ("p05", "p95")
    alloc = frame["allocation"].apply(
        lambda v: "—" if pd.isna(v) else f"${float(v):,.0f}"
    )
    return pd.DataFrame(
        {
            "zip": frame["zip"],
            "metro": frame["metro"],
            "lo": frame[lo_col] * PCT,
            "hi": frame[hi_col] * PCT,
            "rank": frame["rank"],
            "alloc": alloc,
        }
    ).to_numpy()


def choropleth(frame: pd.DataFrame, geojson: dict, config: dict, pal: dict, ci_level: int):
    """Build the two-trace choropleth. Pure: no Streamlit, so it is testable."""
    lo, hi = color_domain(config)
    fig = go.Figure()

    rejected = frame[~frame["qualifies"]]
    if len(rejected):
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=rejected["zip"],
                z=[0] * len(rejected),
                customdata=_customdata(rejected, ci_level),
                colorscale=[[0, pal["excluded"]], [1, pal["excluded"]]],
                showscale=False,
                marker=dict(line=dict(width=0)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                    "Does not qualify under the current filters"
                    "<extra></extra>"
                ),
                name="Not qualifying",
            )
        )

    kept = frame[frame["qualifies"]]
    if len(kept):
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=kept["zip"],
                z=kept["p50"] * PCT,
                zmin=lo * PCT,
                zmax=hi * PCT,
                colorscale=diverging_scale(pal),
                customdata=_customdata(kept, ci_level),
                marker=dict(line=dict(width=0)),
                hovertemplate=_hover(ci_level),
                colorbar=dict(
                    title=dict(
                        text="Predicted 3-mo<br>change (%)",
                        font=dict(color=pal["text_secondary"], size=12),
                    ),
                    tickfont=dict(color=pal["muted"], size=11),
                    ticksuffix="%",
                    thickness=14,
                    len=0.75,
                    outlinewidth=0,
                ),
                name="Qualifying",
            )
        )

    # No `scope`: setting it pins the projection to the whole country and
    # overrides fitbounds, which leaves a single metro as a speck in an empty
    # US-sized canvas. fitbounds="locations" zooms to whatever is drawn, so the
    # map is legible for one metro and for a nationwide selection alike.
    # Explicit axis ranges rather than fitbounds: with a custom GeoJSON,
    # fitbounds leaves a metro-sized selection as a speck in a continent-sized
    # canvas. Framing from the drawn geometry's own bounding box zooms correctly
    # for one metro and for a nationwide selection alike.
    fig.update_geos(
        projection=dict(type="mercator"),
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        lakecolor="rgba(0,0,0,0)",
    )
    box = geo.bounds(geojson)
    if box:
        lon_range, lat_range = geo.padded_ranges(box)
        fig.update_geos(lonaxis=dict(range=lon_range), lataxis=dict(range=lat_range))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=pal["text_secondary"],
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=pal["surface"],
            bordercolor=pal["axis"],
            font=dict(color=pal["text_primary"], size=12),
        ),
    )
    return fig


@st.cache_data(show_spinner="Loading zip boundaries…")
def _cached_geojson(_config: dict) -> dict:
    return geo.zip_geojson(_config)


def render(decision, controls, config: dict) -> None:
    """Draw the map, or explain precisely why it isn't drawn."""
    st.subheader("Predicted appreciation by zip")

    frame = map_frame(decision, controls.metros)
    if frame.empty:
        st.info("Nothing to map under the current parameters.")
        return

    cap = int(config["dashboard"]["map_max_zips"])
    if len(frame) > cap:
        st.info(
            f"{len(frame):,} zips are in scope — more than the {cap:,} the map draws at "
            f"once. Select one or more target metros in the sidebar to map them. "
            f"The table and allocation above still cover the full universe."
        )
        return

    if not geo.cache_path(config).exists():
        st.warning(
            "Zip boundary geometry hasn't been built yet, so the map can't be drawn. "
            "Run this once (it downloads ~66 MB from the US Census):"
        )
        st.code("python -m app.geo", language="bash")
        return

    pal = theme.palette()
    geojson = _cached_geojson(config)

    coverage = geo.join_coverage(frame["zip"], geo.geometry_ids(geojson))
    drawn = frame[~frame["zip"].isin(set(coverage.unmatched_zips))]
    # Ship only the boundaries on screen — the full collection is ~55 MB.
    drawn_geojson = geo.subset_geojson(geojson, drawn["zip"])

    st.plotly_chart(
        choropleth(drawn, drawn_geojson, config, pal, controls.ci_level),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    lo, hi = color_domain(config)
    st.caption(
        f"Colour scale is fixed at {lo * PCT:+.0f}% to {hi * PCT:+.0f}% predicted 3-month "
        f"change and centred at 0%, so colours are comparable across scenarios; values "
        f"beyond the ends clamp. Grey zips were evaluated but do not qualify under the "
        f"current filters."
    )
    # Coverage honesty (C3): say how many zips had no boundary rather than
    # letting them vanish from a map that looks complete.
    if coverage.unmatched:
        st.caption(
            f"{coverage.matched:,} of {coverage.requested:,} zips matched a Census ZCTA "
            f"boundary ({coverage.match_rate:.1%}); {coverage.unmatched:,} had no match "
            f"and are absent from the map but present in the table and allocation."
        )
