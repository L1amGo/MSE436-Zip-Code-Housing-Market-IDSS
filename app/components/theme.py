"""Chart palette and shared Plotly styling.

One place holding the colours so every figure in the dashboard reads as one
system. The values are the validated reference palette: blue as the
sequential/single-series hue, blue-to-red as the diverging pair for the map,
grey at the midpoint. Checked with the palette validator — lightness band,
chroma floor, CVD separation and contrast all pass against the light surface.

**Light only, deliberately.** The app is pinned to Streamlit's light theme in
`.streamlit/config.toml`. A dark palette was tried and dropped: Streamlit
resolves the viewer's theme in the browser, so the server-side figure code could
not reliably tell which surface it was drawing on, and the mismatch shipped a
near-black backtest line onto a near-black background. One palette that is
always correct beats two where one silently misfires — and every figure in
reports/figures/ is light, so the app matches the deck.

Charts here are single-series or diverging-continuous, so no categorical slots
are in play and no legend is needed except where two series carry different
meanings.
"""

from __future__ import annotations

LIGHT = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",  # sequential/single-series blue
    "positive": "#2a78d6",
    "negative": "#d03b3b",
    "neutral": "#f0efec",  # diverging midpoint
    "excluded": "#c3c2b7",  # de-emphasised, non-qualifying marks
}

# Band fill for the backtest: the series blue at low alpha, so the shaded
# interval reads as the same colour as the line it belongs to.
BAND_FILL = "rgba(42, 120, 214, 0.12)"


def palette() -> dict:
    """The chart palette. One surface, so one palette — see the module docstring."""
    return LIGHT


def style(fig, pal: dict, *, x_title: str, y_title: str, height: int | None = None):
    """Apply the shared chart chrome: recessive grid, labelled axes, no legend.

    Backgrounds stay transparent so the figure sits on Streamlit's own surface
    instead of painting a slightly-different rectangle on top of it.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=pal["text_secondary"],
            size=13,
        ),
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=pal["surface"],
            bordercolor=pal["axis"],
            font=dict(color=pal["text_primary"], size=12),
        ),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=pal["text_secondary"])),
        gridcolor=pal["grid"],
        linecolor=pal["axis"],
        zerolinecolor=pal["axis"],
        tickfont=dict(color=pal["muted"]),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=pal["text_secondary"])),
        gridcolor=pal["grid"],
        linecolor=pal["axis"],
        zerolinecolor=pal["axis"],
        tickfont=dict(color=pal["muted"]),
    )
    return fig
