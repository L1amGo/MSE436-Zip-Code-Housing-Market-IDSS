"""Chart palette and shared Plotly styling.

One place holding the colours so every figure in the dashboard reads as one
system, and so light/dark are *selected* rather than an automatic flip. The
values are the validated reference palette: blue as the sequential/single-series
hue, blue-to-red as the diverging pair for the map, gray at the midpoint.

Charts here are single-series or diverging-continuous, so no categorical slots
are in play and no legend is needed — the title names the measure.
"""

from __future__ import annotations

import streamlit as st

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

DARK = {
    "surface": "#1a1a19",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": "#3987e5",
    "positive": "#3987e5",
    "negative": "#d03b3b",
    "neutral": "#383835",
    "excluded": "#52514e",
}


def is_dark() -> bool:
    """Whether Streamlit is currently rendering on a dark surface."""
    try:  # Streamlit >= 1.46 exposes the resolved theme
        theme = st.context.theme
        if theme and getattr(theme, "type", None):
            return theme.type == "dark"
    except Exception:  # pragma: no cover - older Streamlit, or no script context
        pass
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:  # pragma: no cover
        return False


def palette() -> dict:
    return DARK if is_dark() else LIGHT


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
