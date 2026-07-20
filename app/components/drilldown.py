"""Per-zip drill-down: why this prediction, and how the model did here (task D4).

Answers the question the ranked table provokes — "why is this zip near the top,
and should I believe it?" — with four panels: the contribution breakdown behind
the point forecast, the forecast with its band, how the zip's inputs compare to
its metro, and the model's track record on this zip over the holdout.

All explanation values come from `model.explain.explain_zip`; nothing is computed
here. The backtest reads only rows the pipeline tagged as the holdout split, so a
training month can never sneak into a chart that claims to show out-of-sample
performance.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import theme
from pipeline.io_utils import REPO_ROOT

PCT = 100.0
HOLDOUT_SPLIT = "test"
TOP_DRIVERS = 3
OTHER_LABEL = "All other features"

# Feature name -> how a property manager would say it. Anything unmapped falls
# back to the raw column name, so a new feature degrades to jargon, not a crash.
FEATURE_LABELS = {
    "price_mom_1m": "1-month price momentum",
    "price_mom_3m": "3-month price momentum",
    "price_mom_6m": "6-month price momentum",
    "price_mom_12m": "12-month price momentum",
    "zhvi_mom_1m": "1-month Zillow index momentum",
    "zhvi_mom_3m": "3-month Zillow index momentum",
    "zhvi_mom_6m": "6-month Zillow index momentum",
    "zhvi_mom_12m": "12-month Zillow index momentum",
    "median_sale_price": "median sale price",
    "homes_sold": "homes sold",
    "inventory": "inventory",
    "new_listings": "new listings",
    "median_dom": "median days on market",
    "avg_sale_to_list": "sale-to-list ratio",
    "sold_above_list": "share sold above list",
    "listings_to_sales": "listings-to-sales ratio",
    "inventory_chg_3m": "3-month inventory change",
    "mortgage_delta_3m": "3-month mortgage rate change",
    "unrate_delta_3m": "3-month unemployment change",
    "MORTGAGE30US": "30-year mortgage rate",
    "UNRATE": "unemployment rate",
    "CPIAUCSL": "CPI",
    "HOUST": "housing starts",
    "zhvi": "Zillow home value index",
    "month_of_year": "month of year",
}


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def driver_sentence(contributions: pd.Series, top: int = TOP_DRIVERS) -> str:
    """One plain-language line naming the top drivers and their direction.

    Reads as a sentence rather than a legend because this is the line a manager
    quotes in a meeting.
    """
    if contributions is None or not len(contributions):
        return "No feature contributions are available for this zip."

    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    picked = ranked.head(top)
    ups = [label_for(f) for f, v in picked.items() if v > 0]
    downs = [label_for(f) for f, v in picked.items() if v < 0]

    def phrase(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    parts = []
    if ups:
        parts.append(f"**{phrase(ups)}** push this forecast up")
    if downs:
        parts.append(f"**{phrase(downs)}** pull it down")
    if not parts:
        return "The top features contribute almost nothing to this forecast."
    return f"{' , while '.join(parts) if len(parts) > 1 else parts[0]}."


def explain(zip_code: str, month, config: dict) -> dict:
    """Feature contributions for one zip-month, straight from `model.explain`."""
    from model.explain import explain_zip

    return explain_zip(zip_code, month, config)


def contribution_chart(result: dict, pal: dict, top: int = 10):
    """Waterfall from the model's average prediction to this zip's prediction.

    Each bar is one feature's signed contribution; the bars plus the base sum to
    the point forecast, which is what makes the explanation auditable rather
    than illustrative.
    """
    contribs = result["contributions"].head(top)
    base = float(result["expected_value"])
    prediction = float(result["prediction"])
    explained = float(contribs.sum())
    residual = prediction - base - explained

    labels = [label_for(f) for f in contribs.index] + [OTHER_LABEL]
    values = [v * PCT for v in contribs.to_numpy()] + [residual * PCT]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            base=base * PCT,
            x=labels,
            y=values,
            measure=["relative"] * len(values),
            increasing=dict(marker=dict(color=pal["positive"])),
            decreasing=dict(marker=dict(color=pal["negative"])),
            connector=dict(line=dict(color=pal["axis"], width=1)),
            hovertemplate="%{x}<br>Contribution: %{y:+.2f} pp<extra></extra>",
        )
    )
    theme.style(
        fig,
        pal,
        x_title="Feature",
        # Kept short: a longer title is clipped by the plot area at this height.
        y_title="Contribution (pp)",
        height=440,
    )
    # Rotated tick labels and a vertical axis title need more room than the
    # shared 8px gutter.
    fig.update_layout(margin=dict(l=70, r=8, t=8, b=120))
    fig.add_hline(
        y=base * PCT,
        line=dict(color=pal["muted"], width=1, dash="dot"),
        annotation_text=f"Model average {base * PCT:+.2f}%",
        annotation_font=dict(color=pal["muted"], size=11),
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def backtest_frame(zip_code: str, config: dict) -> pd.DataFrame:
    """Predicted vs realized for one zip, holdout months only.

    Reads the feature matrix filtered to this zip, keeps only rows the pipeline
    tagged as the holdout split, and scores them with the committed models. The
    split filter is the point: a backtest that quietly included training months
    would flatter the model.
    """
    from model.predict import score

    path = REPO_ROOT / config["paths"]["processed"] / "features.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["month", "predicted", "realized"])

    rows = pd.read_parquet(path, filters=[("zip", "==", str(zip_code))])
    holdout = rows[rows["split"] == HOLDOUT_SPLIT]
    holdout = holdout[holdout["target"].notna()]
    if holdout.empty:
        return pd.DataFrame(columns=["month", "predicted", "realized"])

    preds = score(holdout, config)
    return (
        pd.DataFrame(
            {
                "month": holdout["month"].to_numpy(),
                "predicted": preds["p50"].to_numpy(),
                "realized": holdout["target"].to_numpy(),
                "lo": preds["p10"].to_numpy(),
                "hi": preds["p90"].to_numpy(),
            }
        )
        .sort_values("month")
        .reset_index(drop=True)
    )


def backtest_chart(frame: pd.DataFrame, pal: dict):
    """Two lines over the holdout months, with the 80% band behind them."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(frame["month"]) + list(frame["month"])[::-1],
            y=list(frame["hi"] * PCT) + list(frame["lo"] * PCT)[::-1],
            fill="toself",
            fillcolor="rgba(42,120,214,0.12)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=True,
            name="80% band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["predicted"] * PCT,
            mode="lines+markers",
            line=dict(color=pal["series"], width=2),
            marker=dict(size=8),
            name="Predicted",
            hovertemplate="%{x|%b %Y}<br>Predicted %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["month"],
            y=frame["realized"] * PCT,
            mode="lines+markers",
            line=dict(color=pal["text_primary"], width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            name="Realized",
            hovertemplate="%{x|%b %Y}<br>Realized %{y:.2f}%<extra></extra>",
        )
    )
    theme.style(
        fig,
        pal,
        x_title="Holdout month (label realized 3 months later)",
        y_title="3-month change in median sale price (%)",
        height=340,
    )
    # Two series carrying different meanings, so a legend is required.
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=pal["text_secondary"], size=12),
        ),
    )
    fig.add_hline(y=0, line=dict(color=pal["axis"], width=1))
    return fig


def metro_comparison(
    zip_code: str, metro: str, features: pd.DataFrame, metro_lookup: pd.DataFrame,
    contributions: pd.Series, top: int = 6,
) -> pd.DataFrame:
    """This zip's top-driver feature values beside its metro's median.

    Presentation-layer aggregation only — a median of values the pipeline
    already produced, for context beside the contribution chart.
    """
    if contributions is None or not len(contributions):
        return pd.DataFrame(columns=["feature", "this_zip", "metro_median"])

    names = list(contributions.abs().sort_values(ascending=False).head(top).index)
    names = [n for n in names if n in features.columns]
    row = features[features["zip"] == str(zip_code)]
    if row.empty or not names:
        return pd.DataFrame(columns=["feature", "this_zip", "metro_median"])

    peers = set(metro_lookup.loc[metro_lookup["metro"] == metro, "zip"]) if len(metro_lookup) else set()
    peer_rows = features[features["zip"].isin(peers)] if peers else features

    return pd.DataFrame(
        {
            "feature": [label_for(n) for n in names],
            "this_zip": [float(row[n].iloc[0]) for n in names],
            "metro_median": [float(peer_rows[n].median()) for n in names],
        }
    )


@st.cache_data(show_spinner=False)
def _cached_explanation(zip_code: str, month, _config: dict) -> dict:
    return explain(zip_code, month, _config)


@st.cache_data(show_spinner=False)
def _cached_backtest(zip_code: str, _config: dict) -> pd.DataFrame:
    return backtest_frame(zip_code, _config)


def render(decision, controls, config: dict, features: pd.DataFrame, metro_lookup) -> None:
    """The drill-down panel for one selected zip."""
    st.subheader("Why this zip?")

    ranked = decision.ranked
    if ranked is None or ranked.empty:
        st.info("Nothing to explain — no zip was evaluated under the current parameters.")
        return

    options = ranked["zip"].tolist()
    selected = st.selectbox(
        "Zip to explain",
        options=options,
        index=0,
        help="Any zip in the ranked table above, best-ranked first.",
    )

    row = ranked[ranked["zip"] == selected].iloc[0]
    lo_col, hi_col = ("p10", "p90") if controls.ci_level == 80 else ("p05", "p95")

    cols = st.columns(4)
    cols[0].metric("Rank", f"{int(row['rank']):,}")
    cols[1].metric("Predicted 3-mo change", f"{row['p50'] * PCT:+.2f}%")
    cols[2].metric(
        f"{controls.ci_level}% band",
        f"{row[lo_col] * PCT:+.2f}% to {row[hi_col] * PCT:+.2f}%",
    )
    allocated = decision.qualifying.set_index("zip")["allocation"].get(selected)
    cols[3].metric(
        "Allocated",
        f"${float(allocated):,.0f}" if allocated is not None and pd.notna(allocated) else "—",
    )

    pal = theme.palette()
    month = features["month"].max()

    try:
        result = _cached_explanation(selected, month, config)
    except RuntimeError as exc:
        st.warning(f"No explanation available for {selected}: {exc}")
        return

    st.markdown(driver_sentence(result["contributions"]))
    st.plotly_chart(
        contribution_chart(result, pal),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        "Each bar is one feature's contribution to this zip's forecast, measured "
        "against the model's average prediction across all zips. Bars and base sum "
        "to the point forecast."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**This zip vs its metro**")
        comparison = metro_comparison(
            selected, row["metro"], features, metro_lookup, result["contributions"]
        )
        if comparison.empty:
            st.caption("No comparable metro values available.")
        else:
            st.dataframe(
                comparison,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "feature": st.column_config.TextColumn("Feature"),
                    "this_zip": st.column_config.NumberColumn(selected, format="%.3f"),
                    "metro_median": st.column_config.NumberColumn(
                        "Metro median", format="%.3f"
                    ),
                },
            )

    with right:
        st.markdown("**Backtest — predicted vs realized**")
        frame = _cached_backtest(selected, config)
        if frame.empty:
            st.caption(
                "No holdout months for this zip, so there is nothing to backtest here."
            )
        else:
            st.plotly_chart(
                backtest_chart(frame, pal),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                f"Holdout months only ({frame['month'].min():%b %Y} to "
                f"{frame['month'].max():%b %Y}) — never months the model trained on."
            )
