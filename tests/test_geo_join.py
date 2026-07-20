"""Zip geometry join and choropleth construction (task D3).

The join is where zips silently disappear from a map that still looks complete,
so coverage is counted explicitly and asserted here. Geometry-dependent tests
skip when the cache hasn't been built; everything else runs offline on synthetic
features.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import controls as C
from app import geo
from app.components import map as M
from app.components import theme
from pipeline.io_utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def decision():
    """Six evaluated zips, four of which qualify and carry an allocation."""
    ranked = pd.DataFrame(
        {
            "rank": [1, 2, 3, 4, 5, 6],
            "zip": ["00001", "00002", "00003", "00004", "00005", "00006"],
            "metro": ["Alpha, XX"] * 3 + ["Beta, YY"] * 3,
            "p50": [0.05, 0.03, 0.01, -0.01, -0.03, 0.09],
            "p10": [0.03, 0.01, -0.01, -0.03, -0.05, 0.02],
            "p90": [0.07, 0.05, 0.03, 0.01, -0.01, 0.16],
            "p05": [0.02, 0.00, -0.02, -0.04, -0.06, 0.00],
            "p95": [0.08, 0.06, 0.04, 0.02, 0.00, 0.18],
            "score": [0.04, 0.02, 0.00, -0.02, -0.04, 0.05],
        }
    )
    qualifying = ranked.head(4).copy()
    qualifying["weight"] = [0.4, 0.3, 0.2, 0.1]
    qualifying["allocation"] = [400_000.0, 300_000.0, 200_000.0, 100_000.0]
    return C.Decision(ranked, qualifying, len(ranked), len(qualifying), 1e6, 1e6)


@pytest.fixture
def geojson():
    """Synthetic boundaries: zip 00006 deliberately has none."""
    def square(i):
        return {
            "type": "Feature",
            "id": f"{i:05d}",
            "properties": {"zip": f"{i:05d}"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[i, 0], [i + 1, 0], [i + 1, 1], [i, 1], [i, 0]]],
            },
        }

    return {"type": "FeatureCollection", "features": [square(i) for i in range(1, 6)]}


# --- coverage is counted, not assumed --------------------------------------


def test_join_coverage_counts_matches_and_misses(geojson):
    cov = geo.join_coverage(
        ["00001", "00002", "00006"], geo.geometry_ids(geojson)
    )
    assert cov.requested == 3
    assert cov.matched == 2
    assert cov.unmatched == 1
    assert cov.unmatched_zips == ("00006",)


def test_join_coverage_reports_a_match_rate(geojson):
    cov = geo.join_coverage(["00001", "00006"], geo.geometry_ids(geojson))
    assert cov.match_rate == pytest.approx(0.5)


def test_join_coverage_is_zero_padding_insensitive(geojson):
    """A zip arriving as 1 or '1' must still match boundary '00001'."""
    cov = geo.join_coverage([1, "1", "00001"], geo.geometry_ids(geojson))
    assert cov.matched == 3
    assert cov.unmatched == 0


def test_join_coverage_on_empty_request():
    cov = geo.join_coverage([], ["00001"])
    assert cov.requested == 0
    assert cov.match_rate == 0.0


def test_unmatched_zips_are_deduplicated(geojson):
    cov = geo.join_coverage(["00006", "00006"], geo.geometry_ids(geojson))
    assert cov.unmatched_zips == ("00006",)


# --- subsetting keeps the payload small ------------------------------------


def test_subset_geojson_keeps_only_requested_zips(geojson):
    sub = geo.subset_geojson(geojson, ["00001", "00003"])
    assert geo.geometry_ids(sub) == ["00001", "00003"]


def test_subset_geojson_ignores_zips_without_geometry(geojson):
    sub = geo.subset_geojson(geojson, ["00001", "00099"])
    assert geo.geometry_ids(sub) == ["00001"]


def test_subset_geojson_is_a_valid_feature_collection(geojson):
    sub = geo.subset_geojson(geojson, ["00002"])
    assert sub["type"] == "FeatureCollection"
    assert sub["features"][0]["geometry"]["type"] == "Polygon"


# --- the map frame keeps rejected zips -------------------------------------


def test_map_frame_keeps_non_qualifying_zips(decision):
    """Rejected zips stay on the map, greyed — the manager must see what was
    filtered out, not a map of survivors only."""
    frame = M.map_frame(decision)
    assert len(frame) == 6
    assert frame["qualifies"].sum() == 4
    assert (~frame["qualifies"]).sum() == 2


def test_map_frame_carries_allocation_only_for_qualifiers(decision):
    frame = M.map_frame(decision)
    assert frame.loc[frame["qualifies"], "allocation"].notna().all()
    assert frame.loc[~frame["qualifies"], "allocation"].isna().all()


def test_map_frame_restricts_to_selected_metros(decision):
    frame = M.map_frame(decision, ("Alpha, XX",))
    assert set(frame["metro"]) == {"Alpha, XX"}
    assert len(frame) == 3


def test_map_frame_on_empty_decision():
    empty = C.Decision(pd.DataFrame(), pd.DataFrame(), 0, 0, 0.0, 0.0)
    assert M.map_frame(empty).empty


# --- colour scale is fixed and centred at zero -----------------------------


def test_color_domain_is_symmetric_about_zero(config):
    lo, hi = M.color_domain(config)
    assert lo < 0 < hi
    assert lo == pytest.approx(-hi), "a diverging scale must be symmetric to centre at 0"


def test_diverging_scale_has_a_neutral_midpoint():
    scale = M.diverging_scale(theme.LIGHT)
    positions = [p for p, _ in scale]
    assert positions == [0.0, 0.5, 1.0]
    assert scale[1][1] == theme.LIGHT["neutral"], "midpoint must be neutral, not a hue"
    assert scale[0][1] != scale[2][1], "poles must differ"


def test_color_domain_is_fixed_across_scenarios(decision, geojson, config):
    """The same colour must mean the same predicted change in every scenario, so
    zmin/zmax come from config and never from the data."""
    frame = M.map_frame(decision)
    fig_a = M.choropleth(frame, geojson, config, theme.LIGHT, 80)

    shifted = decision.ranked.copy()
    shifted["p50"] = shifted["p50"] - 0.02  # a rate shock moves every prediction
    d2 = C.Decision(shifted, decision.qualifying, 6, 4, 1e6, 1e6)
    fig_b = M.choropleth(M.map_frame(d2), geojson, config, theme.LIGHT, 80)

    a = next(t for t in fig_a.data if t.name == "Qualifying")
    b = next(t for t in fig_b.data if t.name == "Qualifying")
    assert (a.zmin, a.zmax) == (b.zmin, b.zmax), "colour domain drifted with the data"


def test_colorbar_is_labelled_with_units(decision, geojson, config):
    fig = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 80)
    kept = next(t for t in fig.data if t.name == "Qualifying")
    assert "%" in kept.colorbar.title.text
    assert kept.colorbar.ticksuffix == "%"


# --- the two traces -------------------------------------------------------


def test_choropleth_draws_rejected_zips_in_flat_grey(decision, geojson, config):
    fig = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 80)
    rejected = next(t for t in fig.data if t.name == "Not qualifying")
    colors = {c for _, c in rejected.colorscale}
    assert colors == {theme.LIGHT["excluded"]}, "rejected zips must be one flat grey"
    assert rejected.showscale is False


def test_hover_reports_the_selected_band(decision, geojson, config):
    at80 = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 80)
    at90 = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 90)
    t80 = next(t for t in at80.data if t.name == "Qualifying")
    t90 = next(t for t in at90.data if t.name == "Qualifying")

    assert "80%" in t80.hovertemplate
    assert "90%" in t90.hovertemplate
    # customdata columns 2 and 3 are the band bounds, which must differ by level
    assert t80.customdata[0][2] != t90.customdata[0][2]


def test_hover_includes_zip_rank_and_allocation(decision, geojson, config):
    fig = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 80)
    kept = next(t for t in fig.data if t.name == "Qualifying")
    assert "Rank" in kept.hovertemplate
    assert "Allocated" in kept.hovertemplate
    assert kept.customdata[0][5].startswith("$")


def test_map_recolors_when_the_scenario_changes(decision, geojson, config):
    """The z values must follow the predictions, or the map is decoration."""
    base = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 80)
    shifted = decision.ranked.copy()
    shifted["p50"] = shifted["p50"] - 0.02
    d2 = C.Decision(shifted, decision.qualifying, 6, 4, 1e6, 1e6)
    after = M.choropleth(M.map_frame(d2), geojson, config, theme.LIGHT, 80)

    z_before = list(next(t for t in base.data if t.name == "Qualifying").z)
    z_after = list(next(t for t in after.data if t.name == "Qualifying").z)
    assert z_before != z_after


# --- against the real cached geometry --------------------------------------


@pytest.mark.skipif(
    not geo.cache_path(load_config()).exists(), reason="geometry cache not built"
)
def test_real_geometry_covers_almost_every_zip(config):
    from model.scenario import live_features

    gj = geo.zip_geojson(config)
    feats = live_features(config)
    cov = geo.join_coverage(feats["zip"], geo.geometry_ids(gj))
    assert cov.match_rate > 0.99, f"only {cov.match_rate:.2%} of zips found a boundary"


@pytest.mark.skipif(
    not geo.cache_path(load_config()).exists(), reason="geometry cache not built"
)
def test_subsetting_shrinks_the_payload_by_orders_of_magnitude(config):
    """Guards the fix that made the map interactive."""
    gj = geo.zip_geojson(config)
    sub = geo.subset_geojson(gj, [f["id"] for f in gj["features"][:400]])
    assert len(sub["features"]) == 400
    assert len(gj["features"]) > 30_000


# --- framing ---------------------------------------------------------------


def test_bounds_covers_every_polygon(geojson):
    box = geo.bounds(geojson)
    assert box == (1.0, 0.0, 6.0, 1.0)


def test_bounds_handles_multipolygon():
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "00001",
                "type": "Feature",
                "properties": {"zip": "00001"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        [[[5, 5], [6, 5], [6, 6], [5, 5]]],
                    ],
                },
            }
        ],
    }
    assert geo.bounds(gj) == (0.0, 0.0, 6.0, 6.0)


def test_bounds_of_empty_collection_is_none():
    assert geo.bounds({"type": "FeatureCollection", "features": []}) is None


def test_padded_ranges_add_a_margin_on_both_axes():
    lon, lat = geo.padded_ranges((-74.0, 40.0, -73.0, 41.0), pad=0.1)
    assert lon[0] < -74.0 and lon[1] > -73.0
    assert lat[0] < 40.0 and lat[1] > 41.0


def test_padded_ranges_keep_a_floor_for_a_single_zip():
    """A one-zip selection has near-zero extent; without a floor the map would
    zoom to an unusable sliver."""
    lon, lat = geo.padded_ranges((-74.0, 40.0, -74.0, 40.0))
    assert lon[1] - lon[0] == pytest.approx(0.1)
    assert lat[1] - lat[0] == pytest.approx(0.1)


def test_map_is_framed_to_the_drawn_geometry(decision, geojson, config):
    """Regression: plotly's fitbounds left a metro as a speck in a US-sized
    canvas, so the map sets explicit ranges from the geometry instead."""
    fig = M.choropleth(M.map_frame(decision), geojson, config, theme.LIGHT, 80)
    lon = fig.layout.geo.lonaxis.range
    lat = fig.layout.geo.lataxis.range
    assert lon is not None and lat is not None, "map must set explicit axis ranges"
    assert lon[0] < 1.0 and lon[1] > 6.0, "ranges must contain the drawn polygons"
