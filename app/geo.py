"""ZCTA boundary geometry for the choropleth (task D3).

Geometry is a *presentation* asset: it is downloaded once from the Census,
simplified for the browser, cached on disk, and never joined into the feature
matrix. Nothing here influences a prediction.

The build is deliberately separable from the app so a first map render doesn't
sit behind a 66 MB download:

    python -m app.geo        # prebuild the cache

`zip_geojson()` returns a GeoJSON FeatureCollection keyed by 5-digit zip, and
`join_coverage()` reports which zips found a boundary — the unmatched count is a
number the data slide quotes, so it is counted and surfaced, never dropped
quietly.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.io_utils import REPO_ROOT, get_logger

log = get_logger("geo")

CACHE_NAME = "zcta_simplified.geojson"


def source(config: dict) -> dict:
    src = config.get("sources", {}).get("zcta")
    if not src:
        raise RuntimeError("config.yaml is missing `sources.zcta` (the boundary file URL).")
    return src


def cache_path(config: dict) -> Path:
    return REPO_ROOT / config["paths"]["processed"] / CACHE_NAME


def raw_path(config: dict) -> Path:
    return REPO_ROOT / config["paths"]["raw"] / Path(source(config)["url"]).name


def normalize_zip(series: pd.Series) -> pd.Series:
    """ZCTA codes as zero-padded 5-character strings, matching the feature matrix."""
    return series.astype(str).str.strip().str.zfill(5)


@dataclass(frozen=True)
class Coverage:
    """How many zips found a boundary, and which didn't."""

    requested: int
    matched: int
    unmatched_zips: tuple[str, ...]

    @property
    def unmatched(self) -> int:
        return len(self.unmatched_zips)

    @property
    def match_rate(self) -> float:
        return self.matched / self.requested if self.requested else 0.0


def join_coverage(zips, geometry_ids) -> Coverage:
    """Which of `zips` have a boundary in `geometry_ids`.

    Separate from any plotting so the count can be asserted in tests and shown
    on screen without drawing anything.
    """
    wanted = [str(z).zfill(5) for z in zips]
    have = {str(g).zfill(5) for g in geometry_ids}
    missing = tuple(sorted({z for z in wanted if z not in have}))
    return Coverage(
        requested=len(wanted),
        matched=len(wanted) - sum(1 for z in wanted if z in missing),
        unmatched_zips=missing,
    )


def _download(config: dict) -> Path:
    import requests

    dest = raw_path(config)
    if dest.exists():
        return dest
    url = source(config)["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading ZCTA boundaries %s -> %s", url, dest)
    part = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    part.rename(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def build_cache(config: dict, force: bool = False) -> Path:
    """Download, simplify, and cache the boundaries as slim GeoJSON.

    Simplification is what makes the map interactive: the raw cartographic
    boundaries carry far more vertices than a screen can show, and shipping them
    all to the browser is the difference between a map that renders in a second
    and one that stalls. Only the zip code is kept as a property; everything the
    hover needs is joined in from the predictions instead.
    """
    out = cache_path(config)
    if out.exists() and not force:
        return out

    import geopandas as gpd

    archive = _download(config)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        shp = next((n for n in zf.namelist() if n.endswith(".shp")), None)
        if shp is None:
            raise RuntimeError(f"{archive.name} contains no .shp file.")
        gdf = gpd.read_file(f"zip://{archive}!{shp}")

    field = source(config)["zip_field"]
    if field not in gdf.columns:
        raise RuntimeError(
            f"Boundary file has no `{field}` column (found {list(gdf.columns)}). "
            "Update `sources.zcta.zip_field` in config.yaml."
        )

    tolerance = float(config["dashboard"]["map_simplify_tolerance"])
    gdf = gdf[[field, "geometry"]].copy()
    gdf["zip"] = normalize_zip(gdf[field])
    gdf = gdf.drop(columns=[field])
    # preserve_topology keeps polygons valid (no self-intersections) as they thin.
    gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)

    gdf[["zip", "geometry"]].to_file(out, driver="GeoJSON")
    log.info(
        "wrote %s (%d boundaries, %.1f MB, tolerance %g deg)",
        out, len(gdf), out.stat().st_size / 1e6, tolerance,
    )
    return out


def zip_geojson(config: dict) -> dict:
    """The cached FeatureCollection, with `id` set to the zip for Plotly."""
    path = cache_path(config)
    if not path.exists():
        raise RuntimeError(
            f"{path.relative_to(REPO_ROOT)} missing. Run `python -m app.geo` to build it."
        )
    with path.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    for feature in gj.get("features", []):
        feature["id"] = str(feature.get("properties", {}).get("zip", "")).zfill(5)
    return gj


def geometry_ids(geojson: dict) -> list[str]:
    return [f["id"] for f in geojson.get("features", [])]


def _iter_coords(geometry: dict):
    """Yield (lon, lat) from any GeoJSON geometry, at any nesting depth."""
    def walk(node):
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
                yield float(node[0]), float(node[1])
            else:
                for child in node:
                    yield from walk(child)

    yield from walk(geometry.get("coordinates", []))


def bounds(geojson: dict) -> tuple[float, float, float, float] | None:
    """(lon_min, lat_min, lon_max, lat_max) of a FeatureCollection, or None.

    Plotly's `fitbounds` does not reliably frame a custom GeoJSON choropleth —
    a single metro renders as a speck inside a continent-sized canvas — so the
    map sets explicit axis ranges from this instead.
    """
    lons, lats = [], []
    for feature in geojson.get("features", []):
        for lon, lat in _iter_coords(feature.get("geometry") or {}):
            lons.append(lon)
            lats.append(lat)
    if not lons:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def padded_ranges(
    box: tuple[float, float, float, float], pad: float = 0.08
) -> tuple[list[float], list[float]]:
    """Axis ranges with a margin so boundary zips aren't flush against the edge."""
    lon_min, lat_min, lon_max, lat_max = box
    lon_pad = max((lon_max - lon_min) * pad, 0.05)
    lat_pad = max((lat_max - lat_min) * pad, 0.05)
    return (
        [lon_min - lon_pad, lon_max + lon_pad],
        [lat_min - lat_pad, lat_max + lat_pad],
    )


def subset_geojson(geojson: dict, zips) -> dict:
    """Keep only the boundaries actually being drawn.

    Plotly serialises the whole FeatureCollection into the page, so handing it
    all 33,791 ZCTAs costs ~55 MB and several seconds no matter how few zips the
    map shows. Subsetting first is what puts a metro-sized map inside the 2 s
    interactivity budget.
    """
    wanted = {str(z).zfill(5) for z in zips}
    return {
        "type": "FeatureCollection",
        "features": [f for f in geojson.get("features", []) if f.get("id") in wanted],
    }


def main() -> int:
    from pipeline.io_utils import load_config

    cfg = load_config()
    path = build_cache(cfg)
    gj = zip_geojson(cfg)
    log.info("geometry cache ready: %s (%d boundaries)", path, len(gj.get("features", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
