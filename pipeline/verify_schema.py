"""Stage: verify-schema — confirm live source schemas, populate config, detect drift.

First run (config `schema:` empty): discovers real column names from live file
headers, writes the logical->actual mapping into config.yaml, and writes
data/raw/schema_report.md. Later runs re-fetch the live headers and raise with
an expected-vs-found diff if anything no longer matches.

Downloads headers only (a few KB per source), never full files.
"""

import datetime
import re
import zlib
from pathlib import Path

import requests
import yaml

from pipeline.io_utils import CONFIG_PATH, REPO_ROOT, get_fred_key, get_logger

log = get_logger("verify-schema")

# Logical feature names the pipeline uses, per spec. region_zip is Redfin's
# "region" field ("Zip Code: 12345"); everything else matches Redfin's own
# naming convention.
REDFIN_LOGICAL = [
    "period_begin",
    "period_end",
    "region_zip",
    "property_type",
    "median_sale_price",
    "homes_sold",
    "inventory",
    "new_listings",
    "median_dom",
    "avg_sale_to_list",
    "sold_above_list",
    "price_drops",
]
REDFIN_CANDIDATES = {"region_zip": ["region", "region_name", "zip_code"]}

ZILLOW_ZIP_CANDIDATES = ["RegionName", "RegionID"]
ZILLOW_MONTH_COL_REGEX = r"^\d{4}-\d{2}-\d{2}$"

_BYTES = {"n": 0}


def _fetch_first_lines_gz(url: str, n_lines: int = 2, timeout: int = 60) -> list[str]:
    """Stream a gzipped URL and return its first n_lines without downloading the file."""
    decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
    text = ""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=65536):
            _BYTES["n"] += len(chunk)
            text += decomp.decompress(chunk).decode("utf-8", errors="replace")
            if text.count("\n") >= n_lines:
                break
    lines = text.split("\n")[:n_lines]
    if len(lines) < n_lines:
        raise RuntimeError(f"Expected {n_lines} lines from {url}, got {len(lines)}")
    return lines


def _fetch_first_line(url: str, timeout: int = 60) -> str:
    """Stream a plain-text URL and return its first line only."""
    text = ""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=65536):
            _BYTES["n"] += len(chunk)
            text += chunk.decode("utf-8", errors="replace")
            if "\n" in text:
                break
    return text.split("\n")[0]


def _infer_dtype(value: str) -> str:
    if value == "" or value is None:
        return "null"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return "date"
    try:
        int(value)
        return "int"
    except ValueError:
        pass
    try:
        float(value)
        return "float"
    except ValueError:
        return "str"


def _map_logical(logical: list[str], header: list[str], candidates: dict) -> tuple[dict, list[str]]:
    """Map logical names onto a live header, case-insensitively (Redfin ships
    uppercase column names). Returns (mapping to actual as-published names, missing)."""
    by_upper = {c.upper(): c for c in header}
    mapping, missing = {}, []
    for name in logical:
        for cand in [name, *candidates.get(name, [])]:
            if cand.upper() in by_upper:
                mapping[name] = by_upper[cand.upper()]
                break
        else:
            missing.append(name)
    return mapping, missing


def _verify_fred(config: dict) -> dict:
    """Confirm each configured series exists; return {series: {frequency, dtype, latest}}."""
    base = config["sources"]["fred"]["base_url"]
    key = get_fred_key()
    out = {}
    for series in config["sources"]["fred"]["series"]:
        meta_resp = requests.get(
            f"{base}/series",
            params={"series_id": series, "api_key": key, "file_type": "json"},
            timeout=30,
        )
        if meta_resp.status_code != 200:
            raise RuntimeError(
                f"FRED series '{series}' lookup failed (HTTP {meta_resp.status_code}): "
                f"{meta_resp.text[:200]}"
            )
        _BYTES["n"] += len(meta_resp.content)
        meta = meta_resp.json()["seriess"][0]
        obs_resp = requests.get(
            f"{base}/series/observations",
            params={
                "series_id": series,
                "api_key": key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc",
            },
            timeout=30,
        )
        obs_resp.raise_for_status()
        _BYTES["n"] += len(obs_resp.content)
        obs = obs_resp.json()["observations"][0]
        out[series] = {
            "frequency": meta["frequency_short"],
            "title": meta["title"],
            "dtype": _infer_dtype(obs["value"]),
            "latest": obs["date"],
        }
        log.info("FRED %s: frequency=%s latest=%s", series, meta["frequency_short"], obs["date"])
    return out


def _discover(config: dict) -> tuple[dict, dict]:
    """Fetch live headers for all sources. Returns (schema, report_info)."""
    redfin_lines = _fetch_first_lines_gz(config["sources"]["redfin"]["url"], n_lines=2)
    redfin_header = [c.strip('"') for c in redfin_lines[0].rstrip("\r").split("\t")]
    redfin_row = [c.strip('"') for c in redfin_lines[1].rstrip("\r").split("\t")]
    log.info("Redfin header (%d cols): %s", len(redfin_header), redfin_header)
    redfin_map, redfin_missing = _map_logical(REDFIN_LOGICAL, redfin_header, REDFIN_CANDIDATES)

    zillow_header = [
        c.strip('"')
        for c in _fetch_first_line(config["sources"]["zillow"]["url"]).rstrip("\r").split(",")
    ]
    log.info("Zillow header (%d cols): first 12 = %s", len(zillow_header), zillow_header[:12])
    zillow_zip = next((c for c in ZILLOW_ZIP_CANDIDATES if c in zillow_header), None)
    if zillow_zip is None:
        raise RuntimeError(
            f"Zillow header has no zip column among {ZILLOW_ZIP_CANDIDATES}; "
            f"found: {zillow_header[:12]}..."
        )
    zillow_month_cols = [c for c in zillow_header if re.fullmatch(ZILLOW_MONTH_COL_REGEX, c)]
    if not zillow_month_cols:
        raise RuntimeError("Zillow header has no YYYY-MM-DD month columns — layout changed?")

    fred_info = _verify_fred(config)

    schema = {
        "redfin": {"columns": redfin_map},
        "zillow": {
            "columns": {"region_zip": zillow_zip},
            "month_col_regex": ZILLOW_MONTH_COL_REGEX,
        },
        "fred": {"frequencies": {s: info["frequency"] for s, info in fred_info.items()}},
    }
    report_info = {
        "redfin_header": redfin_header,
        "redfin_row": redfin_row,
        "redfin_missing": redfin_missing,
        "zillow_header": zillow_header,
        "zillow_zip": zillow_zip,
        "zillow_month_range": (zillow_month_cols[0], zillow_month_cols[-1]),
        "fred": fred_info,
    }
    return schema, report_info


def _diff_error(source: str, expected: list[str], found: list[str]) -> str:
    missing = sorted(set(expected) - set(found))
    return "\n".join(
        [
            f"Schema drift detected for {source}: expected column(s) missing from live header.",
            f"  expected but not found: {missing}",
            f"  live header: {found}",
            "If the source really changed, clear the `schema:` section in config.yaml "
            "and re-run verify-schema to re-discover.",
        ]
    )


def _check_drift(config: dict, schema_live: dict, report_info: dict) -> None:
    stored = config["schema"]
    redfin_expected = list(stored["redfin"]["columns"].values())
    if not set(redfin_expected) <= set(report_info["redfin_header"]):
        raise RuntimeError(_diff_error("redfin", redfin_expected, report_info["redfin_header"]))
    zillow_expected = list(stored["zillow"]["columns"].values())
    if not set(zillow_expected) <= set(report_info["zillow_header"]):
        raise RuntimeError(_diff_error("zillow", zillow_expected, report_info["zillow_header"]))
    stored_freq = stored["fred"]["frequencies"]
    live_freq = schema_live["fred"]["frequencies"]
    if stored_freq != live_freq:
        raise RuntimeError(
            "Schema drift detected for FRED series/frequencies.\n"
            f"  expected: {stored_freq}\n  found:    {live_freq}\n"
            "If the change is real, clear `schema:` in config.yaml and re-run."
        )
    log.info("No schema drift: live headers match config.yaml `schema:`.")


def _write_schema_to_config(schema: dict) -> None:
    """Replace the `schema:` section of config.yaml in place, preserving everything above it."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = re.search(r"^schema:.*$", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("config.yaml has no `schema:` line to replace")
    block = yaml.safe_dump({"schema": schema}, sort_keys=False, default_flow_style=False)
    CONFIG_PATH.write_text(text[: match.start()] + block, encoding="utf-8")
    log.info("Wrote discovered schema mapping into config.yaml")


def _write_report(report_info: dict, path: Path) -> None:
    today = datetime.date.today().isoformat()
    dtypes = dict(
        zip(report_info["redfin_header"], (_infer_dtype(v) for v in report_info["redfin_row"]))
    )
    static_cols = [
        c for c in report_info["zillow_header"] if not re.fullmatch(ZILLOW_MONTH_COL_REGEX, c)
    ]
    z_first, z_last = report_info["zillow_month_range"]
    lines = [
        "# Schema report",
        "",
        f"Verified against live sources on **{today}** (headers only, no full downloads).",
        "",
        "## Redfin zip-code market tracker (gzipped TSV)",
        "",
        f"Columns found ({len(report_info['redfin_header'])}):",
        "",
        "| column | observed dtype (from first data row) |",
        "|---|---|",
        *[f"| `{c}` | {dtypes[c]} |" for c in report_info["redfin_header"]],
        "",
    ]
    if report_info["redfin_missing"]:
        lines += [
            f"**Expected logical columns with no live match (dropped):** "
            f"{', '.join(report_info['redfin_missing'])}",
            "",
        ]
    else:
        lines += ["All 12 expected logical columns matched a live column.", ""]
    lines += [
        "## Zillow ZHVI zip-level (wide CSV)",
        "",
        f"- Columns found: {len(report_info['zillow_header'])} total.",
        f"- Static columns: {static_cols}",
        f"- Month columns: `{z_first}` … `{z_last}` (one column per month).",
        f"- Zip column: `{report_info['zillow_zip']}` (dtypes not sampled — header-only fetch).",
        "",
        "## FRED series",
        "",
        "| series | title | native frequency | latest obs | dtype |",
        "|---|---|---|---|---|",
        *[
            f"| `{s}` | {i['title']} | {i['frequency']} | {i['latest']} | {i['dtype']} |"
            for s, i in report_info["fred"].items()
        ],
        "",
        f"Total bytes downloaded during verification: {_BYTES['n']:,}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path)


def run(config: dict, force: bool = False) -> None:
    _BYTES["n"] = 0
    schema_live, report_info = _discover(config)
    if config.get("schema"):
        _check_drift(config, schema_live, report_info)
    else:
        _write_schema_to_config(schema_live)
    _write_report(report_info, REPO_ROOT / config["paths"]["reports"] / "schema_report.md")
    log.info("verify-schema complete; %s bytes downloaded (< 10 MB required)", f"{_BYTES['n']:,}")
