"""Stage: download — full raw pulls into data/raw/ with caching and a manifest.

Files already on disk are skipped unless --force. The Redfin archive is
streamed straight to disk (never loaded into memory). data/raw/manifest.json
records URL, timestamp, size, and sha256 per file; the FRED API key is never
written to the manifest.
"""

import datetime
import hashlib
import json
from pathlib import Path

import requests

from pipeline.io_utils import REPO_ROOT, get_fred_key, get_logger

log = get_logger("download")

CHUNK = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _stream_to_disk(url: str, dest: Path, params: dict | None = None) -> None:
    """Download url to dest via a temp file so partial downloads never look cached."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with requests.get(url, params=params, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                f.write(chunk)
                done += len(chunk)
                if done % (200 * CHUNK) < CHUNK:
                    log.info("  ... %d MB", done >> 20)
    tmp.replace(dest)
    log.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def _ensure(url: str, dest: Path, force: bool, params: dict | None = None) -> None:
    if dest.exists() and not force:
        log.info("cached: %s (%d bytes) — skipping download", dest.name, dest.stat().st_size)
        return
    log.info("downloading %s -> %s", url, dest.relative_to(REPO_ROOT))
    _stream_to_disk(url, dest, params=params)


def run(config: dict, force: bool = False) -> None:
    raw = REPO_ROOT / config["paths"]["raw"]
    fred_dir = raw / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)

    redfin_url = config["sources"]["redfin"]["url"]
    zillow_url = config["sources"]["zillow"]["url"]
    fred_base = config["sources"]["fred"]["base_url"]
    series = config["sources"]["fred"]["series"]

    targets: dict[Path, str] = {
        raw / Path(redfin_url).name: redfin_url,
        raw / "zillow_zhvi_zip.csv": zillow_url,
    }
    _ensure(redfin_url, raw / Path(redfin_url).name, force)
    _ensure(zillow_url, raw / "zillow_zhvi_zip.csv", force)

    key = get_fred_key()
    for sid in series:
        obs_url = f"{fred_base}/series/observations"
        dest = fred_dir / f"{sid}.json"
        targets[dest] = f"{obs_url}?series_id={sid}"  # manifest URL without the API key
        _ensure(
            obs_url,
            dest,
            force,
            params={"series_id": sid, "api_key": key, "file_type": "json", "limit": 100000},
        )

    manifest = {}
    for dest, url in targets.items():
        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError(f"Expected raw file missing or empty after download: {dest}")
        manifest[str(dest.relative_to(raw)).replace("\\", "/")] = {
            "url": url,
            "timestamp": datetime.datetime.fromtimestamp(
                dest.stat().st_mtime, tz=datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            "size_bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
        }
    manifest_path = raw / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s (%d files)", manifest_path.relative_to(REPO_ROOT), len(manifest))
