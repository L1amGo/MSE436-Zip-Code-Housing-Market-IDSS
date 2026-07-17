"""Shared helpers: config loading and logging setup."""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def get_fred_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free at https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
    return key


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. Run from the repo root or restore the file."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
