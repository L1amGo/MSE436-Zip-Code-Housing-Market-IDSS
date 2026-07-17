"""CLI entry point: python -m pipeline <stage> [--force]."""

import argparse
import sys

from pipeline import clean, download, featurize, join, report, split, verify_schema
from pipeline.io_utils import load_config

STAGES = {
    "verify-schema": verify_schema.run,
    "download": download.run,
    "clean": clean.run,
    "join": join.run,
    "featurize": featurize.run,
    "split": split.run,
    "report": report.run,
}


def run_all(config: dict, force: bool = False) -> None:
    for name, fn in STAGES.items():
        print(f"=== stage: {name} ===")
        fn(config, force=force)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Zip-code housing market IDSS data pipeline.",
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="pipeline stage to run ('all' runs every stage in order)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download / rebuild outputs even if cached files exist",
    )
    args = parser.parse_args(argv)

    config = load_config()
    runner = run_all if args.stage == "all" else STAGES[args.stage]
    try:
        runner(config, force=args.force)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
