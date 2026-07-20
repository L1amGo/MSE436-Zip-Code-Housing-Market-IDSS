"""CLI entry point: python -m model <stage>.

Stages are implemented across spec_model.md tasks M1–M8; in the M0 scaffold
they dispatch to not-yet-implemented stage functions that report which task
will implement them (mirrors the pipeline scaffold's behaviour).
"""

import argparse
import sys

from model import evaluate, explain, retrain, train
from pipeline.io_utils import load_config

STAGES = {
    "train": train.run,
    "evaluate": evaluate.run,
    "explain": explain.run,
    "retrain": retrain.run,
}


def run_all(config: dict) -> None:
    for name in ("train", "evaluate", "explain"):
        print(f"=== model stage: {name} ===")
        STAGES[name](config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model",
        description="Zip-code housing market IDSS model & decision layer.",
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="model stage to run ('all' runs train, evaluate, explain in order)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    runner = run_all if args.stage == "all" else STAGES[args.stage]
    try:
        runner(config)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
