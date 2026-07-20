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
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="evaluate: score the M1 reference baselines only (skip model comparison/holdout)",
    )
    parser.add_argument(
        "--intervals",
        action="store_true",
        help="evaluate: compute confidence-band calibration (M3) and append it to the report",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="evaluate: one-shot 6-month holdout evaluation with figures (M4)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    try:
        if args.stage == "all":
            run_all(config)
        elif args.stage == "evaluate":
            evaluate.run(
                config,
                baselines_only=args.baselines_only,
                intervals=args.intervals,
                holdout=args.holdout,
            )
        else:
            STAGES[args.stage](config)
    except NotImplementedError as exc:
        print(f"not implemented: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
