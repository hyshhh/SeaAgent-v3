"""Command line interface for the isolated ETA research project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .config import ETA_ROOT, load_config
from .utils import deep_update


def _parse_overrides(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Override must use key=value syntax: {value}")
        key, raw = value.split("=", 1)
        parsed = yaml.safe_load(raw)
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parsed
    return result


def _config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    overrides = _parse_overrides(args.set)
    return deep_update(config, overrides)


def _latest_run(artifact_dir: str | Path) -> Path:
    candidates = [
        path
        for path in Path(artifact_dir).iterdir()
        if path.is_dir() and (path / "best.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed runs under {artifact_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(ETA_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a resolved config value, for example train.epochs=2",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-data", help="Build TEA splits from trajectory memory")
    _add_config(build)
    train = commands.add_parser("train", help="Train one ETA variant")
    _add_config(train)
    train.add_argument("--run-name")
    evaluate = commands.add_parser("evaluate", help="Evaluate a trained checkpoint")
    _add_config(evaluate)
    evaluate.add_argument("--run-dir")
    compare = commands.add_parser("compare", help="Compare metrics from two runs")
    _add_config(compare)
    compare.add_argument("--run-a", required=True)
    compare.add_argument("--run-b", required=True)
    compare.add_argument("--tolerance", type=float, default=0.05)
    compare.add_argument("--output")
    run_all = commands.add_parser("run-all", help="Build, train and evaluate one ETA variant")
    _add_config(run_all)
    run_all.add_argument("--run-name")
    research = commands.add_parser(
        "research",
        help="Run structural ablations and KEP-capacity experiments",
    )
    _add_config(research)
    research.add_argument(
        "--ablations",
        default=str(ETA_ROOT / "configs" / "experiments" / "ablations.yaml"),
    )
    research.add_argument(
        "--capacities",
        default=str(ETA_ROOT / "configs" / "experiments" / "capacity.yaml"),
    )
    research.add_argument("--skip-ablations", action="store_true")
    research.add_argument("--skip-capacity", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _config(args)
    if args.command == "build-data":
        from .dataset import build_dataset

        result = build_dataset(config)
    elif args.command == "train":
        from .training import train_model

        result = train_model(config, args.run_name)
    elif args.command == "evaluate":
        from .evaluation import evaluate_run

        run_dir = Path(args.run_dir) if args.run_dir else _latest_run(
            config["paths"]["artifact_dir"]
        )
        result = evaluate_run(config, run_dir)
    elif args.command == "compare":
        from .evaluation.reproducibility import compare_run_metrics

        result = compare_run_metrics(
            args.run_a,
            args.run_b,
            args.tolerance,
            args.output,
        )
    elif args.command == "run-all":
        from .dataset import build_dataset
        from .evaluation import evaluate_run
        from .training import train_model

        build_dataset(config)
        training = train_model(config, args.run_name)
        result = evaluate_run(config, training["run_dir"])
    elif args.command == "research":
        from .research import run_research_series

        result = run_research_series(
            config,
            args.ablations,
            args.capacities,
            include_ablations=not args.skip_ablations,
            include_capacity=not args.skip_capacity,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
