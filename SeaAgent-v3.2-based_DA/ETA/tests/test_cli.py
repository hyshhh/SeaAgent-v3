import json

from eta.cli import _parse_overrides, build_parser


def test_cli_overrides_are_typed_and_nested():
    assert _parse_overrides(
        ["train.epochs=2", "model.variant=ca_only", "evaluation.latency_repeats=3"]
    ) == {
        "train": {"epochs": 2},
        "model": {"variant": "ca_only"},
        "evaluation": {"latency_repeats": 3},
    }


def test_parser_exposes_isolated_workflows():
    parser = build_parser()
    assert parser.parse_args(["build-data"]).command == "build-data"
    assert parser.parse_args(["run-all", "--run-name", "smoke"]).run_name == "smoke"


def test_cli_output_serializes_version_subclasses():
    class VersionString(str):
        pass

    assert json.dumps({"torch": VersionString("2.14.1")}, default=str)
