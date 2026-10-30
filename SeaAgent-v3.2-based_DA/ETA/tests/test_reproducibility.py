import json

from eta.evaluation.reproducibility import compare_run_metrics


def test_metric_comparison_excludes_latency(tmp_path):
    left = tmp_path / "left" / "evaluation"
    right = tmp_path / "right" / "evaluation"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    payload = {
        "metrics": {"ah": {"accuracy": 0.5}, "latency": {"p50_ms": 1.0}},
        "latency": {"p50_ms": 100.0},
    }
    changed = {
        "metrics": {"ah": {"accuracy": 0.51}, "latency": {"p50_ms": 2.0}},
        "latency": {"p50_ms": 200.0},
    }
    (left / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (right / "metrics.json").write_text(json.dumps(changed), encoding="utf-8")
    result = compare_run_metrics(left.parent, right.parent, tolerance=0.05)
    assert result["verdict"] == "REPRODUCIBLE"
    assert result["timing_compared"] is False
    assert result["maximum_relative_difference"] > 0
    assert all("latency" not in item["path"] for item in result["differences"])
