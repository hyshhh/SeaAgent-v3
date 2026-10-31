import csv

from eta.evaluation.paired import paired_comparison


def _write_run(root, probabilities):
    evaluation = root / "evaluation"
    evaluation.mkdir(parents=True)
    with (evaluation / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "sample_id",
            "track_id",
            "admission_probability",
            "ah_label",
            "completeness_probability",
            "ch_label",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "track_id": f"t{index // 2}",
                    "admission_probability": probability,
                    "ah_label": index % 2,
                    "completeness_probability": 0.8,
                    "ch_label": 1,
                }
            )


def test_paired_comparison_uses_shared_samples(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_run(reference, [0.1, 0.9, 0.1, 0.9])
    _write_run(candidate, [0.9, 0.1, 0.1, 0.9])
    result = paired_comparison(reference, candidate, iterations=20)
    assert result["paired_samples"] == 4
    assert result["ah_accuracy"]["absolute_difference"] == -0.5
    assert result["joint_accuracy"] is not None
