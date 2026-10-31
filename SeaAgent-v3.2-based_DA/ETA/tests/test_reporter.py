import csv
import json

import pytest

from eta.evaluation.reporter import publish_run, render_research_report


def test_research_report_is_minimal_and_uses_required_columns(tmp_path):
    report = {
        "best": {"id": "T3", "name": "Unweighted-Both"},
        "results": [
            {
                "id": "A5",
                "name": "CA-SH",
                "group": "消融组",
                "metrics": {
                    "ah": {"accuracy": 0.9},
                    "ch": None,
                },
                "latency": {"p50_ms": 1.25},
                "run_dir": str(tmp_path / "missing"),
            }
        ],
        "data_summary": {"samples": 10},
        "ablation_notes": ["A5 仅保留 AH 单头。"],
    }
    stale = tmp_path / "latest.md"
    stale.write_text("旧内容", encoding="utf-8")
    legacy_plot = tmp_path / "ablation_accuracy.png"
    legacy_plot.write_bytes(b"legacy")

    render_research_report(report, tmp_path / "experiments.md", tmp_path)

    markdown = (tmp_path / "experiments.md").read_text(encoding="utf-8")
    assert "| 实验设置 | 消融组/调参组 | 双头精度 | 推理速度（单次ms） |" in markdown
    assert "CH —" in markdown
    assert "A5 仅保留 AH 单头。" in markdown
    with (tmp_path / "experiments.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        assert next(csv.reader(stream)) == [
            "实验设置",
            "消融组/调参组",
            "双头精度",
            "推理速度（单次ms）",
        ]
    assert json.loads((tmp_path / "dataset_summary.json").read_text()) == {
        "samples": 10
    }
    assert not stale.exists()
    assert not legacy_plot.exists()
    assert not (tmp_path / "training_histories.csv").exists()


def test_publish_run_copies_checkpoint_and_records_validation(tmp_path):
    run_dir = tmp_path / "run"
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "history.csv").write_text(
        "epoch,validation_ah_accuracy,validation_ch_accuracy\n"
        "1,0.8,0.9\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "final",
                "variant": "eta",
                "best_epoch": 1,
                "data_summary": {},
                "environment": {},
            }
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "ah": {"accuracy": 0.8},
                    "ch": {"accuracy": 0.9},
                },
                "latency": {"p50_ms": 1.0},
            }
        ),
        encoding="utf-8",
    )

    report = publish_run(run_dir, tmp_path / "reports", tmp_path / "artifacts")

    assert (tmp_path / "artifacts" / "best_model" / "best.pt").read_bytes() == b"checkpoint"
    assert report["validation_accuracy"]["mean"] == pytest.approx(0.85)
    assert len(report["checkpoint"]["sha256"]) == 64
    assert (tmp_path / "reports" / "best_run.md").read_text(encoding="utf-8")
