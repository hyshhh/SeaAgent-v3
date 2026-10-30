# Generated research reports

The files in this directory are produced by:

```bash
PYTHONPATH=src python -m eta.cli research --config configs/default.yaml
```

`latest.json` is the machine-readable aggregate, `latest.md` is the human
summary, `research_results.csv` is the table used for downstream analysis,
`paired_analysis.json` contains track-clustered paired comparisons,
`training_histories.csv` preserves epoch-level training traces, and the PNG
files visualize structural and capacity comparisons. `reproducibility.json`
compares two independent runs of the full model. `dataset_summary.json` and
`stride5_summary.json` preserve the dataset audits. `checksums.sha256` records
the published report artifacts.

Results are proxy-label and source-split specific. They are not the paper's
reported performance and are not an estimate of human annotation accuracy.
