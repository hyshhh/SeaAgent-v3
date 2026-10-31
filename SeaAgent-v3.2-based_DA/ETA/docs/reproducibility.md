# Reproducibility record

## Recorded training environment

- Python 3.12
- PyTorch 2.14.1 with CUDA 13.0
- torchvision 0.29.1
- NumPy 2.5.4
- FAISS 1.15.1
- Matplotlib 3.11.2
- NVIDIA RTX 3090, 24 GB

The host also ran unrelated GPU workloads during some experiments. Seeds,
deterministic cuDNN settings, split construction, and batch order are fixed, but
GPU reduction kernels can still introduce small run-to-run numerical changes.
The duplicate full-model run matched at the reported metric level. Wall-clock
latency can change substantially with host load.

## Exact procedure

Run from `SeaAgent-v3.2-based_DA/ETA` using the server's `seaagent` environment:

```bash
export PYTHONPATH=src
python -m eta.cli build-data --config configs/default.yaml
python -m eta.cli train --config configs/default.yaml --run-name eta-main
python -m eta.cli evaluate --config configs/default.yaml \
  --run-dir artifacts/eta-main-...
python -m eta.cli research --config configs/default.yaml
python -m eta.cli compare --run-a artifacts/eta-main-... \
  --run-b artifacts/eta-repro-... --tolerance 0.05
```

The build is deterministic given the source tables, FAISS index, configuration,
and seed. A run records:

- resolved configuration;
- hash of the resolved configuration;
- hash of every generated split and manifest;
- Git revision and dirty-worktree state;
- Python, PyTorch, CUDA and GPU environment;
- training class weights and parameter count;
- per-epoch loss and accuracy;
- best checkpoint and selected epoch.

## Verification levels

Unit tests cover config merging, chronological history expansion, source
isolation, five-second sampling, proxy-label decisions, metric formulas,
confusion matrices, KEP replacement, Stop gating, CLI overrides, and all model
variants including empty history.

The end-to-end smoke test covers build, training, checkpoint reload,
classification, calibration, confusion matrices, and latency measurement.
Full training is environment-sensitive for timing but uses deterministic seeds
and cuDNN deterministic mode.

The `compare` command checks all numeric accuracy and calibration metrics from
two saved evaluations using a symmetric relative-difference tolerance. Timing
is reported separately because concurrent host load makes it unsuitable for
bitwise reproduction checks.

For the completed experiment, two full-model runs were compared at a 5%
tolerance. Every non-timing numeric metric matched exactly, producing
`maximum_relative_difference=0.0` and `verdict=REPRODUCIBLE`.

## Prohibited interpretations

- A proxy-label score is not an estimate of expert AH/CH accuracy.
- A single source split does not establish population-level generalization.
- A latency difference under concurrent host load is not a stable benchmark.
- No model checkpoint in this directory is approved for operational deployment.
