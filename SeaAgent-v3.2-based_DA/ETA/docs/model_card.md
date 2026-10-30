# ETA/DPD model card

## Intended use

ETA is a research model that predicts whether a candidate keyframe adds identity
evidence to a historical KEP and whether the updated pool is complete enough to
stop evidence collection. It is designed for evaluation with the TEA protocol,
not for autonomous deployment on safety-critical maritime decisions.

## Architecture

- Candidate query: linear projection from the pooled visual embedding.
- History approximation: each 2048-dimensional pooled vector is split into
  eight 256-dimensional groups and projected to the interaction width.
- Serial interaction: two stages of two shared Post-LN cross-attention blocks.
- Attention: four heads; FFN width 1024.
- Decision heads: independent AH and CH two-class MLPs with hidden width 128.
- Capacity: default KEP size `M=3`.
- Stop rule: actual KEP insertion AND CH positive probability at or above the
  configured threshold.

## Outputs

- `AH`: whether the candidate merits admission or replacement.
- `CH`: whether the updated evidence pool is complete.
- `Stop`: emitted only after the online pool confirms an actual update.

## Training

The default run uses deterministic proxy labels, source-group splits, class
frequency weighting, AdamW, two warmup epochs, cosine decay, batch size 128,
30 epochs, and seed 42. The selected checkpoint maximizes validation joint
accuracy for dual-head models.

## Evaluation

Primary metrics are AH accuracy, CH accuracy, joint accuracy, static Stop-rule
agreement F1, and batch-1 latency. Calibration, confusion matrices, Brier score,
bootstrap intervals, and capacity sensitivity are supporting diagnostics.
Sequence-level online replay is not included in the current test metric; the
actual insertion gate is isolated in `EvidencePool` and unit tested.

## Limitations

- Proxy labels are not a substitute for human TEA annotation.
- The pooled-embedding token approximation differs from the paper's raw token
  encoder output.
- The available dataset is small and source coverage is uneven.
- The model may exploit quality, readability, or source-specific shortcuts.
- Latency excludes visual encoding and depends on hardware and software versions.
- Predictions should be treated as evidence-selection guidance with explicit
  uncertainty, not as identity proof.
