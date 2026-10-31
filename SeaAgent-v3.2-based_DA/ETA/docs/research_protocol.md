# ETA research protocol

## Research questions

1. Does chronological cross-attention over the current KEP improve admission
   and completeness decisions relative to current-frame and pooled-history
   baselines?
2. Does a serial cross-attention architecture outperform a same-token
   self-attention alternative?
3. How do residual/FFN fusion, dual-head supervision, and KEP capacity affect
   accuracy and batch-1 latency?

## Pre-specified comparisons

The structural comparisons are `mlp_cur`, `mlp_hist`, `sa_dh`, `ca_only`,
`ca_sh`, and the full `eta` model. Capacity is evaluated at `M={1,2,3,4,6}`.
All non-capacity variants use the same samples, source split, seed, optimizer,
epochs, batch size, and checkpoint rule.

Primary outcomes are AH accuracy, CH accuracy, joint accuracy for dual-head
models, Stop-rule F1 at the configured completeness threshold, and batch-1
latency. Expected calibration error, confusion matrices, precision, recall,
F1, Brier score, and bootstrap 95% intervals are reported as supporting
diagnostics. Intervals quantify finite-sample uncertainty and are not used to
claim statistical superiority.

Head-level intervals use sample bootstrap resampling. Architecture comparisons
additionally use 2,000 paired bootstrap draws clustered by test track. The
sample intervals are descriptive because windows from one track and source are
correlated; the paired track-clustered intervals control for that dependence.

## Data and split

The available trajectory database contains 214 tracks and 571 cached keyframes.
Each keyframe has a normalized 2048-dimensional Qwen embedding. Source groups
are assigned deterministically with seed 42 to approximately 70/15/15 splits.
No source video crosses a split boundary. The actual build written to
`data/processed/summary.json` is the authoritative size record.

The paper specifies 5-second sampling, but only 43 later candidates remain after
that operation in this database. A pilot build produced an AH target of `305:1`
because most tracks last less than 10 seconds. The main experiment therefore
uses every available cached keyframe, which is already a sparse memory. Every
non-initial observation is paired with non-empty historical KEP states. This
design revision was made before the main 30-epoch run and is recorded as an
experimental limitation.

The database does not contain the paper's human AH/CH annotations. Deterministic
proxy labels are used and are identified everywhere as such. The exact rules
are in `labeling_protocol.md`.

## Model and training

The candidate pooled embedding is projected to a 256-dimensional query. Because
the stored encoder output is pooled rather than raw visual tokens, each history
embedding is split into eight 256-dimensional groups and projected as eight
tokens. This is an implementation approximation, not a claim that pooled
features equal the paper's raw token stream.

The model uses shared Post-LN cross-attention blocks across two stages, two
layers per stage, four heads, a 1024-dimensional FFN, and independent
two-class AH/CH heads with hidden width 128. Training uses AdamW
(`lr=1e-4`, `min_lr=1e-6`, `weight_decay=1e-2`), label-frequency class weights,
two warmup epochs, cosine decay, batch size 128, 30 epochs, and seed 42. The
best validation joint accuracy selects dual-head checkpoints; single-head
ablation checkpoints use AH accuracy. The test split is evaluated from the
selected checkpoint without using test scores for training or hyperparameter
selection.

## Reproducibility and integrity

Each run stores the resolved config, data hashes, Git revision, environment,
class weights, per-epoch history, checkpoint metadata, predictions, calibration
bins, confusion matrices, and latency distribution. Stochastic timing is
environment-sensitive. Accuracy intervals use 1,000 non-parametric bootstrap
resamples with seed 42.

The study does not support causal claims, does not treat proxy accuracy as human
label accuracy, and does not use the paper's placeholder performance values as
observed evidence. The 11-fallacy scan and seven-mode research integrity audit
are recorded in `integrity_and_limitations.md`.
