# Automated ETA research run: research-20261030T143150Z

> Status: proxy-label research run. AH/CH labels are deterministic weak labels,
> not the paper's human annotations. Results cannot be presented as reproduction
> of the paper's target accuracy.

| Experiment | AH acc. [95% CI] | CH acc. [95% CI] | Joint acc. | Static Stop F1 | Latency p50 (ms) |
|---|---:|---:|---:|---:|---:|
| ablation:mlp_cur | 0.3550 [0.2781, 0.4260] | 0.5680 [0.4969, 0.6391] | 0.2663 | 0.0000 | 0.195 |
| ablation:mlp_hist | 0.5030 [0.4259, 0.5740] | 0.7574 [0.6923, 0.8226] | 0.4142 | 0.0000 | 0.357 |
| ablation:sa_dh | 0.6331 [0.5621, 0.7101] | 0.7988 [0.7396, 0.8581] | 0.5385 | 0.4634 | 1.049 |
| ablation:ca_only | 0.5503 [0.4734, 0.6272] | 0.7396 [0.6686, 0.8047] | 0.4320 | 0.4025 | 0.420 |
| ablation:ca_sh | 0.6864 [0.6154, 0.7574] | N/A | N/A | N/A | 0.357 |
| ablation:eta | 0.6805 [0.6036, 0.7516] | 0.7574 [0.6923, 0.8225] | 0.5325 | 0.4403 | 0.435 |
| capacity:1 | 0.7226 [0.6496, 0.8029] | 0.7445 [0.6642, 0.8175] | 0.5401 | 0.3836 | 0.326 |
| capacity:2 | 0.6746 [0.5976, 0.7457] | 0.7574 [0.6923, 0.8225] | 0.5325 | 0.4403 | 0.366 |
| capacity:3 | 0.6805 [0.6036, 0.7516] | 0.7574 [0.6923, 0.8225] | 0.5325 | 0.4403 | 0.435 |
| capacity:4 | 0.6805 [0.6036, 0.7574] | 0.7811 [0.7160, 0.8402] | 0.5680 | 0.4910 | 0.478 |
| capacity:6 | 0.6982 [0.6213, 0.7692] | 0.7751 [0.7160, 0.8343] | 0.5680 | 0.4727 | 0.526 |

## Interpretation boundary

- Accuracy is measured on the held-out source-video split with proxy labels.
- Intervals are 1,000-resample non-parametric bootstrap 95% intervals.
- Batch-1 latency excludes image encoding and includes DPD forward only.
- Static Stop F1 measures two-head threshold agreement on saved samples.
  Actual insertion gating is enforced by `EvidencePool` and covered by tests;
  sequence-level online replay remains future work.
- Timing is environment-sensitive and is not treated as reproducible across GPUs.
- All 11 statistical fallacies and the relevant integrity checks are recorded in
  `docs/integrity_and_limitations.md`.
