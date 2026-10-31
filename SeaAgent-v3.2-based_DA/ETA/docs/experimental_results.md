# Observed experimental results

Run: `research-20261030T143150Z`

These are actual results from the current server database. They are proxy-label
and source-split results, not a reproduction of the paper's human-labelled
accuracy or its placeholder performance table.

## Data

- 214 tracks and 571 cached keyframes; all 571 vectors reconstructed.
- AH labels: 653 positive, 262 negative.
- CH labels: 643 positive, 272 negative.
- Main split: 655 train, 91 validation, and 169 test samples.
- Train/validation/test tracks: 156/19/39.
- No source group crosses a split boundary.

The stricter 5-second audit retained 257 candidates but produced 263 expanded
samples with only one negative AH label. It is retained as a design audit, not
as the main training set.

## Structural comparison

Intervals are 95% bootstrap intervals. Metrics are from the held-out source
split. Latency is batch-1 DPD forward time and excludes image encoding.

| Variant | AH accuracy | CH accuracy | Joint accuracy | Static Stop F1 | p50 latency |
|---|---:|---:|---:|---:|---:|
| MLP-Cur | 0.3550 [0.2781, 0.4260] | 0.5680 [0.4969, 0.6391] | 0.2663 | 0.0000 | 0.195 ms |
| MLP-Hist | 0.5030 [0.4259, 0.5740] | 0.7574 [0.6923, 0.8226] | 0.4142 | 0.0000 | 0.357 ms |
| SA-DH | 0.6331 [0.5621, 0.7101] | 0.7988 [0.7396, 0.8581] | 0.5385 | 0.4634 | 1.049 ms |
| CA-only | 0.5503 [0.4734, 0.6272] | 0.7396 [0.6686, 0.8047] | 0.4320 | 0.4025 | 0.420 ms |
| CA-SH | 0.6864 [0.6154, 0.7574] | N/A | N/A | N/A | 0.357 ms |
| ETA | 0.6805 [0.6036, 0.7516] | 0.7574 [0.6923, 0.8225] | 0.5325 | 0.4403 | 0.435 ms |

The track-clustered paired comparisons against ETA show:

| Comparator | Delta AH | Delta joint |
|---|---:|---:|
| MLP-Cur | -0.3254 [-0.4386, -0.2130] | -0.2663 [-0.3812, -0.1519] |
| MLP-Hist | -0.1775 [-0.2830, -0.0658] | -0.1183 [-0.2273, -0.0060] |
| SA-DH | -0.0473 [-0.1147, 0.0117] | +0.0059 [-0.0663, 0.0683] |
| CA-only | -0.1302 [-0.2171, -0.0552] | -0.1006 [-0.1976, -0.0114] |
| CA-SH | +0.0059 [-0.0481, 0.0608] | N/A |

ETA clearly improves AH and joint accuracy over both MLP baselines and the
CA-only ablation on this split. SA-DH is indistinguishable from ETA on joint
accuracy, while its measured batch-1 latency is about 2.4 times higher. CA-SH
matches ETA on AH and avoids the CH head, but cannot provide completeness
decisions, joint accuracy, or the paper's dual-head Stop semantics.

## Capacity sensitivity

| Capacity | Train samples | Test samples | AH | CH | Joint | Static Stop F1 | p50 latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| M=1 | 533 | 137 | 0.7226 | 0.7445 | 0.5401 | 0.3836 | 0.326 ms |
| M=2 | 655 | 169 | 0.6746 | 0.7574 | 0.5325 | 0.4403 | 0.366 ms |
| M=3 | 655 | 169 | 0.6805 | 0.7574 | 0.5325 | 0.4403 | 0.435 ms |
| M=4 | 655 | 169 | 0.6805 | 0.7811 | 0.5680 | 0.4910 | 0.478 ms |
| M=6 | 655 | 169 | 0.6982 | 0.7751 | 0.5680 | 0.4727 | 0.526 ms |

The current tracks have at most two historical predecessors before a candidate,
so capacities above that range exercise padding and runtime behavior rather
than genuinely longer histories. The data therefore do not support a general
claim that one capacity is optimal. Larger intervals overlap, and M=1 uses a
different sample count.

## Stop threshold sensitivity for ETA

| Completeness threshold | Static Stop F1 |
|---:|---:|
| 0.50 | 0.5838 |
| 0.70 | 0.5371 |
| 0.80 | 0.4403 |
| 0.90 | 0.3000 |

The threshold must be selected on validation data in a deployment protocol.
The table is reported as sensitivity analysis, not as permission to choose the
best value after seeing the test result.

## Reproducibility and calibration

Two independent full-model runs produced exactly the same point accuracy and
calibration values; `reports/reproducibility.json` records
`verdict=REPRODUCIBLE` with maximum relative difference `0.0` under a 5%
tolerance. Timing is excluded from that comparison.

The ETA checkpoint has expected calibration error `0.1674` for AH and `0.1927`
for CH. These are moderate calibration errors, so the probabilities should not
be interpreted as precise confidence estimates.

## Overall interpretation

The historical-conditioned models outperform a current-frame MLP. The serial
ETA and same-token SA-DH have statistically indistinguishable joint accuracy on
this split, while ETA is materially faster in the measured setting. The
experiment therefore supports ETA's efficiency rationale on the available
data, not a universal accuracy advantage. Replacing human TEA labels with proxy
rules remains the largest limitation, followed by the pooled-embedding
approximation and small source-split sample size.
