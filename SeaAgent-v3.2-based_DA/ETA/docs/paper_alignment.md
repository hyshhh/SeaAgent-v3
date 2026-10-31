# Paper-to-implementation alignment

| Paper contract | Implementation | Alignment |
|---|---|---|
| Candidate query from frozen visual encoding | Precomputed normalized Qwen embedding projected to 256 dimensions | Partial: encoder frozen, but pooling already applied |
| History frame `F_i` as Key/Value | 2048 vector split into eight 256 groups and projected as tokens | Approximate: pooled vector, not raw tokens |
| Serial chronological Q-Former | Historical slots processed in timestamp order | Aligned |
| Two stages, two layers per stage, shared parameters | Two shared Post-LN blocks reused across both stages | Aligned |
| Four heads, FFN width 1024 | Configured in `configs/default.yaml` | Aligned |
| Independent AH and CH heads | Independent two-layer MLPs, width 128, two outputs | Aligned |
| Candidate pool CEP with quality floor | Quality floor enforced before AH and in online pool | Aligned with available scalar quality |
| KEP capacity `M=3` | Default capacity 3; capacities 1,2,3,4,6 evaluated | Aligned |
| Replacement by admission and priority | Online pool replaces only the lowest-priority entry | Aligned |
| `Stop = inserted AND p_c >= threshold` | `EvidencePool` enforces actual insertion; threshold is configurable, with 0.8 used in the main run | Aligned |
| AdamW, LR `1e-4 -> 1e-6`, WD `1e-2` | Implemented with two warmup epochs and cosine decay | Aligned |
| Batch 128, 30 epochs, seed 42 | Default configuration | Aligned |
| Label-frequency weighting, equal heads | Weighted cross entropy and equal AH/CH loss weight | Aligned |
| Dual-head checkpoint by joint accuracy | Implemented; single-head ablation uses AH accuracy | Aligned |
| Human TEA labels, 491/4394 train tracks/crops | Current data provides 156 training tracks and 655 expanded samples with proxy labels | Major difference |
| Train/validation domain isolation | Seeded source-video isolation | Partial: adapts to available data to avoid an unusable split |
| 5-second candidate sampling | Optional; main run uses all 571 sparse keyframes | Deviation justified by degenerate pilot target |
| AH accuracy, CH accuracy, joint accuracy, batch-1 latency | Implemented | Aligned |

The code implements the paper's model and decision semantics. It does not claim
to reproduce the paper's dataset scale, human labels, or reported performance
values.
