# Research integrity and limitations

## Evidence status

Every numerical result in `reports/` is generated from a saved run directory
containing the resolved config, environment, dataset hashes, epoch history,
checkpoint, predictions, and evaluation artifacts. Paper tables that are marked
as placeholders are never copied into the result reports.

## Eleven statistical and methodological fallacies

| # | Risk | Status in this study |
|---:|---|---|
| 1 | Simpson's paradox | Not estimable with the current source groups; no pooled causal claim is made. |
| 2 | Ecological fallacy | Not applicable: predictions and labels are at the sample level. |
| 3 | Berkson's paradox | Caution: availability of embedded tracks is an archive-selection mechanism. |
| 4 | Collider bias | Not applicable: no multivariable causal adjustment is performed. |
| 5 | Base-rate neglect | Caution addressed by reporting class counts, precision/recall, F1, and class-weighted loss. |
| 6 | Regression to the mean | Not applicable: no pre/post extreme-group intervention. |
| 7 | Survivorship bias | Caution: only tracks with reconstructable keyframe vectors are included; 571/571 available frame rows reconstructed. |
| 8 | Look-elsewhere effect | Six structural variants and five capacities were declared in configs; no claim is selected solely by the best test score. |
| 9 | Garden of forking paths | Caution: the pilot 5-second build produced a degenerate AH target, so the main design uses all sparse keyframes. This revision is disclosed before main training. |
| 10 | Correlation is not causation | No causal language is used for architecture comparisons. |
| 11 | Reverse causality | Not applicable: training uses fixed observed features and supervised targets. |

These labels assess statistical and methodological risks; they do not certify
the scientific value of the model.

## Seven-mode AI research failure audit

| Mode | Status | Evidence |
|---|---|---|
| 1. Implementation bug passing review | CLEAR at current stage | Model, data, metrics, pool and CLI contracts are tested; full repository checks are required before release. |
| 2. Hallucinated citation | NOT APPLICABLE | This module reports experiments and does not add manuscript citations. |
| 3. Hallucinated experiment result | CLEAR at current stage | Reports are generated from run manifests and metrics files; paper placeholder numbers are excluded. |
| 4. Shortcut reliance | SUSPECTED | Proxy labels use quality and readability, so the model may learn those shortcuts; structural ablations and limitation labels are required. |
| 5. Bug reframed as insight | CLEAR | No unexpected behavior is promoted as a finding; results remain descriptive. |
| 6. Methodology fabrication | CLEAR | Training and evaluation read the resolved config and actual run artifacts. |
| 7. Early-stage frame lock | SUSPECTED | The available data cannot reproduce the paper's human-label scale; all conclusions remain bounded to the proxy-labelled subset. |

## Deployment limitations

- AH/CH probabilities are not identity proof.
- Stop should trigger only after an actual KEP update.
- Pool aggregation and semantic-index refresh are outside this isolated module.
- Human review is required before using this checkpoint in operational video
  analysis or before reporting its accuracy as a human-labelled result.

## Verification record

- Full repository tests: `354 passed, 1 skipped`.
- Full repository tests in the GPU environment: `360 passed`.
- ETA tests in the GPU environment: `24 passed`.
- ETA Ruff check: passed.
- ETA Python compile check: passed.
- The project-wide Ruff parse reports 140 pre-existing findings outside `ETA`;
  these files were intentionally not modified.
- Two independent full-model runs matched all reported non-timing metrics:
  verdict `REPRODUCIBLE`, maximum relative difference `0.0`.
- The strict 5-second audit retained 257 candidates but produced a `262:1`
  AH target, so it was not used for the main training run.
