# TEA data card

## Source

- Application: `SeaAgent-v3.2-based_DA`
- Track table: `data/memory/tracks.csv`
- Keyframe table: `data/memory/track_keyframes.csv`
- Trajectory metadata: `data/memory/trajectories/*.json`
- Vector index: `data/indexes/keyframe_index.faiss`
- Embedding model recorded by the index manifest:
  `Qwen/Qwen3-VL-Embedding-2B`
- Embedding dimension: 2048, L2-normalized

## Available observations

- 214 tracks.
- 571 cached keyframes.
- 571 vectors reconstructed successfully.
- Frames per track: 29 tracks with one, 13 with two, and 172 with three.
- Because no track has more than three cached frames, capacity values at or
  above `M=3` cannot be distinguished by additional history slots on this
  snapshot; `M=4` and `M=6` are infrastructure checks rather than independent
  empirical evidence about longer pools.
- 222 keyframes have readable hull-number evidence.
- 45 known source videos: 191 tracks from public video groups and 12 from the
  self-collected group. Eleven tracks lack usable trajectory source metadata
  and are isolated by track ID.

## Split

The deterministic source-group build used for the main run contains:

| Split | Samples | Tracks | Source groups |
|---|---:|---:|---:|
| Train | 655 | 156 | 39 |
| Validation | 91 | 19 | 9 |
| Test | 169 | 39 | 8 |

Source groups do not overlap. The 70/15/15 split is randomized by a seeded hash,
not by the paper's domain-held-out protocol, because using the 12
self-collected tracks alone would leave too little training data.

| Split | AH negative/positive | CH negative/positive |
|---|---:|---:|
| Train | 191 / 464 | 207 / 448 |
| Validation | 35 / 56 | 30 / 61 |
| Test | 36 / 133 | 35 / 134 |

## Labels

- AH: 653 positive and 262 negative samples.
- CH: 643 positive and 272 negative samples.
- Labels are deterministic proxies described in `labeling_protocol.md`.

## Known coverage gaps

- The paper reports a much larger human-labelled TEA set. This database is a
  software and protocol validation subset, not the paper's target dataset.
- The stored vectors are pooled embeddings, while the paper specifies raw
  2048-dimensional visual tokens.
- The source table provides one precomputed scalar `quality_score`; the four
  raw quality components for independently applying the paper's weighted CEP
  equation are not stored.
- Applying the paper's 5-second sampling to this short-track archive leaves 257
  candidate frames and a `262:1` AH target even after the non-empty-history
  correction. The audit build is stored in `reports/stride5_summary.json`. The
  main run uses all 571 already-sparse memory frames and records this deviation.
- Manual adjudication, inter-annotator agreement, and label error rates are not
  available.
