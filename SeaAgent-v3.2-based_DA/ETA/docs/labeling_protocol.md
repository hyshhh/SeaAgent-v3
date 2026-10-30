# Proxy-label protocol

## Status

These are deterministic weak labels created from the available memory records.
They are not human annotations and must not be described as such. The paper's
TEA protocol requires annotators to inspect images and judge both decisions;
that information is absent from the current database.

## Admission Head (AH)

Let `q` be the current quality score, `v_c` the current normalized embedding,
`v_i` a historical KEP embedding, and `h` a readability flag.

1. If `q < 0.20`, AH is `0` with reason `below_quality_floor`.
2. If the KEP history is empty, AH is `1` with reason `pool_initialization`.
3. Otherwise compute novelty `1 - max_i cosine(v_c, v_i)`.
4. AH is `1` when novelty is at least `0.10` or the candidate is the first
   history member with a readable hull number. Otherwise AH is `0`.

The candidate priority is `0.7*q + 0.3*max_history_quality`. When the updated
pool exceeds capacity `M`, lower-priority entries are removed.

## Completeness Head (CH)

CH is computed after applying the AH decision and capacity trim. It is `1` when
at least one of these proxy conditions holds:

- the updated pool contains readable hull-number evidence;
- the minimum pairwise cosine similarity in the pool is at most `0.96`,
  indicating a different view;
- the pool contains at least two frames and its maximum quality is at least
  `0.75`.

An empty pool is incomplete. Every record stores the reason used by AH and CH,
so the label derivation is auditable.

## History construction

For the initial candidate of a track, history is empty. For every later
candidate, all chronological non-empty subsets up to capacity `M=3` are
expanded. This avoids the degenerate case in which every later candidate is
paired with an empty pool and therefore receives a trivial positive AH label.

## Validation status

The protocol is logically valid for software testing but weak for measuring
human decision quality. It encodes quality, novelty, readability, and visual
coverage assumptions before model training. Reported model accuracy measures
agreement with those rules, not with expert judgement.
