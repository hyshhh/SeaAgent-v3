---
name: statistics
description: "Plan entity counts, then verify registry candidates with the VLM tool."
---

# Statistics

- **Choose one count path:** total-only vessels → two-stage `dedup_tracks`; registered and unregistered quantities → the same `match_entities` pipeline without deduplication. Apply this registry path only when the user explicitly asks how many vessels are registered or unregistered. Recognition, description, image, hull-number and total-only questions must not use it or append registry counts.
- For `dedup_tracks`, only exact-match gray-zone `duplicateCandidates` require `verify_vessel_images`, using each pair's `pairId`, `referenceKeyframeId` and `candidateKeyframeId`. High-threshold pairs are already exact. Call `dedup_tracks` again with all VLM `match` pair IDs in `confirmedPairIds` and all match/mismatch pair IDs in `resolvedPairIds`.
- Deduplication returns at most 12 gray candidate pairs. Additional gray pairs are already treated as non-duplicates for exact matching and must not be sent to VLM.
- Report `exactShipCount` as exact deduplication and `fuzzyShipCount` as fuzzy deduplication, then state the range between them. A VLM mismatch prevents that pair from merging; uncertain/failure stays unresolved.
- `match_entities` performs embedding only and already gives definitive counts: scores at or above `image_match` are in `embeddingConfirmedCount`; scores at or below `image_exclude` are in `outsideRegistryCount`. Read `grayCandidateCount`, `unresolvedTrackCount`, `embeddingCoverageComplete`, `coverageReasons` and `nextStep`.
- Registry matching returns at most 12 gray `verificationCandidates`; all additional gray tracks are already classified outside. Only when `nextStep=verify_vessel_images`, verify each returned item with its `matchedRegistryReferenceId` and `matchedKeyframeId`; do not use the display `keyframeId`.
- Compute one canonical registry result: `total = trackCount`; `registered = embeddingConfirmedCount + gray VLM match`; `unregistered = total - registered`. Registered and unregistered questions must report the same three values.
- Aggregate gray candidates by unique `trackId`: VLM `match` adds to `registered`; VLM `mismatch` confirms the remaining tracks are not registered but does not replace subtraction from `trackCount`. Keep `unresolvedTrackCount`, VLM `uncertain` and failed calls explicit when coverage is incomplete.
- Do not report a final registered/unregistered total until every candidate has a VLM result or the unresolved gap is explicitly stated. Treat `embeddingCoverageComplete=false`, truncated candidates, missing comparisons or any unresolved candidate as incomplete evidence.
- The current video and index are already bound by the Harness; do not request a video path. Copy the plan's `time_range` exactly. If the user gives no time, omit it to search all stored tracks. A null `appliedTimeRange` means all time; otherwise compare applied and planned ranges as absolute instants. A mismatch requires replanning.
- Keep per-track IDs, scores and VLM facts in the evidence panel. Only for registered or unregistered quantity questions, clearly state `总数：{total}`、`在库船：{registered}`、`不在库船：{unregistered}` in natural prose or labels; an exact sentence template is not required. Never append these metrics to recognition or total-only answers. Add scope or material coverage gaps when relevant.
