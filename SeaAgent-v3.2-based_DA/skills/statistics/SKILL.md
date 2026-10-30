---
name: statistics
description: "Plan entity counts, then verify registry candidates with the VLM tool."
---

# Statistics

- **Choose one count path:** total vessels → two-stage `dedup_tracks`; registered or unregistered → `match_entities` without deduplication. A total-only request must not run registry matching.
- For `dedup_tracks`, verify every high-threshold `duplicateCandidates` pair with `verify_vessel_images` using its `pairId`, `referenceKeyframeId` and `candidateKeyframeId`. Call `dedup_tracks` again with all VLM `match` pair IDs in `confirmedPairIds` and all match/mismatch pair IDs in `resolvedPairIds`.
- Report `exactShipCount` as exact deduplication and `fuzzyShipCount` as fuzzy deduplication, then state the range between them. A VLM mismatch prevents that pair from merging; uncertain/failure stays unresolved.
- `match_entities` performs embedding only and already gives definitive counts: scores at or above `image_match` are in `embeddingConfirmedCount`; scores at or below `image_exclude` are in `outsideRegistryCount`. Read `grayCandidateCount`, `unresolvedTrackCount`, `embeddingCoverageComplete`, `coverageReasons` and `nextStep`.
- Only when `nextStep=verify_vessel_images`, call `verify_vessel_images` for each gray `verificationCandidates` item, using that item's `matchedRegistryReferenceId` and `matchedKeyframeId`; issue independent calls in parallel when supported. Do not verify embedding-confirmed or below-exclusion tracks, and do not use the display `keyframeId`.
- Aggregate gray candidates by unique `trackId`: VLM `match` adds to `embeddingConfirmedCount`; VLM `mismatch` adds to `outsideRegistryCount`. Keep `unresolvedTrackCount`, VLM `uncertain` and failed calls unresolved; never count them in either side.
- Do not report a final registered/unregistered total until every candidate has a VLM result or the unresolved gap is explicitly stated. Treat `embeddingCoverageComplete=false`, truncated candidates, missing comparisons or any unresolved candidate as incomplete evidence.
- The current video and index are already bound by the Harness; do not request a video path. Copy the plan's `time_range` exactly. If the user gives no time, omit it to search all stored tracks. A null `appliedTimeRange` means all time; otherwise compare applied and planned ranges as absolute instants. A mismatch requires replanning.
- Keep per-track IDs, scores and VLM facts in the evidence panel; report counts, scope and material coverage gaps in the answer.
