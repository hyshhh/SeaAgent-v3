---
name: statistics
description: "Plan entity counts, then verify registry candidates with the VLM tool."
---

# Statistics

- **Choose one count path:** total vessels → `dedup_tracks`; registered or unregistered → `match_entities` without deduplication. A total-only request must not run registry matching.
- `match_entities` performs embedding only. Read `embeddingCandidateCount`, `outsideRegistryCount`, `unresolvedTrackCount`, `embeddingCoverageComplete`, `coverageReasons` and `nextStep`. When `nextStep=verify_vessel_images`, call `verify_vessel_images` once for every `verificationCandidates` item, using that item's `matchedRegistryReferenceId` and `matchedKeyframeId`; issue independent candidate calls in parallel when supported. Do not use the display `keyframeId`.
- Aggregate by unique candidate `trackId`: VLM `match` counts in registry; VLM `mismatch` counts outside registry. Add `outsideRegistryCount` to mismatch tracks for the confirmed outside total. Keep `unresolvedTrackCount`, VLM `uncertain` and failed calls unresolved; never count them in either side.
- Do not report a final registered/unregistered total until every candidate has a VLM result or the unresolved gap is explicitly stated. Treat `embeddingCoverageComplete=false`, truncated candidates, missing comparisons or any unresolved candidate as incomplete evidence.
- The current video and index are already bound by the Harness; do not request a video path. Copy the plan's `time_range` exactly. If the user gives no time, omit it to search all stored tracks. A null `appliedTimeRange` means all time; otherwise compare applied and planned ranges as absolute instants. A mismatch requires replanning.
- Keep per-track IDs, scores and VLM facts in the evidence panel; report counts, scope and material coverage gaps in the answer.
