---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- **Choose one count path:** total vessels → `dedup_tracks` (raw track count and high/low deduplicated counts); registered or unregistered → `match_entities` (one result, track counts without deduplication). Never registry-match a total-only request or deduplicate a registry count.
- Trust the service's counts; do not inspect or recount records. `inRegistryCount` is an embedding-threshold result, not VLM confirmation. `unregisteredCount` includes only tracks compared with the full searchable registry and below the exclusion threshold. Gray scores, missing vectors and un compared tracks are `unclassifiedTrackCount`.
- The current video and index are already bound by the Harness; do not request a video path. Copy the plan's `time_range` exactly. If the user gives no time, omit it to search all stored tracks. A null `appliedTimeRange` means all time; otherwise compare the applied and planned ranges as absolute instants. A mismatch requires replanning.
- Report nonzero unclassified tracks separately. If `coverageComplete` is false, state the `coverageReasons`; do not present a partial unregistered count as the total, count unknown tracks as unregistered, or infer their registry status. Keep per-track IDs and scores in the evidence panel, not model context or shared state.
