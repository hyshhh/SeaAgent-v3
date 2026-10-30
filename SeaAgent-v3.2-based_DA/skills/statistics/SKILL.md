---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- **Choose one count path:** total vessels → `dedup_tracks` (raw track count and high/low deduplicated counts); registered or unregistered → `match_entities` (one result, track counts without deduplication). Never registry-match a total-only request or deduplicate a registry count.
- Trust the service's `inRegistryCount` and `unregisteredCount`; do not recount records. In `match_entities`, embedding selects candidates and only VLM `match` counts as registered. Every other track counts as unregistered; use `embeddingCandidateCount`, `vlmMismatchCount`, `vlmUncertainCount`, `vlmFailedCount`, `coverageComplete` and `coverageReasons` to explain limitations.
- The current video and index are already bound by the Harness; do not request a video path. Copy the plan's `time_range` exactly. If the user gives no time, omit it to search all stored tracks. A null `appliedTimeRange` means all time; otherwise compare the applied and planned ranges as absolute instants. A mismatch requires replanning.
- Report the returned registered/unregistered counts. If `coverageComplete` is false, state the coverage gaps; unsearchable tracks still count as unregistered under this rule. Keep per-track IDs, scores and VLM facts in the evidence panel, not model context or shared state.
