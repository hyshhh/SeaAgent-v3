---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- Use exactly one matching tool: `count_total_entities`, `count_registry_entities` or `count_unregistered_entities`. Pass `time_range` only when the user specified one; omission means all stored monitoring time.
- The service performs track deduplication and registry matching internally. Do not inspect or recount raw tracks, frames or registry records.
- Qualify the result if `countComplete` is false, `countStability` is not `stable`, or `uncertainCount`/`pendingMergeCount` is nonzero.
- `minimumCount` is available only from `count_total_entities` when deduplication is unstable; do not expect it from registry counts.
- Use returned evidence IDs with `show_evidence` only when visual evidence is needed.
