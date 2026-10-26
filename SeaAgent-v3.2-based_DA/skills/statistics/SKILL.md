---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- Map each requested measure to one tool: total `count_total_entities`, registered `count_registry_entities`, unregistered `count_unregistered_entities`. Never call unrequested count tools. Pass `time_range` only when specified; omission means all stored monitoring time.
- The service performs track deduplication and registry matching internally. Do not inspect or recount raw tracks, frames or registry records.
- Qualify the result if `countComplete` is false, `countStability` is not `stable`, or `uncertainCount`/`pendingMergeCount` is nonzero.
- `minimumCount` is available only from `count_total_entities` when deduplication is unstable; do not expect it from registry counts.
- Use returned evidence IDs with `show_evidence` only when visual evidence is needed.
