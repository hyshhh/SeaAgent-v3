---
name: statistics
description: "Use compact service-side entity statistics; matching and deduplication stay inside the tool."
---

# Statistics

- Choose exactly one `count_*` tool for a count question and pass the user's `time_range` only when present.
- The service reads all matching tracks, deduplicates them, compares registry membership when required, and returns counts plus uncertainty.
- Use `countTotalEntities` for all entities, `countRegistryEntities` for entities in the registry, and `countUnregisteredEntities` for confirmed non-registry entities.
- `countComplete=false`, non-stable `countStability`, nonzero `uncertainCount` or `pendingMergeCount` means qualify the count; `minimumCount` is the lower estimate when deduplication is unstable.
- Use returned evidence IDs with `show_evidence` only when visual evidence is needed. Never copy raw tracks, frames or registry rows.
