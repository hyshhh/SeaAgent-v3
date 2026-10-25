---
name: statistics
description: "Use compact service-side entity statistics; matching and deduplication stay inside the tool."
---

# Statistics

- Choose exactly one `count_*` tool for a count question and pass the user's `time_range` only when present.
- The service reads all matching tracks, deduplicates them, compares registry membership when required, and returns counts plus uncertainty.
- Use `countTotalEntities` for all entities, `countRegistryEntities` for entities in the registry, and `countUnregisteredEntities` for confirmed non-registry entities.
- `countConfidence=uncertain`, `countIsUpperBound=true`, or nonzero `uncertainCount` means the result cannot support an absolute claim.
- Use the returned IDs with `show_evidence` only when the answer needs visual evidence. Never copy raw tracks, frames or registry rows into the conversation.
