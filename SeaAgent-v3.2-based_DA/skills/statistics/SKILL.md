---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- For total entities, call `count_total_entities`: report raw tracks and the high-/low-threshold entity counts as a conservative range.
- For registered or unregistered counts, call `count_entities_by_registry` once; it returns total, registered and unregistered entities together. Mixed requests may also call `count_total_entities`.
- The service performs deduplication and image matching. Do not inspect or recount raw tracks, frames or registry records. Matching IDs and scores stay in the evidence panel, outside model context and shared state.
- Pass `time_range` only when specified; omission means all stored monitoring time. If `coverageComplete` is false, qualify the counts as incomplete.
