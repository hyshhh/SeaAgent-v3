---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- For total-entity threshold counts, call `dedup_tracks` and report raw tracks plus the high-/low-threshold range.
- For registry counts, call `match_entities` once; it uses low-threshold groups and returns total, registered and unregistered counts. Do not also call `dedup_tracks` unless the user asks for the threshold range.
- The service performs deduplication and image matching. Do not inspect or recount raw tracks, frames or registry records. Matching IDs and scores stay in the evidence panel, outside model context and shared state.
- Pass `time_range` only when specified; omission means all stored monitoring time. If `coverageComplete` is false, qualify the counts as incomplete.
