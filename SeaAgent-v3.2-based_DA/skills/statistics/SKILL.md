---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- For a plain total-vessel count, call `dedup_tracks` and report raw track count plus high-/low-threshold totals. Do not call `match_entities` for a total-only request.
- For registered or unregistered counts, call `match_entities` once, even when phrased as “how many vessels.” It matches every track in scope to registry images without deduplication and returns total, registered and unregistered track counts. Do not also call `dedup_tracks` unless the user asks for both counting methods.
- The service performs deduplication or image matching as required. Do not inspect or recount raw tracks, frames or registry records. Matching IDs and scores stay in the evidence panel, outside model context and shared state.
- Pass `time_range` only when specified; omission means all stored monitoring time. If `coverageComplete` is false, qualify the counts as incomplete.
