---
name: statistics
description: "Answer entity-count questions with one service-side count tool."
---

# Statistics

- For a plain total-vessel count, call `dedup_tracks` and report raw track count plus high-/low-threshold totals. Do not call `match_entities` for a total-only request.
- For registered or unregistered counts, call `match_entities` once, even when phrased as “how many vessels.” It matches every track in scope to registry images without deduplication and returns total, registered and unregistered track counts. Do not also call `dedup_tracks` unless the user asks for both counting methods.
- The service performs deduplication or image matching as required. Do not inspect or recount raw tracks, frames or registry records. Matching IDs and scores stay in the evidence panel, outside model context and shared state.
- Pass `time_range` exactly as written in the current plan; never recompute a relative date or reuse a range from older QA history. If the current question has no time condition, omit it to search all stored monitoring time; do not ask the user for a range. Check `appliedTimeRange` against the plan by comparing absolute instants; `null` means all stored monitoring time. If they differ, do not use that result—request replan. If `coverageComplete` is false, qualify the counts as incomplete.
