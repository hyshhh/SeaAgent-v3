---
name: query-rules
description: "Use for query edge cases that can create false negatives."
---

# Query Rules

- Never invent a time range. Omit `time_range` to search all stored monitoring time; explicit ranges use ISO-8601 with timezone.
- A hull filter includes only tracks whose recognized number matches. Empty means unrecognized/not matched, not absent; retry without that filter.
- `get_frames` reports discarded frames and tracks with no searchable frames; carry those gaps into the conclusion.
- Registry lookup proves a record exists, not video presence. A trajectory proves a candidate, not registry identity.
- `match_text` needs a nonempty gallery: video keyframes or registry references. `match_image` compares one side of each; never mix source types.
- Respect returned score bands and coverage. An incomplete image comparison cannot support a confident mismatch; verification verdict governs visual claims.
