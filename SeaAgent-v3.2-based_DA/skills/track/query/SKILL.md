---
name: query
description: "Use for trajectory searches and time-scoped video questions."
---

# Track Queries

- Omit `time_range` for all stored monitoring time; otherwise pass timezone-aware ISO-8601.
- `get_track` pages default to 50 (max 100). For complete coverage, follow `nextCursor` while `hasMore`; keep filters fixed and reset the cursor after changing them.
- `hull_number` excludes tracks without that recognized number. If empty, remove the filter before concluding absence.
- Stop on reliable positive evidence for an existence question. A negative requires all relevant pages and candidate checks.
- Reuse returned IDs; fetch frames in batches of at most 20 and report unsearchable/failed candidates.
