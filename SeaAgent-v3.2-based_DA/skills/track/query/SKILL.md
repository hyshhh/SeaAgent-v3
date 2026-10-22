---
name: query
description: "Use for every Sea-Video question. Select the minimum evidence path and preserve the requested time scope."
---

# Sea-Video Query

1. Determine the user's goal from the request, not from fixed keyword rules.
2. Omit `time_range` when the user gives no time condition; this searches all stored monitoring time.
3. For explicit time conditions, pass ISO-8601 timestamps with a timezone; the tool converts them to epoch seconds.
4. Read at most one `get_track` page at a time. Follow `nextCursor` while `hasMore` when a complete search is required; pass only small track batches to `get_frames`. Reset the cursor whenever filters change.
5. `hull_number` excludes tracks without a recognized hull number. If a hull-filtered query is empty, remove that filter before concluding anything about absence.
6. Stop after reliable positive evidence when the question asks whether something ever occurred. Claim no match only after all relevant pages and candidate evidence have been checked.
7. Reuse IDs returned by tools; never paste an unbounded result set into a later call. If the claim is not established, report the evidence gap.
