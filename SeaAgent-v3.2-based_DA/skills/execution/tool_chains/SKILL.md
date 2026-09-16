---
name: tool_chains
description: Use for every data question to pick the chain that fits the operation, and the repair chain after an empty result.
---

# Sea-Video Tool Chains

1. Run the smallest set that settles the step. Never fetch keyframes before tracks, and never match text or images before you have keyframes or references.
2. Chain by reference: a later step takes the IDs a previous step returned, never a re-typed collection.
3. Reuse what this turn already fetched; never re-run a call whose equivalent already succeeded.

Chains:
- **video existence / list** — track query in the time window, then keyframes only if tracks came back.
- **appearance description** — tracks → keyframes → text match with a cleaned appearance phrase.
- **hull via the registry** — registry lookup on the hull number.
- **hull via the footage** — track query filtered by hull number, then keyframes.
- **visual gap-fill after a hull filter came back empty** — registry lookup → track query **without** the hull number → keyframes → image match (query = registry references, gallery = keyframes).
- **registry membership list** — first a full time-window track query with no hull filter; if it returns zero, stop; otherwise catalogue listing → image match over the keyframes already fetched, plus a hull match when the tracks carry readable numbers.
- **count** — unfiltered track query with no paging → keyframes → dedup.
- **registry-only description** — catalogue listing → text match over registry references.

Repair rules after an empty round:
- hull-filtered tracks = 0 and the registry was requested → look the registry up; do not repeat the identical track call.
- registry already read and a visual check still required → the next chain must contain an image match, with the hull filter dropped.
- a full scan returned zero tracks → stop; do not read the whole registry and do not match images.
- a step skipped on an empty dependency stays skipped until the upstream step is repaired.
- a text match run with the user's whole sentence is wrong → replace it with the catalogue listing plus an image match.
- for a broad registry-versus-video comparison, read the whole registry and reuse earlier results; a paginated track result is **not** full coverage.
