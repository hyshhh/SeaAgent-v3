---
name: tool-chains
description: "Use to choose the shortest valid tool chain and repair empty results."
---

# Tool Chains

Reuse successful results; pass returned IDs forward. Never call a dependent tool with missing/empty input.

- Video presence/list: `get_track`; get frames only for visual checks.
- Appearance: tracks → `get_frames` → `match_text` with a cleaned phrase.
- Hull in registry: `get_registry`/`match_hull`; hull in video: filtered tracks, then remove hull filter if empty.
- Registry-vs-video: complete track scope → frames → `list_registry` → `match_image` (`query_images`=registry references, `gallery_images`=video frames).
- Registry description: `list_registry` → `match_text` over `registryReferences`.
- Count: complete track pages → frames → `dedup_tracks`; raw rows are not vessel counts.

Keep per-track work small: filter/page before `get_frames` or dedup; batches should not exceed 20 tracks. Dedup only when counting. A wide scan may need pagination; never claim complete coverage from one page. After empty results, repair the failed upstream/filter; do not repeat an identical successful call. If registry coverage is incomplete, preserve uncertain results.
