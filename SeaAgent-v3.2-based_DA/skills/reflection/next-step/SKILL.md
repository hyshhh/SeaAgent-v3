---
name: next-step
description: "Use to turn an unresolved evidence gap into one concrete, bounded action."
---

# Next Action

Name the tool, IDs/filters and expected evidence; reuse prior results and change the failed input.

- No tracks and relevant hull/registry question: `get_registry` or remove the hull filter and query tracks.
- Tracks without frames: `get_frames` for those IDs; frames without appearance search: `match_text` with a cleaned phrase.
- Registry/video identity gap: unfiltered tracks → frames → registry references → `match_image` (query=registry, gallery=video).
- Missing exact hull check: `match_hull` on recognized numbers. Missing count: `dedup_tracks` on full records and grouped frames.

Keep scans paged and per-track work to batches of at most 20. Never chain a raw full result into frames/dedup or give a vague “continue searching” instruction.
