---
name: summary
description: "Use when compressing results for the next phase: what to keep, what to truncate."
---

# Sea-Video Result Summary

1. Return IDs and counts, not payloads: track, keyframe, reference and clip IDs, plus the counts that matter.
2. Truncate long ID lists; a summary carrying hundreds of IDs is noise.
3. Never paste image paths, vectors or raw records into a summary.
4. State failures, skips and errors explicitly with the reason — a skipped step that looks successful is the most expensive kind of error.
5. One short sentence for the outcome, and list the unresolved parts separately.
