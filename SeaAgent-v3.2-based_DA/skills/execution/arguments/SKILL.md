---
name: arguments
description: "Use to check tool inputs and dependencies before calling."
---

# Tool Arguments

- Use returned IDs only: `get_frames` needs track IDs; `get_clip` a track ID; registry lookup a hull; hull matching a list of hulls.
- `match_text` needs a description and nonempty image gallery. `match_image` needs video frames on one side and registry references on the other.
- Dedup needs full tracks with `trackId/startTime/endTime` and keyframes grouped by track. `show_evidence` accepts keyframe, clip and reference IDs.
- Missing/failed/empty dependencies mean skip with reason; never use placeholders. Keep `top_k` targeted and small.
