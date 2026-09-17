---
name: arguments
description: "Use before issuing a call to get its arguments and dependencies right."
---

# Sea-Video Arguments And Dependencies

Required per tool: keyframe lookup needs track IDs; clip generation needs a track ID; registry lookup needs a hull number; hull matching needs a list of hull numbers; text matching needs a description plus a gallery when searching images; image matching needs a keyframe side and a registry-reference side; dedup needs full track records plus keyframes keyed by track; the evidence tool needs at least one keyframe, clip or reference ID.

1. Pass IDs a tool returned. Never invent a track, keyframe or reference ID.
2. A dependency that is missing, failed or empty means the step is skipped with the reason recorded — never called with placeholder arguments.
3. When an upstream result is empty, the dependent step does not run; repair the upstream first.
4. Dedup needs track records with id, start and end time, not bare IDs.
5. Keep `topK` small and purposeful: an appearance search over many tracks is a minutes-long job — narrow the candidate set instead of raising the limit.
