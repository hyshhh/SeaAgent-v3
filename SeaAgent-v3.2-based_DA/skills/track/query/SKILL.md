---
name: query
description: "Use for every Sea-Video question. Select the minimum evidence path and preserve the requested time scope."
---

# Sea-Video Query

1. Determine the user's goal from the request, not from fixed keyword rules.
2. Convert any explicit time expression into `time_range` before querying tracks.
3. Choose the smallest tool sequence that can establish the answer.
4. Reuse IDs returned by tools; never paste an unbounded result set into a later call.
5. If the requested claim is not established, keep investigating or report the evidence gap.
