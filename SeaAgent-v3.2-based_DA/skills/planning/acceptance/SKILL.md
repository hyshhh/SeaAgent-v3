---
name: acceptance
description: "Use when writing the acceptance checklist: which evidence must exist before the answer counts as established."
---

# Sea-Video Acceptance Checklist

The checklist is the contract for the turn: the executor collects against it, the reflector audits it item by item.
Write one item per requirement, with `how_to_check` naming the tool and field that settles it.

- **existence + hull** — registry lookup first, then a time-window track query. An empty hull-filtered query is **not** an answer: an unfiltered scan and a visual comparison are still required before "not seen".
- **existence + description** — track query, keyframes, then an appearance match.
- **count** — tracks, keyframes and dedup. A raw row count is not a count of vessels.
- **list** — the time-window track scan; add the registry (listing + image match) only when membership is asked.
- **time** — track start/end timestamps, keyframes when available.
- **explain** — tracks and keyframes, plus an image match when identity matters.
- **registry only** — catalogue listing and, for an appearance question, a text match against registry references. No video tools.

Never accept as sufficient on their own:
- "video OCR did not read a hull number" (recognition failure also returns empty),
- "a hull-filtered query returned zero tracks",
- a checklist whose only step is a track query when the target is a hull number.
