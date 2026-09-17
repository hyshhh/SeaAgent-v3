---
name: next-step
description: "Use when a gap must become one concrete next instruction."
---

# Sea-Video Next Step

Write the next step as tools plus the arguments that matter, so the executor does not re-derive the intent.

- zero tracks and the registry never read → the registry lookup on the hull number.
- registry has references but no image comparison ran → registry → unfiltered track scan → keyframes → image match, query side registry references, gallery side video keyframes.
- membership list unfinished → the catalogue listing, then the image comparison over the keyframes already fetched.
- a text match was run with the user's sentence → replace it with the image comparison.
- tracks without keyframes → fetch keyframes for those track IDs.
- keyframes without an appearance search → the search over those keyframes with a cleaned phrase.
- no exact hull comparison → the hull matcher over the recognised numbers.
- a count without dedup → the dedup step over tracks and keyframes.

A next step must be cheap enough to finish: never ask for a whole-period scan, and never chain a
raw full track result into the keyframe or dedup step. Narrow the window or the candidate set inside
the instruction itself, and say which tracks or time range to use.

Always name the tools and their key arguments. For a visual step state which side is the registry and which is the video, and make sure the track query carries no hull number. Never write a vague "keep looking".
