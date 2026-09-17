---
name: query-rules
description: "Use before running a tool, to run it the way its domain requires and keep the two false negatives out of the answer."
---

# Sea-Video Query Rules

**Tracks.** Always carry a time window or a page limit; an unfiltered full scan is the most expensive call here. A hull-number filter only matches tracks whose recognition already agreed with that number — empty means "no track was recognised as this hull", never "the vessel was absent". When it comes back empty, change the filter instead of repeating the call.

**Keyframes / dedup.** Per-track work: feed at most about twenty tracks per call. Frames with no vector come back as discarded and a track left with nothing searchable is reported as unsearchable — carry that into the answer rather than dropping the track.

**Registry.** A hull lookup matches exactly or through an alias; the catalogue listing returns everything. A registry hit proves the record exists, not that the vessel appeared; a video track proves appearance, not membership. Whether a reference image is searchable is decided by the vector index, not by a flag on the row.

**Image matching.** One side is video keyframes, the other registry references. Scores band as confirmed / uncertain / mismatch; a mismatch is downgraded to uncertain when registry coverage for that track is incomplete, and only the best match per track is kept. Empty input is a soft failure returning a hint — so "never tried" and "tried but could not" stay distinguishable.

**Visual verification.** Pass a description, references, keyframes or clips. When the answer rests on comparing images, the verification tool's verdict outranks your own reading of a text description.
