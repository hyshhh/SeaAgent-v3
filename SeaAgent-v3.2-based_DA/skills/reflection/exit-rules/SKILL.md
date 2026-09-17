---
name: exit-rules
description: "Use to decide whether the loop may end, per question class."
---

# Sea-Video Exit Rules

**Existence by hull — staged.** Registry not read yet ⇒ not settled. Registry read and a searchable reference exists but no image comparison ran ⇒ not settled; the next step is the visual chain. After a registry read plus a visual comparison: matches above the confirmation band ⇒ settled as "suspected presence"; zero matches with zero surviving tracks ⇒ settled as "the registry has a record but the footage does not show it". Only when the registry was read and the visual comparison ran (or no searchable reference exists at all) may you settle on "not seen in the footage".

**Registry membership list.** A full time-window scan returning zero tracks ⇒ settled, "no vessel candidates in this range"; do not read the whole registry or match images afterwards. Tracks exist but the registry was never listed, or no scorable image comparison was formed ⇒ not settled. A text search run with the user's sentence does not count as the appearance step. Correct shape: a full track scan (then keyframes if there were tracks), then reuse of those keyframes with the catalogue listing and the image comparison. In the answer, the in-registry list carries only confirmed matches, the not-in-registry list only tracks whose best match is still a mismatch, and uncertain tracks are listed separately as pending.

**Registry only.** A hull lookup that succeeded settles it. A description question needs the catalogue listed **and** an appearance search run through it; an explicitly empty catalogue may be reported as not found. A list or count needs the catalogue listing.

**Counting.** Settled only when the dedup step produced a usable count. Say "at least N" while groups could still merge, and carry unsearchable tracks as a caveat.
