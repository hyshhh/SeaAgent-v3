---
name: recognition
description: "Search tracks, then run the required registry or visual verification step for unresolved candidates."
---

# Recognition

- Search all stored time unless the user gives a range. Use `recognize_tracks.hull_number` for a hull/alias, `description` for appearance, and `image_path` for an attached image.
- Follow `nextStep` in order:
  1. If a hull/alias search is not confirmed and returns `match_registry_tracks`, call it once with the same hull and time range.
  2. If a search returns `verify_vessel_images`, verify each uncertain candidate, one pair per call. Do not skip this step or answer from embedding scores alone.
- Pair VLM evidence by query type: registry-assisted hull search (including aliases such as “小红”) uses `reference_registry_reference_id` + `candidate_keyframe_id`; description search uses `reference_description` + `candidate_keyframe_id`; image search uses `reference_image` + `candidate_keyframe_id`. Copy the reference ID and matched frame ID from the same candidate.
- Use `matchedKeyframeId` for VLM input, not the representative display `keyframeId`. The registry reference must come from the same candidate's match result.
- A registry entry alone does not prove video presence. Keep embedding scores, VLM decisions and coverage separate. `uncertain` or incomplete coverage cannot prove presence or absence.
- If VLM returns `formatRecovered=true`, use only its explicit `decision`; the malformed visual facts were discarded, so do not quote or invent supporting details.
- Image path or encoding errors mean the search failed, not that the target is absent. Never read raw track files or registry listings.
