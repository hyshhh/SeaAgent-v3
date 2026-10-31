---
name: recognition
description: "Search tracks, then run the required registry or visual verification step for unresolved candidates."
---

# Recognition

- A model tool-call batch may combine only `read_file`, `read_shared_state` and `verify_vessel_images`. Emit one domain search or state write per model turn.
- Search all stored time unless the user gives a range. Use `recognize_tracks.hull_number` for a hull/alias, `description` for appearance, and `image_path` for an attached image.
- Follow the returned `nextStep` before concluding. For a hull/alias query, if direct status is not `confirmed` and `nextStep` or `recommendation` names `match_registry_tracks`, call it once with the same query and time range. This does not depend on `image_path`; do not conclude absence before the registry-assisted search.
- If a result names `verify_vessel_images`, submit all returned uncertain candidates that contain `matchedKeyframeId` in the same assistant tool-call batch. Each call verifies one pair, and all calls for the current tool result must be emitted before waiting for any result. The initial tool caps recognition gray candidates at 5. Do not answer from embedding scores alone.
- Pair VLM evidence by query type: registry-assisted hull search (including aliases such as “小红”) uses `reference_registry_reference_id` + `candidate_keyframe_id`; description search uses `reference_description` + `candidate_keyframe_id`; image search uses `reference_image` + `candidate_keyframe_id`. Copy the reference ID and matched frame ID from the same candidate.
- Use `matchedKeyframeId` for VLM input, not the representative display `keyframeId`. The registry reference must come from the same candidate's match result.
- A registry entry alone does not prove video presence. Keep embedding scores, VLM decisions and coverage separate. `uncertain` or incomplete coverage cannot prove presence or absence.
- If VLM returns `formatRecovered=true`, use only its explicit `decision`; the malformed visual facts were discarded, so do not quote or invent supporting details.
- Image path or encoding errors mean the search failed, not that the target is absent. Never read raw track files or registry listings.
