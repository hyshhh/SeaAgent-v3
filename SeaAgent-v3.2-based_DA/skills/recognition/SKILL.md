---
name: recognition
description: "Use the compact recognition tools for track, registry-assisted and two-image identification."
---

# Recognition

- Use `recognize_tracks.hull_number` for a supplied hull number or target name/alias (for example, “小红”); use `description` for appearance and `image_path` for an image. Omit `time_range` unless the user supplied a range.
- If the result sets `nextStep=match_registry_tracks`, or a hull/alias query returns `unconfirmed` or `mismatch`, call `match_registry_tracks` once with the same value and time range.
- Use `verify_vessel_images` for uncertain visual candidates: pass the user's image as `reference_image` or a returned `reference_registry_reference_id`, then a candidate image path or `candidate_keyframe_id`.
- Treat `confirmed`, `unconfirmed` and `mismatch` as separate outcomes. `mismatch` means no usable candidate; it is not evidence that every track was checked unless `searchCoverageComplete` is true.
- Keep the returned track, keyframe and registry-reference IDs for the evidence tool. Do not request raw track pages or registry listings.
