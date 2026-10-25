---
name: next-step
description: "Use to turn an unresolved evidence gap into one concrete, bounded action."
---

# Next Action

Name the tool, IDs/filters and expected evidence; reuse prior results and change the failed input.

- Uncertain hull/name recognition: call `match_registry_tracks` with the same value and time range when the direct result requests it.
- Uncertain image or registry candidate: use `verify_vessel_images` with a reference image/reference ID and candidate keyframe ID.
- Incomplete count: report the tool's uncertainty and coverage fields; do not rebuild the pipeline with raw track rows.

Keep the next step to one applicable business tool. Do not read trajectory files or registry listings directly.
