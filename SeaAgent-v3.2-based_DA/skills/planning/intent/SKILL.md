---
name: intent
description: "Identify the requested operation, target and evidence scope before planning."
---

# Identify Intent

- Map the six presets to hull-number lookup, description lookup and image lookup (`recognize_tracks`), total count (`dedup_tracks`), registered count (`match_entities`) or unregistered count (`match_entities`). The Harness already binds the video/index; never ask for a video path for these presets. Treat time as scope, not a separate operation.
- Description and image lookups ask whether a target appears in video; they are recognition, not registry-membership queries. Registered/unregistered counts match every trajectory independently without deduplication. `match_entities` may return `nextStep=verify_vessel_images` for gray-zone candidates only. A total count uses two-stage `dedup_tracks`; only exact-match gray-zone duplicate pairs require VLM confirmation, but it must not trigger registry matching.
- Keep targets separate and preserve the user's hull number, leading zeros and prefix exactly; tools normalize lookup values.
- A hull/alias lookup is a two-stage search when direct recognition is not confirmed: plan `recognize_tracks`, then conditionally `match_registry_tracks` using the same query and time range. Do not make registry assistance depend on an uploaded image.
- Resolve a follow-up from the QA history only when its target, IDs or time range are clear; otherwise ask a focused question. For count-only questions with no time condition, use all stored trajectories and omit `time_range`.
- Distinguish video presence, vessel identity and registry membership. A registry entry alone does not prove video presence.
- Write supported `requirements` and evidence-backed `plan` steps; do not invent fields such as `restated_question`.
