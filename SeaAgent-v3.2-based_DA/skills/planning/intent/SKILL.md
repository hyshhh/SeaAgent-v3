---
name: intent
description: "Identify the requested operation, target and evidence scope before planning."
---

# Identify Intent

- Map the six presets to hull-number lookup (`recognize_tracks.hull_number`), description lookup (`recognize_tracks.description`), image lookup (`recognize_tracks.image_path`), total count (`dedup_tracks`), registered count (`match_entities`), or unregistered count (`match_entities`). Treat time as scope, not a separate operation. Handle other explanatory or mixed requests by their actual intent; do not force them into a preset category.
- Description and image lookups ask whether a target appears in video; they are recognition, not registry-membership queries. Registered/unregistered counts match every trajectory independently without deduplication. A plain total count uses `dedup_tracks` and must not trigger registry matching.
- Keep targets separate and preserve the user's hull number, leading zeros and prefix exactly; tools normalize lookup values.
- A hull/alias lookup is a two-stage search when direct recognition is not confirmed: plan `recognize_tracks`, then conditionally `match_registry_tracks` using the same query and time range. Do not make registry assistance depend on an uploaded image.
- Resolve a follow-up from the QA history only when its target, IDs or time range are clear; otherwise ask a focused question.
- Distinguish video presence, vessel identity and registry membership. A registry entry alone does not prove video presence.
- Write supported `requirements` and evidence-backed `plan` steps; do not invent fields such as `restated_question`.
