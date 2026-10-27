---
name: intent
description: "Identify the requested operation, target and evidence scope before planning."
---

# Identify Intent

- Classify the operation as recognition, total count, registered count, unregistered count or explanation; treat time as scope, not a separate operation.
- Registry intent takes priority over generic count wording: if the user asks how many are registered/in the registry or unregistered/outside it, use only `match_entities`, which matches every trajectory independently. Use `dedup_tracks` for a plain total count or threshold range; a total-only question must not trigger registry matching.
- Keep targets separate and preserve the user's hull number, leading zeros and prefix exactly; tools normalize lookup values.
- Resolve a follow-up from the QA history only when its target, IDs or time range are clear; otherwise ask a focused question.
- Distinguish video presence, vessel identity and registry membership. A registry entry alone does not prove video presence.
- Write supported `requirements` and evidence-backed `plan` steps; do not invent fields such as `restated_question`.
