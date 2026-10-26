---
name: intent
description: "Identify the requested operation, target and evidence scope before planning."
---

# Identify Intent

- Classify the operation as recognition, total count, registered count, unregistered count or explanation; treat time as scope, not a separate operation.
- Keep targets separate and preserve the user's hull number, leading zeros and prefix exactly; tools normalize lookup values.
- Resolve a follow-up from the QA history only when its target, IDs or time range are clear; otherwise ask a focused question.
- Distinguish video presence, vessel identity and registry membership. A registry entry alone does not prove video presence.
- Write supported `requirements` and evidence-backed `plan` steps; do not invent fields such as `restated_question`.
