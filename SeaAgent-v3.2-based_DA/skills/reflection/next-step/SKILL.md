---
name: next-step
description: "Choose one bounded tool action when it can close an evidence gap."
---

# Next Action

Recommend one bounded action only when it can close the gap. In `evidence_gaps`, state the missing evidence and the changed input or filter needed; never retry an unchanged failure.

Reflector cannot call business tools or write `next_action`. Use `decision=replan` to return a concrete gap to Planner; otherwise stop with the uncertainty stated. Do not rebuild service workflows from raw files or records.
