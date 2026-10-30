---
name: next-step
description: "Choose one bounded tool action when it can close an evidence gap."
---

# Next Action

Recommend one bounded action only when it can close the gap. For recognition, registry-count and dedup tasks, execute the tool-declared `nextStep` before considering unrelated retries. In `evidence_gaps`, state the missing evidence and the changed input or filter needed; never retry an unchanged failure. If the remaining gap requires the same failed call, write `assessment.final_answer` and terminate with the item marked unresolved.

Reflector cannot call business tools or write `next_action`. Use `decision=replan` to return a concrete gap to Planner; otherwise stop with the uncertainty stated. Do not rebuild service workflows from raw files or records.
