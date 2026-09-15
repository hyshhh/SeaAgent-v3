---
name: planning
description: Use at the start of every turn, before delegating, and again when merging subagent reports.
---

# Sea-Video Planning

1. Restate the question as a claim that tool results can settle, then decide which memory layer can settle it: conversation, trajectory, or video evidence.
2. Resolve every reference before delegating. A follow-up such as "and the ones not in the registry?" must be rewritten into a self-contained task with an absolute time range and the IDs it excludes.
3. Convert explicit time expressions into an absolute range first; never hand a vague scope to a subagent.
4. Delegate by data scope, one delegation per scope. Write down the plan with `write_todos` when a question needs more than one.
5. A subagent sees only the task text you write. Include the scope, the input IDs, and the exact output fields you need.
6. Merge the returned tables yourself, keep their "uncertain" groups uncertain, and never re-delegate the same scope hoping for a cleaner answer.
7. If a report is empty or contradictory, say so in the answer instead of filling the gap with plausible detail.
