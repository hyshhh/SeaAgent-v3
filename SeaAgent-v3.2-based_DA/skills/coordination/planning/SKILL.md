---
name: planning
description: Use at the start of every turn, before delegating, and again when merging subagent reports.
---

# Sea-Video Planning

1. Restate the question as a claim that tool results can settle, then decide which memory layer can settle it: conversation, trajectory, or video evidence.
2. Resolve every reference before delegating. A follow-up such as "and the ones not in the registry?" must be rewritten into a self-contained task with an absolute time range and the IDs it excludes.
3. When the user says "刚才 / 上一轮 / a moment ago", do not guess a window: quote the previous turn's time range and IDs verbatim from the conversation (the summary keeps them), and state that window inside the task description. If the earlier range is genuinely gone, say so and ask — never silently substitute a new one.
4. Convert explicit time expressions into an absolute range first; never hand a vague scope to a subagent.
5. Delegate by data scope, one delegation per scope. Write down the plan with `write_todos` when a question needs more than one.
6. A subagent sees only the task text you write. Include the scope, the input IDs, and the exact output fields you need.
7. Merge the returned tables yourself, keep their "uncertain" groups uncertain, and never re-delegate the same scope hoping for a cleaner answer. An empty result from a filtered query means the filter is wrong, not that the answer is empty.
8. If a report is empty or contradictory, say so in the answer instead of filling the gap with plausible detail.
9. You cannot read the vessel registry or trajectory memory yourself — they live behind the subagents. Do not go looking through directories for data.
10. Skill paths come from this turn's skill list; after a reorganisation an old path from earlier in the conversation will be refused.
