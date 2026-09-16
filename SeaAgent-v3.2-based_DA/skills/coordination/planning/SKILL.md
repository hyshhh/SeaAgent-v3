---
name: planning
description: Use at the start of every turn, before the first domain tool call, and again whenever a result forces a change of plan.
---

# Sea-Video Planning

1. Restate the question as a claim that tool results can settle, then decide which memory layer can settle it: conversation, trajectory, or video evidence.
2. Resolve every reference before you touch data. A follow-up such as "and the ones not in the registry?" must be rewritten into a self-contained claim with an absolute time range and the IDs it excludes.
3. When the user says "刚才 / 上一轮 / a moment ago", do not guess a window: quote the previous turn's time range and IDs verbatim from the conversation (the summary keeps them). If the earlier range is genuinely gone, say so and ask — never silently substitute a new one.
4. Convert explicit time expressions into an absolute range first; never start querying with a vague scope.
5. Write the plan with `write_todos` as soon as a question needs more than one step, and keep that list current: mark a step done when its result is in hand, and add the step a result just made necessary.
6. Work one todo at a time. A step that returns nothing usable means the step's inputs were wrong — change the filter, the window or the ID set rather than repeating the same call.
7. A filtered query that returns empty is not evidence of absence. Say which filter was used and what an unfiltered scan would have covered.
8. If a result is empty or contradictory, say so in the answer instead of filling the gap with plausible detail.
9. Skill paths come from this turn's skill list; after a reorganisation an old path from earlier in the conversation will be refused.
