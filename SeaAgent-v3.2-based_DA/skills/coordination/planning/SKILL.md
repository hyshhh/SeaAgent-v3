---
name: planning
description: "Use when starting or revising a plan, especially after new evidence."
---

# Shared Planning

- Resolve follow-ups from this conversation's QA record; preserve the referenced time scope and IDs.
- Use real state fields: `requirements`, `plan`, `next_action`, `clarification`; each step has `id`, `content`, `required_evidence`.
- No time condition means all stored monitoring time. Explicit time must become an absolute, timezone-aware range.
- Empty/failed results require a changed input or an explicit gap, never an identical retry.
- Planner writes plans; Executor records progress; Reflector judges evidence and termination.
