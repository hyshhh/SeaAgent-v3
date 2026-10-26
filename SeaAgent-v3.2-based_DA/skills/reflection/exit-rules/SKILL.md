---
name: exit-rules
description: "Use to decide whether evidence supports termination for this question type."
---

# Exit Rules

- **Recognition:** use `status`, candidate `scoreBand` and `coverageComplete`; `unconfirmed` or incomplete coverage cannot support an absolute absence claim.
- **Registry-assisted recognition:** use it only for a hull/alias supplied by the user; separate a catalogue hit from video evidence.
- **Counts:** use the matching count tool. Qualify the count when `countComplete=false`, stability is not `stable`, or uncertain items remain.
