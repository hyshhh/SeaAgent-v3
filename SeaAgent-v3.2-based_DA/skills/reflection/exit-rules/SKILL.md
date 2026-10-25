---
name: exit-rules
description: "Use to decide whether evidence supports termination for this question type."
---

# Exit Rules

- **Recognition:** use the service result's confidence and coverage. `unconfirmed` or incomplete coverage cannot support an absolute absence claim.
- **Registry-assisted recognition:** use it only for a hull/alias supplied by the user; separate a catalogue hit from video evidence.
- **Counts:** use the matching count tool. Report its uncertainty and upper-bound fields when matching or coverage is incomplete.
