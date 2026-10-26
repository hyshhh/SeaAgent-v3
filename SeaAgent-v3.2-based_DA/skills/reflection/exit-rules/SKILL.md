---
name: exit-rules
description: "Decide whether this question's evidence supports a conclusion or needs follow-up."
---

# Exit Rules

- **Recognition:** use `status`, candidate `scoreBand` and `coverageComplete`. Absence requires a complete search with no matching candidate; `unconfirmed` or incomplete coverage is not proof.
- **Registry-assisted recognition:** a catalogue hit alone is not evidence that the vessel appears in video.
- **Counts:** report the high-/low-threshold range for total entities and the requested registry counts. Qualify results when `coverageComplete` is false.
- Stop when the claim is supported or the remaining gap is explicit. Replan only when one distinct tool call can address it.
