---
name: exit-rules
description: "Decide whether this question's evidence supports a conclusion or needs follow-up."
---

# Exit Rules

- **Recognition:** use `status`, candidate `scoreBand` and `coverageComplete`. Absence requires a complete search with no matching candidate; `unconfirmed` or incomplete coverage is not proof.
- **Registry-assisted recognition:** a catalogue hit alone is not evidence that the vessel appears in video.
- **Counts:** use `dedup_tracks` for total-vessel high-/low-threshold counts; use `match_entities` for registered/unregistered track counts, which does not deduplicate. Do not infer registry counts from `dedup_tracks`. Qualify results when coverage is incomplete.
- Stop when the claim is supported or the remaining gap is explicit. Replan only when one distinct tool call can address it.
