---
name: exit-rules
description: "Decide whether this question's evidence supports a conclusion or needs follow-up."
---

# Exit Rules

- **Recognition:** use `status`, candidate `scoreBand`, `coverageComplete`, `nextStep` and `recommendation`. Complete each requested helper step before terminating. Absence requires a complete direct and applicable assisted search with no match; `unconfirmed`, VLM `uncertain` or incomplete coverage is not proof.
- **Registry-assisted recognition:** a catalogue hit alone is not evidence that the vessel appears in video.
- **Counts:** use `dedup_tracks` for total-vessel high-/low-threshold counts. For registered/unregistered counts, use `embeddingConfirmedCount` and `outsideRegistryCount` directly, then verify gray candidates only and classify VLM `match` as registered and VLM `mismatch` as unregistered. Qualify results when embedding coverage is incomplete or any candidate remains unresolved.
- Stop when the claim is supported or the remaining gap is explicit. Replan only when one distinct tool call can address it.
