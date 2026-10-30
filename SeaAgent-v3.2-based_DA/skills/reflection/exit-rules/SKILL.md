---
name: exit-rules
description: "Decide whether this question's evidence supports a conclusion or needs follow-up."
---

# Exit Rules

- **Recognition:** use `status`, candidate `scoreBand`, `coverageComplete`, `nextStep` and `recommendation`. Complete each requested helper step before terminating. Absence requires a complete direct and applicable assisted search with no match; `unconfirmed`, VLM `uncertain` or incomplete coverage is not proof.
- **Registry-assisted recognition:** a catalogue hit alone is not evidence that the vessel appears in video.
- **Counts:** for total vessels, complete exact-match gray-zone VLM checks and finalize `dedup_tracks`; high-threshold pairs are already exact. Report `exactShipCount` and `fuzzyShipCount` and their range. For registered/unregistered counts, use `embeddingConfirmedCount` and `outsideRegistryCount` directly, then verify gray candidates only. Qualify results when coverage is incomplete or any candidate remains unresolved.
- Stop when the claim is supported or the remaining gap is explicit. Replan only when one distinct tool call can address it.
