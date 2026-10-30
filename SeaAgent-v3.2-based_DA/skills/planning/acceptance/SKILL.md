---
name: acceptance
description: "Map each requirement to the tool result that can verify it."
---

# Plan Acceptance

Put each verifiable claim in `requirements`. Every plan step needs `id`, `content` and a string-array `required_evidence`; do not add unsupported fields.

- For the six presets, plan distinctly: hull number, description and image lookup use `recognize_tracks` with the matching input; total count uses `dedup_tracks`; registered and unregistered counts use `match_entities`. Handle other or mixed requests by their explicit requirements. Hull/alias plans include a conditional `match_registry_tracks` step; run it when direct recognition is not confirmed and returns that `nextStep` or recommendation. Count-only plans must not include recognition steps.
- Plan the recognition chain explicitly: hull/alias search → conditional `match_registry_tracks` → conditional VLM verification; description/image search → conditional VLM verification. Add the verification step when the corresponding result returns `nextStep=verify_vessel_images`; skip it only when no verifiable uncertain candidate exists.
- Plan the registry-count chain explicitly: `match_entities` → conditional gray-zone `verify_vessel_images`. Verify only returned gray `verificationCandidates`; embedding-confirmed and below-exclusion tracks need no VLM call.
- Plan the total-count chain explicitly: first `dedup_tracks` → conditional duplicate-pair `verify_vessel_images` → final `dedup_tracks` with confirmed/resolved pair IDs. The final tool returns exact and fuzzy entity counts.
- Pair VLM evidence by query type: original user description + candidate `matchedKeyframeId`, original uploaded image + candidate `matchedKeyframeId`, or matched registry reference + candidate `matchedKeyframeId`.
- Use `match_entities` for registered or unregistered counts, matching every track without deduplication. Plain total-vessel or threshold-range counts use `dedup_tracks`; a total-only request must not include registry matching. Combine both only when the user explicitly asks for total and registry counts. Never invent tool names.
- Add `show_evidence` only when visual evidence must be displayed, using IDs returned by business tools.
