---
name: finalize
description: "Use before ending any answer, to land the visual evidence and shape the final Chinese response."
---

# Sea-Video Finalize

1. Before the final answer, call `show_evidence` once with every keyframe, clip and registry reference ID that this turn's tool results produced.
1b. Report the keyframe, clip and registry-reference IDs themselves. The evidence tool also returns a display id (`display-…`) for the panel: that one is not evidence and must not be presented as such.
2. Only use IDs that appear in this turn's tool results. Never invent an ID, and never reuse one from an earlier turn of the conversation.
3. If the turn produced no visual evidence, still call `show_evidence` without arguments, then state the evidence gap in the answer instead of implying proof.
4. After the tool returns, write the final Chinese answer: conclusion first, then the evidence with IDs, then limitations and uncertainty.
5. You decide when the question is answered. Stop once the claim is established or the gap is stated; do not keep calling unrelated tools to fill rounds.
