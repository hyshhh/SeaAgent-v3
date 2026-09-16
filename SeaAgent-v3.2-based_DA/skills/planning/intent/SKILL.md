---
name: intent
description: Use at the start of every question to turn the request into one self-contained claim.
---

# Sea-Video Intent

1. Restate the question as a claim tool results can settle, then fix the operation: existence / count / list / time / explain.
2. Fix the target: a hull number when one is readable, otherwise an appearance description. Keep a Chinese prefix on a hull number intact ("小蓝320"); normalised form is upper-cased.
3. Several vessels joined by 、，和/与/及/以及 are several targets — list them separately, never merge into one string.
4. Resolve references before starting: "and the ones not in the registry?" becomes a self-contained claim carrying the previous turn's absolute range and the IDs it excludes.
5. Scope words (哪些/有哪些/在库/未在库/先验库/库里) belong to the scope and relation, never to the appearance description. "库里有哪些黄色快艇" leaves 黄色快艇 as the description.
6. "有没有/是否出现" over a hull number asks about the footage; "库里有吗" asks about the database. Scope is `both` only when both sides are named — "有哪些在库船出现在视频中" is `both`.
7. Write the claim, operation and scope into `restated_question` so the executor needs no further guessing.
