# SeaQA Evaluation

This directory is isolated from the serving harness. It contains:

- `evaluator/`: the user-facing evaluation program and usage guide.
- `dataset/`: the server-side generated SeaQA subset and supporting assets.
- `collector/`: an internal maintenance script used only to derive the dataset from server-side memory.

The real `questions.jsonl`, query images, feature caches, and manifest are
generated only on the server. They are excluded from Git.

Users should start with [`evaluator/README.md`](evaluator/README.md).

## Dataset Protocol

The benchmark follows the SeaQA protocol described in the paper:

- recognition: hull number, description, and image lookup;
- statistics: total, registered, and unregistered entity counts;
- ground truth: an entity set with track and keyframe provenance;
- unresolved registry relations remain unresolved and are evaluated separately;
- recognition uses entity-set Exact Match;
- statistics use entity-set AP/mAP and exact-count accuracy.

`collector/collect.py` is not part of the user workflow. It reads server-only
trajectory and registry files, performs cached VLM verification, and regenerates
the deterministic 100-question subset.
