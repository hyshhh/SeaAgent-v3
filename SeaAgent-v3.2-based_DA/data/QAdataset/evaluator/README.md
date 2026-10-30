# SeaQA Evaluator

This evaluator is dependency-free and evaluates a prediction file against the
server-generated SeaQA subset.

## Input Files

`--dataset` expects the server-only file:

```text
data/QAdataset/dataset/questions.jsonl
```

Each line is one question with:

- `id`: stable question identifier;
- `category`: one of the six SeaQA categories;
- `question` and `time_range`;
- `inputs`: hull number, description, or image path;
- `ground_truth.entities`: the target entity set;
- `ground_truth.unresolved_entities`: relations that cannot be determined;
- `provenance`: source track, keyframe, image, and registry evidence.

## Prediction Format

Provide one JSON object per line:

```json
{
  "id": "SeaQA-0001",
  "status": "answered",
  "answer": "appears",
  "entities": [
    {
      "entity_id": "entity-000002",
      "confidence": 0.91
    }
  ]
}
```

Fields:

- `status`: `answered`, `unresolved`, or `error`;
- `answer`: the model answer; count questions should return an integer;
- `entities`: ranked entity candidates. Higher confidence must be placed first
  or supplied through the `confidence` field.

For an unresolved question, use:

```json
{"id": "SeaQA-0012", "status": "unresolved", "entities": []}
```

## Run

From `SeaAgent-v3.2-based_DA`:

```bash
python data/QAdataset/evaluator/evaluate.py \
  --dataset data/QAdataset/dataset/questions.jsonl \
  --predictions /path/to/predictions.jsonl \
  --output /path/to/evaluation.json
```

Omit `--output` to print the result to standard output.

## Metrics

Recognition categories use entity-set Exact Match. Entity order and duplicate
returns do not affect EM.

Count categories use the paper's entity-set AP/mAP protocol. The first valid hit
for each truth entity is positive; duplicate, incorrect, or unsupported entity
returns do not earn another hit. Exact-count accuracy is reported separately.

Questions whose ground truth is explicitly unresolved are excluded from the
determinate EM/mAP denominator and reported through `unresolved_accuracy`.

The JSON result contains:

- per-category `count`, `scoreable`, and `unresolved`;
- recognition `em`;
- count `map` and `count_accuracy`;
- unresolved-question accuracy;
- per-question scoring rows for error analysis.

## Provenance

Every generated question includes source track IDs and keyframe IDs. Image
queries additionally include the copied image path, original server path, and
SHA-256 checksum. Registry questions include the registry entry and reference
image used for VLM verification.
