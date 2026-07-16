# BiVES-CXR Manifest Schema

Each JSONL row must contain:

```json
{
  "sample_id": "unique-row-id",
  "patient_id": "patient-or-subject-id",
  "image_path": "relative/or/absolute/path",
  "canonical_statement_id": "pleural_effusion|right",
  "statement_text": "A right pleural effusion is present.",
  "state": "support"
}
```

Allowed state values:

- `support`
- `contradict`
- `uncertain`
- `insufficient`

Recommended additional fields:

```json
{
  "finding": "pleural_effusion",
  "location": "pleural",
  "laterality": "right",
  "severity": null,
  "view": "PA",
  "source_dataset": "CheXpert",
  "evidence_requirement": "single_frontal",
  "insufficient_reason": null,
  "annotation_status": "expert_reviewed",
  "split": "train"
}
```

Rules:

1. Split patients before constructing image-statement pairs.
2. Use one non-negated canonical proposition across all four states.
3. Report omission is not a contradict label.
4. Natural insufficient samples must be distinguishable from synthetic
   deletion interventions.
5. Image loading is fail-fast. Missing/corrupt files are data errors, not
   insufficient examples.
6. Horizontal flips are disabled unless laterality text and annotations are
   changed consistently.
