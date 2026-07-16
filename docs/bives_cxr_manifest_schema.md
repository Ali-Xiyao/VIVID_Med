# BiVES-CXR Manifest Schema

Each JSONL row must contain:

```json
{
  "sample_id": "unique-row-id",
  "patient_id": "patient-or-subject-id",
  "image_path": "relative/or/absolute/path",
  "image_sha256": "lowercase-sha256",
  "study_id": "source-study-id",
  "group_id": "exact-matched-quartet-id",
  "canonical_statement_id": "pleural_effusion|right",
  "statement_text": "A right pleural effusion is present.",
  "state": "support",
  "label_source": "expert|rule|adjudicated",
  "annotation_status": "expert_reviewed"
}
```

Allowed state values:

- `support`
- `contradict`
- `uncertain`
- `insufficient`

Required for `insufficient` rows:

```json
{
  "insufficient_kind": "natural"
}
```

Both `natural` and `synthetic` insufficient examples must be present in each
formal split.

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
7. A canonical statement ID must map to exactly one normalized statement text.
8. The same image/study/hash cannot cross patient-disjoint splits.
9. The same image-statement pair cannot carry conflicting state labels.
10. `group_id` identifies one exact matched S/C/U/I quartet, not an ontology
    statement. It must contain exactly four rows: one support, one contradict,
    one uncertain, and one insufficient row.
11. All four rows in a `group_id` must share one canonical statement ID and
    one normalized statement text. Multiple `group_id` values may reuse the
    same canonical statement.
12. `image_sha256` is verified against file bytes in streaming chunks and is
    cached per resolved path during an audit. A mismatch is a hard failure.
