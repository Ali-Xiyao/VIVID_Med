# Attention Diagnostic Split Implementation Failure

The first CPU-only attention-group analysis stopped before loading any image:

```text
ValueError: locked attention diagnostic row count unavailable
```

The frozen hard-UMS manifest names its internal validation split `validate`;
the new read-only diagnostic used `validation`. Training, checkpoints, S3
predictions, and the already-frozen terminal verdict were unaffected.

The single identity-preserving repair changes the literal split name to the
manifest's existing `validate` value in both the descriptive lock and reader.
The same lexicographically first 128 rows, transforms, metrics, and checkpoint
hashes remain fixed. The analysis reruns from an empty output surface.
