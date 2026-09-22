# Case Memory and Persistence

The `Memory` module stores and retrieves the cases used by the CBR engine.

## Case collections

- **Base cases:** expert-calibrated problems covering Density Classification, the French Flag, Drosophila embryogenesis, Hungary, and Japan.
- **Learned cases:** new solutions discovered by the evolutionary optimiser and retained incrementally.

## Main responsibilities

- Load base cases when the system starts.
- Provide historical cases to the retrieval phase.
- Assign identifiers to successful new cases.
- Persist and restore cases using `pickle`, preserving NumPy matrices and rule vectors without lossy conversion.

The statistical runner can use a clean, fixed experimental memory containing
the selected base cases. This is separate from the normal CBR memory and does
not prevent the application from loading all configured base cases, including
France and Drosophila.

## Workflow

1. **Retrieve:** the CBR engine requests the available cases and computes similarities.
2. **Retain:** solutions that exceed the adaptive fitness threshold receive an identifier and are serialised for future experiments.
