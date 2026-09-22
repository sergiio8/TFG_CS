# Configuration and Case Data Structures

The `Configuration` module defines the typed data structures and global schemas shared by the CBR engine, evolutionary optimiser, and memory layer. Python `dataclasses` provide consistent, serialisable configuration objects across the pipeline.

## CBR case model

- **`CBRCase`** — root container linking an identifier to problem features and a solution.
- **`ProblemFeatures`** — descriptors of the target state, including its matrix, dimensions, number of states, stripes, connected components, and border uniformity.
- **`CBRSolution`** — framework configuration, best local rule $(\varphi^*)$, and final fitness $(f^*)$.
- **`FrameworkConfig`** — combines cellular-automaton and evolutionary settings.
- **`CAConfig`** — CellPyLib parameters such as size, neighbourhood, boundary handling, and time steps.
- **`GAConfig`** — DEAP parameters such as population size, generations, selection, genetic operators, and metric weights.

## Design features

- **Automatic validation:** `__post_init__` derives runtime properties such as chromosome size from the selected neighbourhood.
- **Single data contract:** changes to a configuration propagate consistently through the complete thesis pipeline.
- **Serialisable structures:** cases can be persisted and restored by the memory module.
