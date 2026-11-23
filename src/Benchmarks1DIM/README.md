# One-Dimensional Benchmark Problems

The `Benchmarks1DIM` directory contains the one-dimensional problems used to validate the first version of the evolutionary framework and to provide static base cases for the CBR memory.

## Included problems

### Density Classification Problem

Find a deterministic local rule that classifies the initial density of an automaton. If more than half of the cells contain ones, the system should converge to all ones; otherwise, it should converge to all zeros.

**Configuration:** binary state space, seven-cell neighbourhood, and periodic boundaries.

### French Flag Problem

Starting from a random configuration, the automaton must self-organise into three equal stripes: blue, white, and red.

**Configuration:** ternary state space, three-cell neighbourhood, and fixed boundaries.

### Drosophila Embryogenesis

A bio-inspired problem based on early fruit-fly development. Local interactions must stabilise a periodic pattern of seven transverse stripes from an initially noisy state.

**Configuration:** binary state space and a seven-cell neighbourhood.

Each case exports its optimised `CAConfig` and `GAConfig` for use as bootstrapping knowledge by the CBR engine.
