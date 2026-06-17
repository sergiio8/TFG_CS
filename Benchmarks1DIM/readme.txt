# 1-Dimensional Case Studies (1D Base Cases)

This directory contains the modeling, configuration, and experimental scripts for the problems implemented in a single dimension. These scenarios served to validate the first version of the evolutionary framework and act as static base cases within the CBR system's memory.

---

## Implemented Problems

### 1. Density Classification Problem (DCP)
A classic benchmark whose objective is to find a deterministic local rule that classifies the initial density of the automaton. If the number of ones exceeds 50%, the entire system must converge to a uniform state of ones; otherwise, to all zeros.
- **Configuration:** Binary space, 7-cell neighborhood, and periodic (circular) boundaries.

### 2. French Flag Problem (1D)
A problem that conceptualizes low-level biological morphogenesis. The automaton must self-organize from a completely random initial state until it stabilizes an axial pattern divided into three perfect stripes of equal size (blue, white, and red).
- **Configuration:** Ternary space (3 states), neighborhood of size 3, and fixed (constant) boundaries to guide symmetry.

### 3. Drosophila Embryogenesis
A bio-inspired simulation based on the early development of the fruit fly. Starting from an initial phase of instability and noise (gene expression), the cells must communicate locally until they stabilize a periodic pattern of seven transverse stripes.
- **Configuration:** Binary space and an extended neighborhood of 3 neighbors on each side (7 cells in total).

---

## Utility in the System

Each of these files exports its respective optimized `CAConfig` and `GAConfig`. The CBR engine uses them as bootstrapping knowledge to transfer search hyperparameters and linear packaging rules to new problems with similar growth or classification dynamics.
