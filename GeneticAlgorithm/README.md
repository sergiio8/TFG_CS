# Evolutionary Optimisation Framework

The `GeneticAlgorithm` module searches for and refines local transition rules for cellular automata. It connects the physical simulation provided by **CellPyLib** with evolutionary operators from **DEAP**.

## Supported strategies

- **Classical Genetic Algorithm:** tournament selection, two-point crossover, and bit-flip mutation. This is well suited to homogeneous problems such as Density Classification.
- **$(\mu+\lambda)$ Evolution Strategy:** mutation-driven search with strong elitism and adaptive mutation rates. It is used for more complex morphogenesis problems such as the French Flag, Japan, and Hungary.

## Optimisation features

- **Adaptive fitness evaluation:** quickly filters candidates on a small subpopulation before evaluating promising rules in the full environment.
- **Multi-criteria 2D fitness:** combines SSIM, state-wise Jaccard similarity, and mutual information.
- **Knowledge injection:** clones and lightly mutates rules retrieved by the CBR engine to give the initial population a competitive starting point.
- **Exploration-to-exploitation schedule:** progressively reduces mutation as candidates approach convergence.
