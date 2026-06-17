# Evolutionary Framework Module

This directory houses the optimization core responsible for searching and refining the local transition rules of Cellular Automata (CA) using Evolutionary Computation. 

It directly connects the physical simulation of **CellPyLib** with the search algorithms from **DEAP**.

---

## Supported Strategies

The module is designed as a hybrid system and switches between two approaches depending on the problem configuration:

- **Classical Genetic Algorithm (GA):** Implements the standard cycle with tournament selection, two-point crossover, and bit-flip mutation. Ideal for homogeneous problems like *Density Classification*.
- **$(\mu + \lambda)\text{-ES}$ Evolutionary Strategy:** Operates exclusively through mutation operators and intense elitism-based selective pressure. It includes dynamic mutation rates (exploration vs. exploitation) that are critical for solving complex morphogenesis problems (*French Flag*, Japan, Hungary).

---

## Special Features

- **Adaptive Fitness:** Executes rapid filtering by evaluating rules on a small subpopulation of automata; if they surpass the initial threshold, it performs a full evaluation in the actual environment, saving up to 80% in computational cost.
- **2D Multi-Criteria Metrics:** Evaluates the quality of the generated image by simultaneously combining the Structural Similarity Index (SSIM), state-wise Jaccard Index, and Mutual Information.
- **Knowledge Injection (Transfer Learning):** Capable of receiving external rules (retrieved by the CBR engine), cloning them, and slightly mutating them to initialize the population with a competitive advantage.
