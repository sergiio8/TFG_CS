# 2-Dimensional Case Studies (2D Base Cases)

This directory contains the modeling, configuration, and experimental scripts for the problems implemented in two dimensions. These scenarios evaluate more complex cellular interaction dynamics and serve as an advanced knowledge base for the CBR engine in planar morphogenesis problems.

---

## Implemented Problems

### 1. Hungary Flag (Horizontal Pattern Formation)
An artificial morphogenesis problem that requires the automaton to segment the two-dimensional canvas into three perfect horizontal stripes (red, white, and green). 
- **Configuration:** Ternary space (3 states), von Neumann neighborhood, and a hybrid boundary scheme (fixed horizontally to separate the red/green and periodic vertically to maintain continuity).

### 2. Japan Flag (Geometric Morphogenesis from Seed)
A challenge that introduces a radically new problem typology: generating a precise geometric shape (a centered circle) from a single initial red cell (central seed) in a white environment.
- **Configuration:** Binary space, Moore neighborhood (indispensable for processing diagonals and smoothing curvature), and periodic boundaries. It evaluates the temporal emergent behavior of the automaton against variations in time steps (*timesteps*).

---

## Technical Innovations of the 2D Environment

- **Advanced Fitness Metrics:** Performance evaluation in these cases required abandoning linear accuracy in favor of a weighted combination of the multi-channel Structural Similarity Index (SSIM), state-importance weighted Jaccard Index, and Mutual Information.
- **Dynamic Adaptive Mutation:** Implements a strict exploration vs. exploitation regime that step-down reduces the number of mutated genes per chromosome as fitness surpasses critical convergence thresholds.
