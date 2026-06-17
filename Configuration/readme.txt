# Configuration and Data Structure Module (CBR Case)

This directory defines the strongly typed data structures and global configuration schemas of the system. It utilizes Python `dataclasses` to ensure consistent, scalable, and easily serializable data handling across the CBR engine, the evolutionary framework, and the memory module.

---

## Structure of a CBR Case

Each instance is formally modeled following the problem's mathematical tuple, decoupled into the following objects:

- **`CBRCase`:** The root container. It links a unique identifier (`id`) with the problem features and its associated solution.
- **`ProblemFeatures`:** Topological and structural descriptor of the target state (image matrix, spatial dimensions, number of states, presence of stripes, connected components, and border uniformity).
- **`CBRSolution`:** Stores the framework configuration used, the best local rule found ($\varphi^*$), and the final fitness score achieved ($f^*$).
- **`FrameworkConfig`:** A master class that unifies both execution environments:
    - **`CAConfig`:** Cellular Automata parameters in *CellPyLib* (size, Moore/von Neumann neighborhood, boundary handling, and time steps).
    - **`GAConfig`:** Evolutionary Algorithm hyperparameters in *DEAP* (population size, generations, selection methods, genetic operators, and similarity metric weights).

---

## Design Features

- **Automated Validation (`__post_init__`):** Dynamically computes runtime system properties, such as the chromosome size (`ind_size`) based on the chosen neighborhood ($n^5$ for von Neumann and $n^9$ for Moore).
- **Complete Modularity:** Acts as the single data contract for the repository, ensuring that any changes in parameterization automatically propagate throughout the entire TFG pipeline.
