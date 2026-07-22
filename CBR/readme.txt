# CBR Core Engine Module

This directory houses the intelligent core of the system. Its function is to orchestrate and execute the Case-Based Reasoning lifecycle (Retrieve, Reuse, Revise, and Retain) to automate configuration and accelerate the optimization of cellular automata rules.

---

## Phase Functionalities

### 1. Retrieve
Given a new problem (target matrix), it extracts its features and computes its similarity against the knowledge base using an expert function:
$$Sim = 0.2 \cdot Sim_{topo} + 0.5 \cdot Sim_{struct} + 0.1 \cdot Sim_{states} + 0.2 \cdot Sim_{dims}$$

### 2. Reuse
It adapts the parameters of the retrieved cases that exceed the threshold ($0.5$). Qualitative parameters are inherited from the best case, and quantitative parameters are calculated using weighted averages.
* **Transfer Learning:** If the dimensions and states match (or after geometrically reversing the 90° rotation in the Moore/von Neumann neighborhoods), it **injects the previous winning rule** into the initial population of the evolutionary algorithm (5% clones and 5% subtly mutated).

### 3. Revise
It triggers the evolution framework using the optimized initial population from the previous phase, refining and adapting the local rule for the specific new problem much faster than a from-scratch execution.

### 4. Retain
It evaluates the success of the new rule using an adaptive fitness threshold (which becomes more demanding as the database grows). If it is suitable, it requests its permanent storage as a new learned case from the memory module.
