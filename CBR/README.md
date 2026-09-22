# Case-Based Reasoning Engine

The `CBR` module is the intelligent core of the project. It orchestrates the four phases of the Case-Based Reasoning lifecycle to configure and accelerate the search for cellular-automaton rules:

1. **Retrieve** similar cases from the knowledge base.
2. **Reuse** their configurations and compatible rules.
3. **Revise** the transferred solution with evolutionary optimisation.
4. **Retain** successful solutions as new learned cases.

## Retrieve

For a target matrix, the engine extracts problem features and computes similarity against the existing cases:

$$
\mathrm{Sim} =
0.2\,\mathrm{Sim}_{topo} +
0.5\,\mathrm{Sim}_{struct} +
0.1\,\mathrm{Sim}_{states} +
0.2\,\mathrm{Sim}_{dims}
$$

## Reuse

Cases above the similarity threshold contribute their configuration to the new problem. Qualitative parameters are inherited from the best match, while quantitative parameters are combined using weighted averages.

When dimensions and state counts are compatible, the previous winning rule is transferred to the initial population. The engine can also compensate for the 90-degree rotation induced by Moore and von Neumann neighbourhood representations.

## Revise and retain

The evolutionary framework refines the transferred population for the target problem. If the resulting rule meets the adaptive fitness threshold, the engine asks the memory module to persist it as a new learned case.
