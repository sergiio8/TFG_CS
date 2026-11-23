# Two-Dimensional Morphogenesis Cases

The `BaseCases2DIM` directory contains the two-dimensional problems used to study more complex cellular interactions and to extend the CBR knowledge base to planar morphogenesis.

## Included problems

### Hungary Flag

Generate three horizontal stripes—red, white, and green—across a two-dimensional canvas.

**Configuration:** ternary state space, von Neumann neighbourhood, and a hybrid boundary scheme with fixed horizontal and periodic vertical boundaries.

### Japan Flag

Generate a centred circle from a single red seed in a white environment.

**Configuration:** binary state space, Moore neighbourhood, and periodic boundaries. The experiment evaluates emergent behaviour across different numbers of time steps.

## Technical features

- **Advanced fitness metrics:** weighted SSIM, state-aware Jaccard similarity, and mutual information.
- **Adaptive mutation:** balances exploration and exploitation by reducing the number of mutated genes as fitness improves.
- **Planar knowledge transfer:** provides challenging cases for testing compatibility and rule transfer in the CBR engine.
