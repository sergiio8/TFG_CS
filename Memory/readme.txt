# Memory and Persistence Module (CBR Memory)

This directory manages the storage and persistence of cases within the CBR system, divided into two categories:

- **Base Cases:** Static initial knowledge calibrated by an expert (*Density Classification, French Flag, Hungary, and Japan*).
- **Learned Cases:** New solutions optimized by the evolutionary algorithm that are stored incrementally.

---

## Key Components

- **Memory Class:** The orchestrator responsible for loading base cases at startup, retrieving historical cases, and saving new solutions to disk.
- **Serialization (`pickle`):** Binary storage that preserves the exact structure of Python objects (such as `numpy` matrices for target states and rule vectors), guaranteeing ultra-fast read and write operations.

---

## Workflow

1. **Retrieve:** The core engine requests the collection of cases from this module to compute similarities.
2. **Retain:** If a new rule surpasses the adaptive fitness threshold, this module assigns it a unique identifier (`id`) and permanently serializes it.
