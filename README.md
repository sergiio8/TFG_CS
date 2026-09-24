# Cellular Automata Rule Optimisation

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-GPL--3.0-green)
![Research](https://img.shields.io/badge/project-Bachelor's%20Thesis-6f42c1)

> A Case-Based Reasoning and evolutionary-optimisation framework for discovering local rules that generate target patterns in cellular automata.

This repository contains the implementation and experimental framework developed for a Bachelor's Thesis. It combines **Case-Based Reasoning (CBR)** with **Evolutionary Algorithms (EA)** to automate the search for cellular-automaton transition rules capable of producing artificial morphogenesis from random initial states or localised seeds.

## Visualise the results

Explore interactive two-dimensional cellular-automaton evolutions in the [Cellular Automata Web Visualiser](https://gaia.fdi.ucm.es/research/cellautomata/).

## Highlights

- **Complete CBR cycle:** retrieve, reuse, revise, and retain previously learned cases.
- **Evolutionary search:** Genetic Algorithms and $(\mu+\lambda)$ Evolution Strategies with adaptive mutation.
- **Knowledge transfer:** inject compatible rules from similar problems into a new evolutionary population.
- **2D evaluation metrics:** combine SSIM, Jaccard similarity, and mutual information to compare generated patterns.
- **Reproducible experiments:** benchmark scripts and predefined one- and two-dimensional case studies.
- **Modular architecture:** separate configuration, memory, CBR, evolutionary, and experiment layers.

## Repository structure

| Directory | Purpose |
| --- | --- |
| [`CBR/`](CBR/README.md) | Retrieve, reuse, revise, and retain logic |
| [`Configuration/`](Configuration/README.md) | Typed case and experiment configuration |
| [`GeneticAlgorithm/`](GeneticAlgorithm/README.md) | Evolutionary optimisation strategies |
| [`Memory/`](Memory/README.md) | Base-case loading and learned-case persistence |
| [`Benchmarks1DIM/`](Benchmarks1DIM/README.md) | One-dimensional benchmark problems |
| [`BaseCases2DIM/`](BaseCases2DIM/README.md) | Two-dimensional morphogenesis problems |
| `main.py` | Proposed CBR-based approach |
| `main_heuristica.py` | Expert-configured baseline |
| `main_random.py` | Random-search baseline |
| `run_experiments.py` | Reproducible multi-seed statistical runner |

## Requirements

- Python 3.12 or newer
- The dependencies listed in [`requirements.txt`](requirements.txt)

Install the project locally:

```bash
git clone https://github.com/sergiio8/TFG_CS.git
cd TFG_CS
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the experiments

### Proposed CBR approach

```bash
python main.py
```

This entry point retrieves similar cases, transfers their knowledge, and refines the resulting rule with the evolutionary optimiser. Configure the target problem in `main.py` before running a new experiment.

### Expert-configured baseline

```bash
python main_heuristica.py
```

Runs experiments using manually selected configurations based on prior domain knowledge.

### Random baseline

```bash
python main_random.py
```

Runs control experiments with valid parameters selected randomly.

### Statistical evaluation

The repository also includes `run_experiments.py` for reproducible batch
experiments. It runs five seeds per case and keeps the experimental CBR memory
fixed during the batch, so learned cases don't leak from one run into another.
The generated data is written to CSV files:

```bash
python run_experiments.py
```

The runner uses a separate `casos_aprendidos_backup.pkl` file and restricts the
CBR retrieval memory to the fixed Japan and Hungary base cases. It produces
`resultados_experimentos_nuevos_casos.csv` and `resumen_estadistico.csv`.
These generated files are intentionally kept outside the source tree until the
corresponding experiment has been run.

## Research context

The framework studies how previously solved cellular-automaton problems can accelerate the discovery of rules for new problems. The included cases cover density classification, stripe formation, embryonic patterning, and geometric morphogenesis.

For the academic background and experimental results, see [`PaperCAEPIASergioGit.pdf`](PaperCAEPIASergioGit.pdf).

## License

This project is distributed under the [GNU General Public License v3.0](LICENSE).
