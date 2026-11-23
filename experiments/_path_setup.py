"""Make the source modules importable when experiment scripts run from root."""

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "Configuration",
    "Memory",
    "CBR",
    "GeneticAlgorithm",
    "BaseCases2DIM",
    "Benchmarks1DIM",
)

for source_dir in SOURCE_DIRS:
    source_path = str(ROOT_DIR / "src" / source_dir)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
