# Data Generator

This module produces synthetic datasets for testing entity resolution workflows. It follows a factory‑pattern architecture that cleanly separates the generation of a "ground‑truth" entity space from the creation of noisy observations (datasets A and B).

## Architecture Overview

1. **True Entity Space Construction** – Create a set of \(N\) unique global entities.
2. **Dataset Extraction & Flaw Injection** – For each dataset, sample entities and apply configurable noise (missing values, numerical/Gaussian noise, temporal shifts, etc.) to simulate real‑world data sources.

The core components are:

- **DataTypeGenerator** (abstract base) – Defines how individual data types (categorical, numerical, datetime, string) are generated and corrupted.
- **EntitySpace (Entity Profile Manager)** – Manages the ground‑truth entity mapping and enforces the desired overlap (5‑15%) between datasets.
- **Factory Registry** – Allows plugging in new data‑type generators without touching the core logic.

## Configuration

All generation parameters are driven by YAML files located in `configs/`:
- `base.yaml` – Default settings shared by all dataset sizes.
- `smallest_config.yaml`, `smaller_config.yaml`, `low_card_config.yaml`, `o50_config.yaml`, `smallestest_config.yaml` – Presets for different dataset scales and characteristics.

Each YAML contains sections such as:
```yaml
n_entities: 1000
missing_probability: 0.05
numerical_noise_sigma: 0.1
datetime_noise_sigma_hours: 2
categorical_missing_toggle: true
```
Adjust these values to control volume, noise level, missingness, and cardnality.

## Adding Custom Data Types

To support a new column type (e.g., embeddings, IP addresses, custom strings):

1. **Create a generator class** that inherits from `DataTypeGenerator` (see `entitymanager.py` for the base interface). Implement:
   - `generate_clean(size)` – returns an array of pristine values.
   - `apply_flaws(clean_array, cfg)` – returns the noisy/missing version according to the supplied config.

2. **Register the class** in `entitymanager.py`'s `DATA_TYPE_REGISTRY` dictionary, mapping a string key (used in the YAML) to your class.

3. **Reference the key** in your dataset’s YAML under the `columns` list, e.g.:
   ```yaml
   columns:
     - name: "embedding"
       type: "embedding_generator"   # matches the registry key
       missing_probability: 0.02
   ```

No modifications to the generation orchestration pipelines (`generator.py`, `pipeline.py`) are required.

## Usage

Run the top‑level script to generate all configured datasets:
```bash
python generate_data.py
```
Outputs are written to `data/` as Parquet files (or whatever format is specified in the config).

For programmatic use, import the modules:
```python
from data_generator.entitymanager import EntitySpace
from data_generator.generator import generate_datasets
```
See the docstrings in those files for detailed API information.