# Entity Resolution Demo

This repository contains a modular entity resolution pipeline built with Splink, along with utilities for synthetic data generation, aggregation, and helper functions.

## Project Structure

- `data_generator/` – Synthetic data generation utilities (see [data_generator/README.md](./data_generator/README.md))
- `helpers/` – Helper packages for entity resolution pipelines:
  - [`em_pipeline/`](./helpers/em_pipeline/README.md) – Modular Splink pipeline with factory pattern
  - [`aggregation/`](./helpers/aggregation/README.md) – Behavioural aggregation framework
  - `entity_resolution_aggregation.py` – Aggregation entry point
  - `visualization_helper.py` – Plotting utilities
- `notebooks/` – Example notebooks (e.g., `splink_local_demo.ipynb`)
- `public_data/` – Example public datasets (ignored in git)
- `latex/` – Report templates (ignored in git)

## Quick Start

1. **Generate synthetic data** (optional):  
   ```bash
   python generate_data.py
   ```
   Configure via `data_generator/configs/*.yaml`.

2. **Run the demo notebook**:  
   Open `notebooks/splink_local_demo.ipynb` and execute the cells.  
   The notebook uses the helper functions in `helpers/` to load data, configure the Splink pipeline, run training/prediction, and produce visualisations.

3. **Add your own data**:  
   Place data files (Parquet/CSV) in `public_data/` or any directory and update the configuration YAMLs in `data_generator/configs/` or the helper factory registries.

## Configuration

All major components are configured via YAML files:
- Data generation: `data_generator/configs/`
- Helper factories: look for `registry.py` files in each helper subpackage (e.g., `helpers/em_pipeline/registry.py`)

The system uses a factory pattern; to add custom functionality (new data types, comparisons, blocking rules, aggregation processors) you only need to:
1. Implement a class that adheres to the appropriate base interface.
2. Register it in the corresponding registry.
No changes to core logic are required.

## Further Reading

- [Data Generator Guide](./data_generator/README.md)
- [EM Pipeline (Splink Wrapper) Guide](./helpers/em_pipeline/README.md)
- [Aggregation Framework Guide](./helpers/aggregation/README.md)