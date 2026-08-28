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

## Extending and Customizing

### Adding New Data Config Files

The `notebooks/notebook_helper.py` module provides automatic dataset discovery and generation:

1. **Add a new config file** to `data_generator/configs/` (e.g., `my_dataset_config.yaml`)
2. **Update `notebooks/notebook_config.yaml`** to map a friendly name to your dataset:
   ```yaml
   dataset_mapping:
     # ... existing entries ...
     my_dataset:
       a: "my_dataset_a"
       b: "my_dataset_b"
   ```
3. **Use in notebooks** with the helper functions:

   ```python
   # In your notebook setup
   from notebooks.notebook_helper import setup_notebook_environment
   helper = setup_notebook_environment()

   # This will automatically generate the dataset if missing
   path_a, path_b = helper.ensure_dataset_exists("my_dataset")
   ```

4. **The helper will automatically**:
   - Check if `data/my_dataset_a.parquet` and `data/my_dataset_b.parquet` exist
   - If not found, call `generate_data.py` to create them using your config
   - Return the paths to the generated/found files

### Adding New Data Types to Data Config

To support a new column type (e.g., embeddings, IP addresses, custom strings) in synthetic data generation:

1. **Create a generator class** in `data_generator/entitymanager.py` that inherits from `DataTypeGenerator`:

   ```python
   from data_generator.entitymanager import DataTypeGenerator
   import numpy as np

   class EmbeddingGenerator(DataTypeGenerator):
       def generate_clean(self, size):
           # Generate pristine embedding vectors (e.g., 128-dim)
           return np.random.rand(size, 128).astype(np.float32)

       def apply_noise(self, clean_array, cfg):
           # Add noise/missing values to embeddings
           # Example: randomly zero out some dimensions
           if cfg.get('missing_probability', 0) > 0:
               mask = np.random.random(clean_array.shape) < cfg['missing_probability']
               clean_array[mask] = 0.0
           return clean_array
   ```

2. **Register the class** in `data_generator/entitymanager.py`'s `DATA_TYPE_REGISTRY`:
   ```python
   DATA_TYPE_REGISTRY = {
       # ... existing entries ...
       "embedding": EmbeddingGenerator
   }
   ```
3. **Reference the key** in your dataset's YAML under the `columns` list:
   ```yaml
   columns:
     - name: "user_embedding"
       type: "embedding" # matches the registry key
       missing_probability: 0.02
   ```

### Adding New Comparisons to the Comparator

To add a new comparison type for use in the EM pipeline:

1. **Create a comparison class** in `helpers/em_pipeline/comparisons.py` that inherits from `ComparisonBase`:

   ```python
   from helpers.em_pipeline.base import ComparisonBase
   import splink.comparison_library as cl

   class EmbeddingSimilarityComparison(ComparisonBase):
       def create_comparison(self, column_name, **kwargs):
           # Example: cosine similarity for embeddings
           threshold = kwargs.get('threshold', 0.8)
           return cl.EucldieanDistance(
               col_name_or_literal=column_name,
               threshold=threshold
           )
   ```

2. **Register the class** in `helpers/em_pipeline/comparisons.py`:
   ```python
   # At the bottom of the file
   ComparisonRegistry._MAP.update({
       # ... existing entries ...
       "embedding_similarity": EmbeddingSimilarityComparison
   })
   # Or use the decorator:
   # @ComparisonRegistry.register
   # class EmbeddingSimilarityComparison(ComparisonBase):
   #     ...
   ```
3. **Use in configuration** (YAML or factory calls):
   ```yaml
   comparisons:
     - name: "embedding_similarity"
       column: "user_embedding"
       threshold: 0.85
   ```
   Or in code:
   ```python
   from helpers.em_pipeline.comparisons import ComparisonRegistry
   emb_comp = ComparisonRegistry.get("embedding_similarity")(
       column_name="user_embedding", threshold=0.85
   )
   ```

## Further Reading

- [Data Generator Guide](./data_generator/README.md)
- [EM Pipeline (Splink Wrapper) Guide](./helpers/em_pipeline/README.md)
- [Aggregation Framework Guide](./helpers/aggregation/README.md)
- [Pipeline Architecture Details](./helpers/em_pipeline/ARCHITECTURE.md)
