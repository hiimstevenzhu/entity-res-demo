# Aggregation Framework

A flexible, factory‑based framework for computing behavioural aggregations over entity‑resolved datasets. It lets you define column‑wise aggregation logic (mean, sum, custom vector operations, etc.) via pluggable processors, and automatically handles the creation of dataset‑level features ready for downstream modeling.

## Overview

The framework revolves around the `Aggregator` class (`engine.py`), which:
- Takes a list of **type groups** (e.g., `[["numerical"], ["categorical"]]`) and corresponding **column groups** (e.g., `[["price"], ["category"]]`).
- Builds a pipeline that first **fits** on raw data to learn global vocabularies, categorical mappings, or other statistics.
- Then **transforms** each dataset, producing aggregated features with informative column names (e.g., `price_mean`, `category_mode`, `vector_features_mean_vector`).

Custom aggregation logic is added by implementing a `ColumnProcessor` subclass and registering it in `registry.py`.

## Usage

```python
import pandas as pd
from aggregator_framework.engine import Aggregator

# 1. Prepare raw tracking data (two example sources)
raw_data_1 = pd.DataFrame({
    "local_source_id": [1, 2, 3],
    "price": [25.0, 35.0, 100.0],
    "category": ["electronics", "gadgets", "apparel"]
})

raw_data_2 = pd.DataFrame({
    "local_source_id": [4, 5, 6],
    "price": [30.0, 90.0, 110.0],
    "category": ["electronics", "shoes", "apparel"]
})

# 2. Instantiate the Aggregator with structural mapping rules
# Format: type_groups matched index-for-index with col_groups
agg = Aggregator(
    type_groups=[["numerical"], ["categorical"]],
    col_groups=[["price"], ["category"]]
)

# 3. Fit the pipeline (extracts unified global vocabulary indexes)
agg.fit(raw_data_1, raw_data_2)

# 4. Transform individual DataFrames natively
clean_df_1 = agg.transform(raw_data_1, dataset_label="store_alpha_west")
clean_df_2 = agg.transform(raw_data_2, dataset_label="store_beta_east")

print(clean_df_1.columns)
# Output: Index(['local_source_id', 'price_mean', 'category_mode', 'source_dataset'], dtype='object')
```

## Extending: Adding New Processors

To support a new column type (e.g., embeddings, IP addresses, custom strings):

1. **Create the class definition** in `processors.py`, for example:

   ```python
   # Inside processors.py
   class EmbeddingProcessor(ColumnProcessor):
       def get_agg_func(self):
           # Example: mean‑pool vector strings "0.1,0.2,0.3"
           return lambda s: np.mean(np.stack(s.str.split(',').astype(float)), axis=0)
       
       def get_output_name(self, base_name: str) -> str:
           return f"{base_name}_mean_vector"
       
       # Define custom post_process logic here if needed...
   ```

2. **Link it to registry.py**:

   ```python
   # Inside registry.py
   _MAP = {
       # ... existing entries ...
       "embedding": EmbeddingProcessor
   }
   ```

3. **Use it**:

   ```python
   from aggregator_framework.engine import Aggregator
   import pandas as pd

   df = pd.DataFrame({
       "local_source_id": [1, 2],
       "vector_features": ["0.1,0.2,0.3", "0.5,0.6,0.7"]
   })

   # The engine dynamically initializes the EmbeddingProcessor behind the scenes
   agg = Aggregator(
       type_groups=[["embedding"]],
       col_groups=[["vector_features"]]
   )

   agg.fit(df)  # fit on single or multiple dataframes as needed
   result = agg.transform(df, dataset_label="nn_embeddings_v2")
   print(result.columns) 
   # Outputs columns: ['local_source_id', 'vector_features_mean_vector', 'source_dataset']
   ```

No changes to the core `Aggregator` or `engine.py` are required.

## Configuration

The framework is configured entirely through the arguments passed to `Aggregator`:
- `type_groups`: List of lists, where each inner list contains the keys (as defined in `registry.py`) for the columns that should be processed together.
- `col_groups`: Parallel list of lists containing the actual column names from your DataFrames.

All other behaviour (missing‑value handling, output naming, etc.) is encapsulated within the individual `ColumnProcessor` implementations.

## Dependencies

- `pandas`
- `numpy` (used by some processors)

See the imports in `engine.py` and `processors.py` for exact version requirements.

-- 

*For more details on the existing processors (numerical, categorical, etc.) refer to the source code in `processors.py`.*