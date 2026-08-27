# EM Pipeline (Splink Wrapper)

A modular, factory‑based pipeline for entity resolution with the Splink library. This package abstracts data loading, comparison definition, blocking rule creation, training, prediction, and visualisation into interchangeable components, allowing you to switch datasets or tweak settings without touching the core logic.

## Module Structure

```
em_pipeline/
├── __init__.py
├── base.py            # Abstract base classes & registries
├── comparisons.py     # Comparison factory & implementations
├── blocking_rules.py  # Blocking rule factory & implementations
├── data_loader.py     # Data loader factory & implementations
├── data_generator.py  # Utility functions for synthetic data (optional)
├── splink_helper.py   # Splink‑specific helper functions
├── utils.py           # Common utilities (register dataframes, labels, etc.)
└── # (training_pipeline.py, prediction_pipeline.py, visualization_pipeline.py are implied via imports in helper)
```

*(The actual training, prediction and visualization logic is exposed through helper modules that are imported from the notebooks; see the source files for details.)*

## Core Concept: Factory Pattern

Each major concern (loading data, defining comparisons, specifying blocking rules) follows the same pattern:

1. **Abstract Base Class** – defines the interface (e.g., `DataLoaderBase`, `ComparisonBase`, `BlockingRuleBase`).
2. **Registry** – a dictionary mapping string names to concrete classes (populated via decorators or manual registration).
3. **Factory Helper** – a function like `DataLoaderRegistry.get(name)` that returns the class, which you then instantiate.

### Example Usage (from the notebooks)

```python
from em_pipeline.data_loader import load_dataset
from em_pipeline.comparisons import ComparisonRegistry
from em_pipeline.blocking_rules import BlockingRuleRegistry
from em_pipeline.utils import register_dataframe, register_labels_table
from splink import DuckDBAPI, Linker, SettingsCreator

# 1. Load data (using factory)
df1 = load_dataset("path/to/file1.parquet", format="parquet")
df2 = load_dataset("path/to/file2.parquet", format="parquet")

# 2. Create comparisons via registry
num_comp = ComparisonRegistry.get("numerical")(
    column_name="amount_noisy", thresholds=[0.05, 0.15, 0.4, 0.7]
)
dt_comp = ComparisonRegistry.get("datetime")(
    column_name="timestamp_noisy", day_thresholds=[0, 1, 2, 15]
)
cat_comp = ComparisonRegistry.get("categorical")(
    column_name="status"
)

comparisons = [num_comp, dt_comp, cat_comp]

# 3. Get blocking rules
blocking_rules = BlockingRuleRegistry.get("general").get_rules()
strict_training_rules = BlockingRuleRegistry.get("strict").get_rules()

# 4. Settings & linker
con = duckdb.connect(database="splink_temp_workspace.duckdb")
db_api = DuckDBAPI(connection=con)

df1_splink = register_dataframe(db_api, df1, "dataset1")
df2_splink = register_dataframe(db_api, df2, "dataset2")

settings = SettingsCreator(
    link_type="link_only",
    unique_id_column_name="unique_id",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=comparisons,
    retain_intermediate_calculation_columns=True,
)

linker = Linker([df1_splink, df2_splink], settings, db_api=db_api)
```

## Extending the Pipeline

To add a new **data‑type generator**, **comparison**, **blocking rule**, or **data loader**:

1. **Create a class** that inherits from the appropriate base class in `base.py`.
   - For comparisons: inherit from `ComparisonBase` and implement `create_comparison`.
   - For blocking rules: inherit from `BlockingRuleBase` and implement `get_rules`.
   - For data loaders: inherit from `DataLoaderBase` and implement `load_data`.
2. **Register the class** in the corresponding registry:
   - In `comparisons.py`: add to `ComparisonRegistry._MAP` or use the `@ComparisonRegistry.register` decorator.
   - In `blocking_rules.py`: similarly for `BlockingRuleRegistry`.
   - In `data_loader.py`: for `DataLoaderRegistry`.
3. **Use the new component** by referencing its registry key (the string you used to register it) in YAML config or factory calls.

No changes to the orchestration code (`splink_helper.py`, `utils.py`, or the notebooks) are necessary.

## Configuration

The pipeline is driven primarily by the arguments you pass to the factory objects. In practice, you will likely keep these arguments in YAML files (see the notebooks for examples) and load them at runtime.

Typical configurable items:
- **Data loader**: file path, format (`parquet`, `csv`), preprocessing options.
- **Comparison**: column name, thresholds/noise parameters specific to the data type.
- **Blocking rules**: rule set name (`general`, `strict`, `datetime`, `column‑specific`) or custom rule lists.
- **Linker settings**: `link_type`, `retain_intermediate_calculation_columns`, etc.

Because all components are lightweight and configurable, the same pipeline can process different datasets simply by swapping the configuration values or YAML files.

## Dependencies

- `splink`
- `duckdb` (or another backend supported by Splink)
- `pandas`
- Optional: `numpy`, `altair` (for visualisation helpers)

See `requirements.txt` or the notebook’s imports for a full list.

-- 

*Tip: Study the helper modules’ docstrings and the `ARCHITECTURE.md` file for deeper details on each factory’s interface.*