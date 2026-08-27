# Splink Pipeline Architecture for Entity Resolution

## Overview
This document outlines a modular architecture for refactoring the existing Splink notebook into a reusable pipeline that can process different datasets with minimal configuration changes. The pipeline abstracts the core components: data loading, comparison creation, blocking rule configuration, training, prediction, and visualization.

## Module Structure with Factory Pattern

```
em_pipeline/
├── __init__.py
├── ARCHITECTURE.md
├── base.py                  # Abstract base classes and registries
├── comparisons.py           # Comparison factory and implementations
├── blocking_rules.py        # Blocking rule factory and implementations
├── data_loader.py           # Data loader factory and implementations
├── data_generator.py        # Data generation utility functions
├── splink_helper.py         # Splink-specific helper functions
└── utils.py                 # Utility functions for common operations
```

## Factory Pattern Implementation

The pipeline now implements a factory pattern with registries for easy extensibility:

### Comparison Factory (`comparisons.py`)
- **Abstract Base**: `ComparisonBase`
- **Registry**: `ComparisonRegistry` 
- **Implementations**: `NumericalComparison`, `DateTimeComparison`, `CategoricalComparison`, `TextComparison`
- **Usage**: 
  ```python
  from em_pipeline.comparisons import ComparisonRegistry
  factory = ComparisonRegistry.get("numerical")
  comparison = factory().create_comparison("column_name", thresholds=[0.1, 0.5])
  ```

### Blocking Rule Factory (`blocking_rules.py`)
- **Abstract Base**: `BlockingRuleBase`
- **Registry**: `BlockingRuleRegistry`
- **Implementations**: `GeneralBlockingRules`, `StrictTrainingRules`, `DateTimeBlockingRules`, `ColumnSpecificBlockingRules`
- **Usage**:
  ```python
  from em_pipeline.blocking_rules import BlockingRuleRegistry
  factory = BlockingRuleRegistry.get("general")
  rules = factory().get_rules()
  ```

### Data Loader Factory (`data_loader.py`)
- **Abstract Base**: `DataLoaderBase`
- **Registry**: `DataLoaderRegistry`
- **Implementations**: `ParquetLoader`, `CSVLoader`
- **Usage**:
  ```python
  from em_pipeline.data_loader import DataLoaderRegistry
  factory = DataLoaderRegistry.get("parquet")
  df = factory().load_data("file.parquet")
  ```

## Module Interfaces

### data_loader.py
Handles loading and preprocessing of datasets.

```python
def load_dataset(file_path: str, format: str = "parquet", **kwargs) -> pd.DataFrame:
    """Load dataset from various formats using factory pattern."""
    
def preprocess_datetime(df: pd.DataFrame, column: str, freq: str = 'D') -> pd.DataFrame:
    """Floor datetime column to specified frequency for splink compatibility"""

def handle_missing_values(df: pd.DataFrame, strategy: str = 'keep') -> pd.DataFrame:
    """Handle missing values according to strategy"""

def prepare_for_splink(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Prepare dataframe for Splink (ensure proper column types, etc.)"""
```

### comparisons.py
Defines comparison levels for different data types using factory pattern.

```python
def create_numerical_comparison(
    column_name: str = "numerical_noisy",
    thresholds: List[float] = [0.05, 0.15, 0.4, 0.7]) -> CustomComparison:
    """Create numerical comparison with percentage difference levels"""

def create_datetime_comparison(
    column_name: str = "datetime_noisy",
    day_thresholds: List[int] = [0, 1, 2, 15]) -> dict:
    """Create datetime comparison with day thresholds"""

def create_categorical_comparison(column_name: str = "categorical") -> CustomComparison:
    """Create categorical comparison"""

def create_text_comparison(
    column_name: str,
    thresholds: List[float] = [0.9, 0.8, 0.7]) -> CustomComparison:
    """Create text comparison using string similarity metrics"""
```

### blocking_rules.py
Manages blocking rule definitions for training and prediction using factory pattern.

```python
def get_general_blocking_rules() -> List[str]:
    """Get general blocking rules for prediction"""

def get_strict_training_rules() -> List[str]:
    """Get strict blocking rules for training/estimation"""

def get_datetime_blocking_rules() -> List[str]:
    """Get datetime-specific blocking rules"""

def get_blocking_rules_for_column(column_name: str, rule_type: str) -> List[str]:
    """Get blocking rules specific to a column"""
```

### training_pipeline.py
Handles model training and parameter estimation.

```python
def run_training_pipeline(linker: Linker, strict_rules: List[str], labels_table_name: str = None) -> Linker:
    """Run complete training pipeline: estimate probability, u params, then EM training"""

def estimate_m_from_labels(linker: Linker, label_colname: str) -> None:
    """Estimate m parameters from label column"""

def estimate_u_random_sampling(linker: Linker, max_pairs: int = 1e7, seed: int = None) -> None:
    """Estimate u parameters using random sampling"""

def estimate_parameters_em(linker: Linker, blocking_rule: str, **kwargs) -> EMTrainingSession:
    """Estimate parameters using expectation maximisation"""
```

### prediction_pipeline.py
Handles prediction and clustering operations.

```python
def run_prediction(linker: Linker, threshold_match_probability: float = None, threshold_match_weight: float = None) -> SplinkDataFrame:
    """Run prediction and return scored pairwise comparisons"""

def cluster predictions: Splinking_above_threshold(link
: SplinkDataFrame, threshold):
    """ew matches above a specified threshold"""

def cluster_pairwise_predictions(link
: SplinkDataFrame, threshold_match_probability: float = None, threshold_match_weight: float = None) -> SplinkDataFrame:
    """ter pairwise predictions at a given threshold"""
```

### visualization_pipeline.py
Handles visualization and evaluation functions.

```python
def plot_match_weights(linker: Linker) -> alt.Chart:
    """Create match weights chart"""

def plot_match_weights_histogram(df_predict: SplinkDataFrame, target_bins: int = 30) -> alt.Chart:
    """Create match weights histogram"""

def accuracy_analysis_from_labels(linker: Linker, labels_table_name: str, **kwargs) -> alt.Chart:
    """Perform accuracy analysis from labels table"""

def prediction_errors_from_labels(linker: Linker, labels_table_name: str, **kwargs) -> SplinkDataFrame:
    """Find prediction errors from labels table"""

def unlinkables_chart(linker: Linker, **kwargs) -> alt.Chart:
    """Create unlinkables chart"""
```

### utils.py
Utility functions for common operations.

```python
def register_dataframe(linker: Linker, df: pd.DataFrame, table_name: str, overwrite: bool = False) -> SplinkDataFrame:
    """Register a dataframe with the linker's database"""

def register_labels_table(linker: Linker, df_labels: pd.DataFrame, table_name: str, overwrite: bool = False) -> SplinkDataFrame:
    """Register a labels table with the linker's database"""

def get_dataset_names(linker: Linker) -> List[str]:
    """Get the internal dataset names used by Splink"""

def safe_drop_table(linker: Linker, table_name: str) -> None:
    """Safely drop a table if it exists"""
```

## Configuration

Configuration is handled through a YAML file (`config.yaml`) that specifies:

- Dataset paths and preprocessing parameters
- Comparison definitions and parameters (using factory types)
- Blocking rules for training and prediction (using factory types)
- Linker settings

## Usage Example

```python
from em_pipeline.data_loader import load_dataset, preprocess_datetime
from em_pipeline.comparisons import create_numerical_comparison, create_datetime_comparison, create_categorical_comparison
from em_pipeline.blocking_rules import get_general_blocking_rules, get_strict_training_rules
from em_pipeline.training_pipeline import run_training_pipeline
from em_pipeline.prediction_pipeline import run_prediction, cluster_pairwise_predictions
from em_pipeline.visualization_pipeline import plot_match_weights, accuracy_analysis_from_labels
from em_pipeline.utils import register_dataframe, register_labels_table
from splink import DuckDBAPI, Linker, SettingsCreator

# Load and preprocess data
df1 = load_dataset("path/to/dataset1.parquet")
df1 = preprocess_datetime(df1, "datetime_column")

df2 = load_dataset("path/to/dataset2.parquet")
df2 = preprocess_datetime(df2, "datetime_column")

# Setup database connection
con = duckdb.connect(database="splink_temp_workspace.duckdb")
con.execute("SET memory_limit = '24GB';")
con.execute("SET temp_directory = './duckdb_spill_dir.tmp';")
db_api = DuckDBAPI(connection=con)

# Register dataframes
df1_splink = register_dataframe(db_api, df1, "dataset1")
df2_splink = register_dataframe(db_api, df2, "dataset2")

# Create comparisons using factory pattern
comparisons = [
    create_numerical_comparison("numerical_column", [0.05, 0.15, 0.4, 0.7]),
    create_datetime_comparison("datetime_column", [0, 1, 2, 15]),
    create_categorical_comparison("categorical_column")
]

# Get blocking rules
blocking_rules = get_general_blocking_rules()
strict_training_rules = get_strict_training_rules()

# Create settings
settings = SettingsCreator(
    link_type="link_only",
    unique_id_column_name="unique_id",
    blocking_rules_to_generate_predictions=blocking_rules,
    comparisons=comparisons,
    retain_intermediate_calculation_columns=True,
    additional_columns_to_retain=["source_id", "global_id"]
)

# Create linker
linker = Linker([df1_splink, df2_splink], settings, db_api=db_api)

# Run training pipeline
linker = run_training_pipeline(linker, strict_training_rules)

# Run prediction
df_predict = run_prediction(linker)

# Cluster predictions
df_clustered = cluster_pairwise_predictions(df_predict, threshold_match_probability=0.5)

# Generate visualizations
match_weights_chart = plot_match_weights(linker)

# If labels are available, run evaluation
# labels_df = load_labels("path/to/labels.csv")
# labels_table = register_labels_table(db_api, labels_df, "ground_truth_labels")
# accuracy_chart = accuracy_analysis_from_labels(linker, "ground_truth_labels", add_metrics=["f1", "accuracy"])

# Save model
linker.misc.save_model_to_json("trained_model.json", overwrite=True)
```

## Benefits

1. **Modularity**: Each component is isolated and can be tested/replaced independently
2. **Configurability**: Dataset-specific parameters are externalized in config files
3. **Reusability**: The same pipeline can be used with different datasets by changing configuration
4. **Maintainability**: Changes to one component don't affect others
5. **Extensibility**: New comparison types, blocking rules, or preprocessing steps can be added easily without modifying existing code
6. **Separation of Concerns**: Data loading, comparison definition, training, prediction, and visualization are cleanly separated

## Implementation Notes

1. The pipeline assumes DuckDB as the backend but can be adapted for other backends
2. Error handling and logging should be added in production implementation
3. The configuration system supports environment-specific overrides
4. Unit tests should be written for each module to ensure correctness
5. Performance optimizations (like caching intermediate results) can be added to specific modules
6. Adding new types only requires:
   - Creating a new class that implements the appropriate base class
   - Registering it with the corresponding registry using the decorator
   - No changes to existing code required