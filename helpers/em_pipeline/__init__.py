"""Entity Resolution Pipeline Package

A modular pipeline for entity resolution using the Splink library.
Implements factory pattern for extensibility.
"""

# Import factories and registries for extension
from .base import (
    ComparisonBase,
    ComparisonRegistry,
    BlockingRuleBase,
    BlockingRuleRegistry,
    DataLoaderBase,
    DataLoaderRegistry,
)

# Import concrete implementations for convenience
from .comparisons import (
    NumericalComparison,
    DateTimeComparison,
    CategoricalComparison,
    TextComparison,
    FirstNameComparison,
    LastNameComparison,
    create_numerical_comparison,
    create_datetime_comparison,
    create_categorical_comparison,
    create_text_comparison,
    create_first_name_comparison,
    create_last_name_comparison,
    create_custom_comparison_from_config,
)

from .blocking_rules import (
    GeneralBlockingRules,
    StrictTrainingRules,
    DateTimeBlockingRules,
    ColumnSpecificBlockingRules,
    get_general_blocking_rules,
    get_strict_training_rules,
    get_datetime_blocking_rules,
    get_blocking_rules_for_column,
    get_blocking_rules_from_config,
)

from .data_loader import ParquetLoader, CSVLoader, load_pair_dfs_from_paths

# Import utility functions
from .data_loader import plot_numerical_distribution, standardise_dt, normalize_numerical_to_percentile

# Import helper modules
from . import data_loader, blocking_rules, comparisons, splink_helper

# Version information
__version__ = "1.0.0"
__author__ = "Steven"
