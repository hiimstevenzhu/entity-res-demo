"""
Abstract base classes and registries for the entity resolution pipeline components.
Implements factory pattern for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List
import pandas as pd


# ===========================================
# COMPARISON FACTORY
# ===========================================

class ComparisonBase(ABC):
    """Abstract base class for all comparison types."""

    @abstractmethod
    def create_comparison(self, column_name: str, **kwargs) -> Any:
        """Create a comparison object for the specified column."""
        pass


class ComparisonRegistry:
    """Registry for comparison types."""

    _registry: Dict[str, Type[ComparisonBase]] = {}

    @classmethod
    def register(cls, comparison_type: str):
        """Decorator to register a comparison type."""
        def wrapper(comparison_class: Type[ComparisonBase]):
            cls._registry[comparison_type] = comparison_class
            return comparison_class
        return wrapper

    @classmethod
    def get(cls, comparison_type: str) -> Type[ComparisonBase]:
        """Get a comparison class by type."""
        if comparison_type not in cls._registry:
            raise ValueError(f"Unknown comparison type: {comparison_type}")
        return cls._registry[comparison_type]

    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered comparison types."""
        return list(cls._registry.keys())


# ===========================================
# BLOCKING RULE FACTORY
# ===========================================

class BlockingRuleBase(ABC):
    """Abstract base class for all blocking rule types."""

    @abstractmethod
    def get_rules(self, **kwargs) -> List[str]:
        """Get blocking rules."""
        pass


class BlockingRuleRegistry:
    """Registry for blocking rule types."""

    _registry: Dict[str, Type[BlockingRuleBase]] = {}

    @classmethod
    def register(cls, rule_type: str):
        """Decorator to register a blocking rule type."""
        def wrapper(rule_class: Type[BlockingRuleBase]):
            cls._registry[rule_type] = rule_class
            return rule_class
        return wrapper

    @classmethod
    def get(cls, rule_type: str) -> Type[BlockingRuleBase]:
        """Get a blocking rule class by type."""
        if rule_type not in cls._registry:
            raise ValueError(f"Unknown blocking rule type: {rule_type}")
        return cls._registry[rule_type]

    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered blocking rule types."""
        return list(cls._registry.keys())


# ===========================================
# DATA LOADER FACTORY
# ===========================================

class DataLoaderBase(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file."""
        pass


class DataLoaderRegistry:
    """Registry for data loader types."""

    _registry: Dict[str, Type[DataLoaderBase]] = {}

    @classmethod
    def register(cls, loader_type: str):
        """Decorator to register a data loader type."""
        def wrapper(loader_class: Type[DataLoaderBase]):
            cls._registry[loader_type] = loader_class
            return loader_class
        return wrapper

    @classmethod
    def get(cls, loader_type: str) -> Type[DataLoaderBase]:
        """Get a data loader class by type."""
        if loader_type not in cls._registry:
            raise ValueError(f"Unknown data loader type: {loader_type}")
        return cls._registry[loader_type]

    @classmethod
    def list_types(cls) -> List[str]:
        """List all registered data loader types."""
        return list(cls._registry.keys())