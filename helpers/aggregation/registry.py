from typing import Dict, Type
from .base import ColumnProcessor
from .processors import NumericalProcessor, CategoricalProcessor, SetUnionProcessor, TextualProcessor

class ProcessorRegistry:
    """The central map binding string identifiers to their handling classes."""
    
    _MAP: Dict[str, Type[ColumnProcessor]] = {
        "numerical": NumericalProcessor,
        "categorical": CategoricalProcessor,
        "textual": TextualProcessor,
        "set_union": SetUnionProcessor
    }

    @classmethod
    def register(cls, identifier: str, processor_cls: Type[ColumnProcessor]) -> None:
        """Enables runtime/plugin injections of external custom types."""
        cls._MAP[identifier] = processor_cls

    @classmethod
    def get(cls, identifier: str) -> Type[ColumnProcessor]:
        if identifier not in cls._MAP:
            raise KeyError(f"Type identifier '{identifier}' is not configured in the registry.")
        return cls._MAP[identifier]
