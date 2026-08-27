from typing import List
import pandas as pd
from .registry import ProcessorRegistry
from .base import ColumnProcessor

class Aggregator:
    def __init__(self, type_groups: List[str], col_groups: List[List[str]], unique_id_col: str = "unique_id"):
        if len(type_groups) != len(col_groups):
            raise ValueError("type_groups and col_groups must have the same length")
            
        self.unique_id_col = unique_id_col
        self.processors: List[ColumnProcessor] = []
        
        for type_string, column_list in zip(type_groups, col_groups):
            processor_cls = ProcessorRegistry.get(type_string)
            for col_name in column_list:
                self.processors.append(processor_cls(col_name))

    def fit(self, df1: pd.DataFrame, df2: pd.DataFrame) -> "Aggregator":
        for processor in self.processors:
            # Pass identity tracking context to internal models
            processor.fit_vocab(df1, df2, unique_id_col=self.unique_id_col)
        return self

    def transform(self, df_raw: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
        if self.unique_id_col not in df_raw.columns:
            raise KeyError(f"DataFrame must contain initialized unique id column: '{self.unique_id_col}'")
            
        agg_spec = {}
        for proc in self.processors:
            if proc.col_name in df_raw.columns:
                agg_spec[proc.col_name] = pd.NamedAgg(column=proc.col_name, aggfunc=proc.get_agg_func())

        # Fixed: Natively groupby whichever column key was passed at startup
        df_agg = df_raw.groupby(self.unique_id_col).agg(**agg_spec).reset_index()

        for proc in self.processors:
            df_agg = proc.post_process(df_agg)

        df_agg["source_dataset"] = dataset_label
        return df_agg
