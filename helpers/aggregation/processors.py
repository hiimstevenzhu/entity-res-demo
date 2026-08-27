from typing import Any, List
import pandas as pd
from .base import ColumnProcessor
import json

class NumericalProcessor(ColumnProcessor):
    def get_agg_func(self):
        return lambda s: (float(s.mean()), float(s.min()), float(s.max())) if not s.dropna().empty else (0.0, 0.0, 0.0)

    def post_process(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        col = self.col_name
        if col in df_agg.columns:
            df_agg[f"{col}_mean"] = df_agg[col].apply(lambda tp: tp[0])
            df_agg[f"{col}_min"]  = df_agg[col].apply(lambda tp: tp[1])
            df_agg[f"{col}_max"]  = df_agg[col].apply(lambda tp: tp[2])
            df_agg = df_agg.drop(columns=[col])
        return df_agg


class CategoricalProcessor(ColumnProcessor):
    def __init__(self, col_name: str):
        super().__init__(col_name)
        self.vocab: List[str] = []

    def get_agg_func(self):
        return lambda s: list(map(str, s.dropna()))

    def fit_vocab(self, df1: pd.DataFrame, df2: pd.DataFrame, unique_id_col: str) -> None:
        v_set = set()
        for df in [df1, df2]:
            if self.col_name in df.columns and unique_id_col in df.columns:
                # Fixed: Dynamically grouping by unique_id_col
                tokens = df.groupby(unique_id_col)[self.col_name].apply(lambda x: list(map(str, x.dropna())))
                for item in tokens:
                    v_set.update(item)
        self.vocab = sorted(v_set)

    def post_process(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        col = self.col_name
        if col in df_agg.columns and self.vocab:
            v_idx = {t: i for i, t in enumerate(self.vocab)}
            def build_vector(tokens):
                vec = [0.0] * len(self.vocab)
                for t in tokens:
                    if t in v_idx:
                        vec[v_idx[t]] += 1.0
                return vec
            df_agg[f"{col}_vector"] = df_agg[col].apply(build_vector)
        return df_agg


# ------------------------------------------------------------------
# Example of a developer adding a new type (e.g. text/embedding)
# ------------------------------------------------------------------
class TextualProcessor(CategoricalProcessor):
    """Can be custom overridden later if it diverges from Categorical behavior"""
    pass


## New processor for taking an a list of tokens per row - and outputting a combined union list.
class SetUnionProcessor(ColumnProcessor):
    """
    Processor that takes rows containing lists or elements, merges them, 
    and returns a single, unique, flat list for each aggregated group.
    """
    def get_agg_func(self):
        def parse_and_flatten_elements(series: pd.Series) -> List[Any]:
            unique_elements = set()
            for row_value in series.dropna():
                # Avoid capturing unwanted literal string flags
                if row_value in (None, "None", "NaN", "nan"):
                    continue
                
                # If row is already a native Python list structure
                if isinstance(row_value, list):
                    for element in row_value:
                        if element is not None and str(element) != "None":
                            unique_elements.add(str(element))
                
                # Fallback: handles cases where lists are read as raw string literals e.g. "[1911, 2011]"
                elif isinstance(row_value, str) and row_value.startswith("[") and row_value.endswith("]"):
                    try:
                        parsed_list = json.loads(row_value.replace("'", '"'))
                        if isinstance(parsed_list, list):
                            for element in parsed_list:
                                if element is not None and str(element) != "None":
                                    unique_elements.add(str(element))
                    except json.JSONDecodeError:
                        unique_elements.add(row_value)
                else:
                    unique_elements.add(str(row_value))
                    
            return sorted(list(unique_elements))
            
        return parse_and_flatten_elements

    def post_process(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        # Renames output column name to explicitly track it as tokens for Splink matching
        return df_agg
