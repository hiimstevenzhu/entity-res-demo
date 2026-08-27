import abc
from typing import Any
import pandas as pd

class ColumnProcessor(abc.ABC):
    def __init__(self, col_name: str):
        self.col_name = col_name

    @abc.abstractmethod
    def get_agg_func(self) -> Any:
        pass

    def fit_vocab(self, df1: pd.DataFrame, df2: pd.DataFrame, unique_id_col: str) -> None:
        """Updated to accept the dynamic identity column tracking rule."""
        pass

    def post_process(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        return df_agg
