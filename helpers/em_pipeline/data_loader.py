"""
Basic data loading and processing abstraction.
Implements factory pattern for extensibility.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet
import sys
import os
from .base import DataLoaderBase, DataLoaderRegistry


# ===========================================
# CONCRETE DATA LOADER IMPLEMENTATIONS
# ===========================================

@DataLoaderRegistry.register("parquet")
class ParquetLoader(DataLoaderBase):
    """Parquet file loader."""

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from parquet file.

        Args:
            file_path: Path to the parquet file
            **kwargs: Additional arguments passed to pd.read_parquet

        Returns:
            Loaded DataFrame
        """
        sys.path.insert(0, os.path.dirname(os.getcwd()))
        return pd.read_parquet(file_path, **kwargs)


@DataLoaderRegistry.register("csv")
class CSVLoader(DataLoaderBase):
    """CSV file loader."""

    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments passed to pd.read_csv

        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(file_path, **kwargs)


# ===========================================
# BACKWARD COMPATIBLE FUNCTIONS
# ===========================================
# These maintain backward compatibility with existing code

def load_pair_dfs_from_paths(df1_path, df2_path):
    """
    Reads paths and loads the dataframes. Must be in parquet format.
    Backward compatible wrapper.
    """
    loader = ParquetLoader()
    df1 = loader.load_data(df1_path)
    df2 = loader.load_data(df2_path)
    return df1, df2


import numpy as np
import matplotlib.pyplot as plt


def plot_numerical_distribution(df1, df2, num_col):
    df1_clean = df1[num_col].dropna()
    df2_clean = df2[num_col].dropna()

    # Choose bin edges that span the combined range with a width of 50
    min_val = min(df1_clean.min(), df2_clean.min())
    max_val = max(df1_clean.max(), df2_clean.max())
    bins = np.arange(np.floor(min_val / 50) * 50,
                    np.ceil (max_val / 50) * 50 + 50,  # +50 to make the last bin inclusive
                    50)

    # Plot histograms (alpha lets you see overlap)
    df1_clean.plot(kind='hist',
                bins=bins,
                density=False,          # change to True if you want a probability density
                alpha=0.6,
                label='df1',
                color='steelblue',
                edgecolor='black',
                linewidth=1.2)

    df2_clean.plot(kind='hist',
                bins=bins,
                density=False,
                alpha=0.6,
                label='df2',
                color='tomato',
                edgecolor='black',
                linewidth=1.2)

    # Titles / labels / legend / grid – unchanged from your original snippet
    plt.title(f'Histogram of {num_col} (bin width = 50) for df1 and df2')
    plt.xlabel('num_col')
    plt.ylabel('Count')          # use 'Density' if you set density=True above
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def standardise_dt(df1, df2, dt_col):
    """
    Sets the datetimes for both dfs as just date - this is uniquely for the specific situation where
    df1 and df2 have discrepancies in their temporal records.
    Will set both as date.
    Returns a NULL.
    """
    df1[dt_col] = df1[dt_col].dt.floor('D')
    df2[dt_col] = df2[dt_col].dt.floor('D')
    print(f"Set {dt_col} to date for both df1 and df2.")


def normalize_numerical_to_percentile(df1, df2, num_col, new_col_name=None):
    """
    Normalize numerical columns to [0,1] range representing percentiles for use with
    numerical_percentile comparison.

    This function computes the empirical percentile rank of each value across both datasets
    combined, then normalizes to [0,1] where 0 = minimum (0th percentile) and 1 = maximum (100th percentile).

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        num_col: Name of the numerical column to normalize
        new_col_name: Name for the normalized column (if None, uses f"{num_col}_normalized")

    Returns:
        Tuple of (df1_with_normalized_col, df2_with_normalized_col, new_col_name)
    """
    if new_col_name is None:
        new_col_name = f"{num_col}_normalized"

    # Combine values from both datasets to compute percentiles
    combined_values = pd.concat([df1[num_col], df2[num_col]]).dropna()

    if len(combined_values) == 0:
        # Handle edge case where all values are NaN
        df1[new_col_name] = 0.5  # Default to median
        df2[new_col_name] = 0.5
        print(f"Warning: All values in {num_col} are NaN. Set {new_col_name} to 0.5 for both dataframes.")
        return df1, df2, new_col_name

    # Compute percentile ranks using the empirical distribution function
    # This gives us the proportion of values <= each value
    def empirical_percentile_rank(series, values):
        """Compute empirical percentile rank for each value in series."""
        if len(values) == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        # For each value, compute percentage of values in reference set that are <= it
        ranks = []
        for val in series:
            if pd.isna(val):
                ranks.append(np.nan)
            else:
                # Proportion of reference values <= current value
                rank = (values <= val).sum() / len(values)
                ranks.append(rank)
        return pd.Series(ranks, index=series.index)

    # Calculate normalized values (percentile ranks) for both dataframes
    df1[new_col_name] = empirical_percentile_rank(df1[num_col], combined_values)
    df2[new_col_name] = empirical_percentile_rank(df2[num_col], combined_values)

    print(f"Normalized {num_col} to {new_col_name} using empirical percentile ranking.")
    print(f"  Range: [{df1[new_col_name].min():.3f}, {df1[new_col_name].max():.3f}] U [{df2[new_col_name].min():.3f}, {df2[new_col_name].max():.3f}]")

    return df1, df2, new_col_name