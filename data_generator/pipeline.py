import numpy as np
import pandas as pd
from .generator import DataGeneratorFactory


class DataPipelineOrchestrator:

    def __init__(self, config: dict, rng: np.random.Generator = None):
        """Stores the configuration dictionary for features and rows."""
        self.config = config
        if rng == None:
            raise ValueError("Create a root RNG first and pass it here.")
        self.rng = rng  # ✅ Root seed lives here

    def expand_entities(self, mapping_df: pd.DataFrame, rows_per_entity: int,
                        std_dev: float = None) -> pd.DataFrame:
        """
        Replicates entity rows to simulate multiple transaction footprints per individual.

        Each entity gets a transaction count X drawn from a truncated normal distribution:
        X ~ N(mean=rows_per_entity, std=std_dev), clamped to [1, 200].
        Using 1 as the lower bound avoids creating empty entity batches.

        :param mapping_df: DataFrame containing ['global_entity_id', 'local_source_id'].
        :param rows_per_entity: Base mean number of transactions per entity.
        :param std_dev: Controls the spread. Defaults to ~1/3 of the mean for ~99.7% coverage.
        :return: A new expanded DataFrame with duplicated entity rows and a reset index.
        """
        n_entities = len(mapping_df)

        # Fallback standard deviation if not explicitly provided
        if std_dev is None:
            std_dev = rows_per_entity / 3.0

        # 1. Sample transaction counts per entity from a truncated normal distribution
        # np.clip ensures bounds, .astype(int) converts floats to integer counts
        transaction_counts = np.clip(
            self.rng.normal(loc=rows_per_entity, scale=std_dev, size=n_entities),
            1, 200  # Lower bound 1 prevents dropping entities; change to 0 if strict [0,200] needed
        ).astype(int)

        # 2. Vectorised expansion: np.repeat accepts an array of repeat counts
        expanded_indices = np.repeat(mapping_df.index.values, transaction_counts)
        expanded_df = mapping_df.loc[expanded_indices].copy()

        # 3. Reset index for downstream processing
        expanded_df = expanded_df.reset_index(drop=True)
        return expanded_df

    def generate_dataset(self, mapping_df: pd.DataFrame, source_config_key: str,
                         overlap_count: int, unique_count: int):
        """
        Generates an observed dataset where each row gets its own latent ground-truth value.
        For overlapping entities the latent value is shared across datasets; for unique entities it is independent.
        Applies optional asymmetric dropout to overlapping rows in dataset B.
        Returns the observed dataset.
        """
        sim_config = self.config["simulation"]
        rows_per_entity = sim_config[source_config_key]
        dropout_frac = sim_config.get("overlap_dropout_frac", 0.0)

        # Determine overlapping entity IDs from the first part of mapping_df
        overlap_ids = mapping_df.iloc[:overlap_count]["global_entity_id"].values

        # Expand entities to get one row per transaction
        expanded = self.expand_entities(mapping_df, rows_per_entity)  # DataFrame with global_entity_id, local_source_id per row
        # Mask for overlapping rows
        overlap_mask = expanded["global_entity_id"].isin(overlap_ids)

        # Process each feature
        for feat in self.config["features"]:
            f_name = feat["name"]
            f_type = feat["type"]
            generator = DataGeneratorFactory.get_generator(f_type, rng=self.rng)

            n_total = len(expanded)
            # Prepare array for base values; we'll fill according to dtype later
            base_vals = np.empty(n_total, dtype=object)

            # Overlap rows: shared latent truth
            n_overlap = overlap_mask.sum()
            if n_overlap > 0:
                base_overlap = generator.generate_base(n_overlap, feat)  # shape (n_overlap,)
                base_vals[overlap_mask] = base_overlap

            # Unique rows: independent draws for this source
            n_unique = (~overlap_mask).sum()
            if n_unique > 0:
                base_unique = generator.generate_base(n_unique, feat)
                base_vals[~overlap_mask] = base_unique

            # Apply noise and missingness
            expanded[f_name] = generator.apply_noise(base_vals, feat)

        # ----- Optional: drop some overlapping observed rows to simulate missing matches (asymmetric, dataset B only) -----
        if dropout_frac > 0.0 and source_config_key.startswith("source_b"):
            # Identify rows belonging to overlapping entities
            overlap_mask_B = expanded["global_entity_id"].isin(overlap_ids)
            n_overlap_rows = overlap_mask_B.sum()
            n_to_drop = int(n_overlap_rows * dropout_frac)
            if n_to_drop > 0:
                # Randomly select rows to drop among overlapping rows
                drop_indices = self.rng.choice(
                    expanded.index[overlap_mask_B], size=n_to_drop, replace=False
                )
                expanded = expanded.drop(index=drop_indices).reset_index(drop=True)
                # Recompute overlap_mask after dropping rows? Not needed for further processing.

        # Optionally sort by datetime column
        dt_col = next((f["name"] for f in self.config["features"] if f["type"] == "datetime"), None)
        if dt_col and dt_col in expanded.columns:
            expanded = expanded.sort_values(by=dt_col).reset_index(drop=True)

        return expanded
    
    