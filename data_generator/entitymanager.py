import numpy as np
import pandas as pd
from typing import Tuple


class EntitySpaceManager:
    """Manages the hidden true entity space, cross-source overlaps, and non-corresponding IDs."""

    def __init__(self, dataset_size: int, overlap_rate: float, rng: np.random.Generator = None):
        """
        Initializes the entity space parameters and allocates the true underlying global identity pool.

        :param dataset_size: The number of unique entities that should exist within EACH generated dataset.
        :param overlap_rate: Target fractional match rate between Dataset A and B (must be 0.05 to 0.15).
        """
        if dataset_size <= 0:
            raise ValueError("Dataset size must be a positive integer greater than zero")
        
        self.rng = rng
        self.dataset_size = dataset_size
        self.overlap_rate = overlap_rate

        # Calculate the exact number of overlapping entities shared between both datasets
        self.overlap_count = int(self.dataset_size * self.overlap_rate)

        # Calculate the number of exclusive, non-overlapping entities needed for each data source
        self.unique_per_source = self.dataset_size - self.overlap_count

        # Total unique global entities required to build the entire simulation universe without duplicates
        self.total_needed_entities = self.overlap_count + (2 * self.unique_per_source)

        # Allocate a large hidden true Global Identity space array using contiguous 64-bit integers
        self.global_entity_ids = np.arange(100000, 100000 + self.total_needed_entities, dtype=np.int64)


    def generate_source_mappings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partitions the global identity pool and assigns non-corresponding identifiers to each source.

        Ensures that Dataset A and Dataset B share exactly the configured overlap percentage.

        :return: A tuple containing two DataFrames (mapping_df_a, mapping_df_b) with columns:
                 ['global_entity_id', 'local_source_id']
        """
        # Create a randomized copy of the global space to shuffle characteristics across datasets
        shuffled_space = self.global_entity_ids.copy()
        self.rng.shuffle(shuffled_space) # Use the random root.

        # Slice out the shared overlap segment from the front of the shuffled pool
        overlap_entities = shuffled_space[: self.overlap_count]

        # Slice out the exclusive segment for Source A
        source_a_end_idx = self.overlap_count + self.unique_per_source
        source_a_only = shuffled_space[self.overlap_count : source_a_end_idx]

        # Slice out the remaining exclusive segment for Source B
        source_b_only = shuffled_space[source_a_end_idx :]

        # Combine the shared and exclusive segments for each source using fast array concatenation
        entities_a = np.concatenate([overlap_entities, source_a_only])
        entities_b = np.concatenate([overlap_entities, source_b_only])

        # Load arrays directly into Pandas DataFrames to initialize the structural tables
        mapping_df_a = pd.DataFrame({"global_entity_id": entities_a})
        mapping_df_b = pd.DataFrame({"global_entity_id": entities_b})

        # Generate non-corresponding string identifiers using highly optimized vectorized series casting
        mapping_df_a["local_source_id"] = "DF1_" + mapping_df_a["global_entity_id"].astype(str)
        mapping_df_b["local_source_id"] = "DF2_" + mapping_df_b["global_entity_id"].astype(str)

        return mapping_df_a, mapping_df_b
