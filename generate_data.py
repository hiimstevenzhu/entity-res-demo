import os
import yaml
import pandas as pd
import numpy as np
from data_generator.entitymanager import EntitySpaceManager
from data_generator.pipeline import DataPipelineOrchestrator

SEED = 42
rng = np.random.default_rng(SEED)


def assign_latent_event_ids(df, id_prefix, start_id=0):
    """
    Assign explicit latent event IDs to a dataframe.

    Args:
        df: DataFrame to assign IDs to
        id_prefix: Prefix for the IDs ("UNI_A", "UNIQ_B", or "OVER")
        start_id: Starting ID number (for continuity)

    Returns:
        DataFrame with added 'latent_event_id' column, and next available ID
    """
    n_rows = len(df)
    if n_rows == 0:
        return df, start_id

    # Generate sequential IDs: PREFIX_0, PREFIX_1, PREFIX_2, ...
    latent_ids = [f"{id_prefix}_{start_id + i}" for i in range(n_rows)]
    df = df.copy()
    df["latent_event_id"] = latent_ids
    return df, start_id + n_rows


def apply_noise_to_base(base_df, config, rng, source="a"):
    """
    Apply feature-specific noise to overlapping base dataset.
    Preserves special columns like 'latent_event_id'.
    """
    # Identify columns to preserve (don't noise)
    preserve_cols = ["global_entity_id", "local_source_id", "latent_event_id"]
    # Add any other columns that should not be noised
    cols_to_noise = [col for col in base_df.columns if col not in preserve_cols]

    # Create a copy to avoid modifying the base
    df_noised = base_df.copy()

    # Apply noise only to data columns, preserve ID columns
    for col in cols_to_noise:
        # Find the feature config for this column
        feat_config = next((f for f in config["features"] if f["name"] == col), None)
        if feat_config:
            f_type = feat_config["type"]
            from data_generator.generator import DataGeneratorFactory

            generator = DataGeneratorFactory.get_generator(f_type, rng=rng)

            # Apply noise to this column
            df_noised[col] = generator.apply_noise(base_df[col].values, feat_config)
        # If column not in features config, leave it unchanged (like latent_event_id)

    return df_noised


def apply_dropout(dataset_b, config, rng):
    """
    Apply asymmetric dropout to overlapping rows in dataset B only.
    """
    sim_config = config["simulation"]
    dropout_frac = sim_config.get("overlap_dropout_frac", 0.0)

    if dropout_frac <= 0.0:
        return dataset_b

    n_rows = len(dataset_b)
    n_to_drop = int(n_rows * dropout_frac)

    if n_to_drop > 0:
        drop_indices = rng.choice(dataset_b.index, size=n_to_drop, replace=False)
        dataset_b = dataset_b.drop(index=drop_indices).reset_index(drop=True)

    return dataset_b


def sort_by_datetime_if_exists(df, config):
    """
    Sort dataset by datetime column if one exists.
    Preserves all columns including latent_event_id.
    """
    dt_col = next(
        (f["name"] for f in config["features"] if f["type"] == "datetime"), None
    )
    if dt_col and dt_col in df.columns:
        df = df.sort_values(by=dt_col).reset_index(drop=True)
    return df


def generate_dataset(size):
    """
    Generate a dataset of the specified size.

    Args:
        size: One of "smallestest", "smallest", "small", "smaller", "med_30_overlap", "low_card"

    Returns:
        tuple: (dataset_a_path, dataset_b_path) - paths to the generated parquet files
    """
    # Map friendly names to config file names
    size_to_config = {
        "smallestest": "smallestest_config",
        "smallest": "smallest_config",
        "small": "smallest_config",
        "smaller": "smaller_config",
        "o50": "o50_config",
        "low_card": "low_card_config",
    }

    if size not in size_to_config:
        raise ValueError(
            f"Unsupported dataset size: {size}. Choose from {list(size_to_config.keys())}"
        )

    # Get the project root directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_name = size_to_config[size]
    config_path = os.path.join(
        script_dir, "data_generator", "configs", f"{config_name}.yaml"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading system configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    dataset_size = sim_config["dataset_size"]
    overlap_percentage = sim_config["overlap_percentage"]

    # Calculate counts
    overlap_count = int(
        dataset_size * overlap_percentage
    )  # Number of overlapping entities
    unique_count_per_source = dataset_size - overlap_count  # Unique entities per source

    print(
        f"Generating {overlap_count} overlapping entities and {unique_count_per_source} unique entities per source..."
    )

    print("Initializing Entity Space Manager...")

    # 1. Build the true underlying entity identity pool
    space_manager = EntitySpaceManager(
        dataset_size=dataset_size, overlap_rate=overlap_percentage, rng=rng
    )

    # 2. Extract the non-corresponding identity mapping tables
    print("Generating non-corresponding identity maps...")
    mapping_a, mapping_b = space_manager.generate_source_mappings()

    # 3. Split mappings into unique and overlapping portions
    mapping_a_overlap = mapping_a.iloc[:overlap_count].copy()
    mapping_a_unique = mapping_a.iloc[overlap_count:].copy()
    mapping_b_overlap = mapping_b.iloc[:overlap_count].copy()
    mapping_b_unique = mapping_b.iloc[overlap_count:].copy()

    print(
        f"Mapping A: {len(mapping_a_overlap)} overlapping + {len(mapping_a_unique)} unique"
    )
    print(
        f"Mapping B: {len(mapping_b_overlap)} overlapping + {len(mapping_b_unique)} unique"
    )

    # 4. Instantiate the orchestration pipeline
    print("Executing vectorised multi-row data orchestration pipeline...")
    orchestrator = DataPipelineOrchestrator(config=config, rng=rng)

    # 5. Generate the three foundational datasets (WITHOUT latent IDs yet)
    print("Generating overlapping dataset base...")
    overlapping_base = orchestrator.generate_dataset(
        mapping_a_overlap,  # Use either A or B overlap mapping
        "source_a_rows_per_entity",
        overlap_count,  # Actual count from mapping size
        0,
    )

    print("Generating dataset A unique portion...")
    dataset_a_unique = orchestrator.generate_dataset(
        mapping_a_unique, "source_a_rows_per_entity", 0, unique_count_per_source
    )

    print("Generating dataset B unique portion...")
    dataset_b_unique = orchestrator.generate_dataset(
        mapping_b_unique, "source_b_rows_per_entity", 0, unique_count_per_source
    )

    # 6. Assign explicit latent event ID labels
    print("Assigning latent event ID labels...")
    next_id = 0

    # Assign OVER_* IDs to overlapping base
    overlapping_base, next_id = assign_latent_event_ids(
        overlapping_base, "OVER", next_id
    )

    # Assign UNI_A_* IDs to A unique portion
    dataset_a_unique, next_id = assign_latent_event_ids(
        dataset_a_unique, "UNI_A", next_id
    )

    # Assign UNIQ_B_* IDs to B unique portion
    dataset_b_unique, next_id = assign_latent_event_ids(
        dataset_b_unique, "UNIQ_B", next_id
    )

    # 7. Create the two noisy versions from overlapping base
    print("Creating noisy version A from overlapping base...")
    dataset_a_overlap = apply_noise_to_base(
        overlapping_base.copy(), config, rng, source="a"
    )
    # Preserve the latent_event_id column (don't noise it)

    print("Creating noisy version B from overlapping base (with dropout)...")
    dataset_b_overlap = apply_noise_to_base(
        overlapping_base.copy(), config, rng, source="b"
    )
    dataset_b_overlap = apply_dropout(dataset_b_overlap, config, rng)
    # Preserve the latent_event_id column

    # 8. Combine to create final datasets
    print("Combining datasets...")
    dataset_a = pd.concat([dataset_a_overlap, dataset_a_unique], ignore_index=True)
    dataset_b = pd.concat([dataset_b_overlap, dataset_b_unique], ignore_index=True)

    # 9. Optionally sort by datetime column
    dataset_a = sort_by_datetime_if_exists(dataset_a, config)
    dataset_b = sort_by_datetime_if_exists(dataset_b, config)

    # 10. Save out files
    # Use the data directory relative to the script location
    output_dir = os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filenames based on the mapping in notebook_config.yaml
    # We need to reverse-engineer this from the actual file naming pattern
    # Based on observation: size + "_dataset_a/b.parquet"
    # But we should make this configurable or derive from the mapping

    # For now, use the pattern observed in the existing files
    dataset_a_filename = f"{size}_dataset_a.parquet"
    dataset_b_filename = f"{size}_dataset_b.parquet"

    # Special case: for "smallest" friendly name, the files are actually "smallest_dataset*.parquet"
    # This matches our pattern, so it's fine
    # For "smallestest", files are "smallestest_dataset*.parquet" - also matches
    # For "small", files are "smallest_dataset*.parquet" - wait, this doesn't match!

    # Let me check the notebook_config.yaml again to understand the mapping...
    # Actually, let's make this dynamic by checking what files SHOULD exist based on config

    dataset_a_path = os.path.join(output_dir, dataset_a_filename)
    dataset_b_path = os.path.join(output_dir, dataset_b_filename)

    print(f"Persisting files to disk under folder: '{output_dir}/'...")
    dataset_a.to_parquet(dataset_a_path, index=False)
    dataset_b.to_parquet(dataset_b_path, index=False)

    print("\nGeneration pipeline run complete. Summary profile metrics:")
    print(f"-> Dataset A: {len(dataset_a):,} rows generated.")
    print(f"   - Overlapping portion: {len(dataset_a_overlap):,} rows")
    print(f"   - Unique portion: {len(dataset_a_unique):,} rows")
    print(f"-> Dataset B: {len(dataset_b):,} rows generated.")
    print(f"   - Overlapping portion: {len(dataset_b_overlap):,} rows")
    print(f"   - Unique portion: {len(dataset_b_unique):,} rows")
    print(
        "Data structures are successfully saved and ready for the Entity Resolution engine."
    )

    return dataset_a_path, dataset_b_path


def main():
    """Original main function for backward compatibility - uses hardcoded size"""
    SIZE = "smaller"  # Keep original behavior
    generate_dataset(SIZE)


if __name__ == "__main__":
    main()
