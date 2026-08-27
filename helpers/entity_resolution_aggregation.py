"""
Entity Resolution Aggregation Helper Functions

Contains helper functions for advanced aggregation strategies in entity resolution,
based on the logic from splink_low_card.ipynb.
"""

import numpy as np
import pandas as pd


def calc_noisy_or_probs(probs):
    """
    Calculate noisy-or aggregation: 1 - ∏(1-p) with log1p for numerical stability.

    Args:
        probs: Array-like of probabilities

    Returns:
        Noisy-or aggregated probability
    """
    # Clip probabilities to avoid log(0) issues
    clipped_probs = np.clip(probs, 0.0, 1.0 - 1e-12)
    # Use log1p for numerical stability: log(1 + x) where x = -p
    log_sum = np.sum(np.log1p(-clipped_probs))
    # Result: 1 - exp(log_sum) = 1 - ∏(1-p)
    return 1.0 - np.exp(log_sum)


def calc_softmax_prob(probs):
    """
    Calculate softmax-based aggregation that highlights high-confidence signals.

    Args:
        probs: Array-like of probabilities

    Returns:
        Softmax-weighted probability
    """
    # Convert to numpy array
    p_arr = np.array(probs, dtype=np.float64)

    # Handle edge cases
    if len(p_arr) == 0:
        return 0.0
    if len(p_arr) == 1:
        return p_arr[0]

    # For numerical stability, subtract max before exponentiation
    # This prevents overflow in exp()
    max_val = np.max(p_arr)
    exp_vals = np.exp(p_arr - max_val)
    sum_exp = np.sum(exp_vals)

    # Avoid division by zero
    if sum_exp == 0:
        return np.mean(p_arr)

    # Calculate softmax weights
    softmax_weights = exp_vals / sum_exp

    # Return weighted sum
    return np.sum(p_arr * softmax_weights)


def calc_top_k_mean_weight(weights, k=3):
    """
    Calculate mean of top k weights to reduce noise.

    Args:
        weights: Array-like of weights (typically match weights)
        k: Number of top weights to consider (default: 3)

    Returns:
        Mean of top k weights
    """
    # Convert to numpy array and sort
    w_arr = np.array(weights)

    # Handle edge case where we have fewer than k elements
    if len(w_arr) <= k:
        return np.mean(w_arr)

    # Get top k weights and return their mean
    top_k_weights = np.sort(w_arr)[-k:]
    return np.mean(top_k_weights)


def compute_source_sizes(df1, df2, id_column):
    """
    Compute source sizes mapping for entity IDs.

    Args:
        df1: First dataframe (left)
        df2: Second dataframe (right)
        id_column: Column name containing entity IDs

    Returns:
        Dictionary mapping entity IDs to their counts in each dataset
    """
    # Count occurrences in each dataframe
    counts_l = df1[id_column].value_counts()
    counts_r = df2[id_column].value_counts()

    # Create mapping dictionary
    size_mapping = {}

    # Get all unique IDs from both dataframes
    all_ids = set(counts_l.index) | set(counts_r.index)

    for entity_id in all_ids:
        size_mapping[entity_id] = {
            "size_l": counts_l.get(entity_id, 0),
            "size_r": counts_r.get(entity_id, 0),
        }

    return size_mapping


def calculate_match_density(sum_p, size_l, size_r, epsilon=1e-6):
    """
    Calculate match density: sum_p / (size_l * size_r)

    Args:
        sum_p: Sum of match probabilities for entity pair
        size_l: Size of left entity (number of records)
        size_r: Size of right entity (number of records)
        epsilon: Small value to prevent division by zero

    Returns:
        Match density score
    """
    denominator = size_l * size_r + epsilon
    return sum_p / denominator


def calculate_weighted_jaccard(sum_p, size_l, size_r, epsilon=1e-6):
    """
    Calculate weighted Jaccard coefficient: sum_p / (size_l + size_r - sum_p)

    Args:
        sum_p: Sum of match probabilities for entity pair
        size_l: Size of left entity (number of records)
        size_r: Size of right entity (number of records)
        epsilon: Small value to prevent division by zero

    Returns:
        Weighted Jaccard coefficient
    """
    denominator = size_l + size_r - sum_p + epsilon
    return sum_p / denominator


def calculate_weighted_overlap(sum_p, size_l, size_r, epsilon=1e-6):
    """
    Calculate weighted overlap coefficient: sum_p / min(size_l, size_r)

    Args:
        sum_p: Sum of match probabilities for entity pair
        size_l: Size of left entity (number of records)
        size_r: Size of right entity (number of records)
        epsilon: Small value to prevent division by zero

    Returns:
        Weighted overlap coefficient
    """
    denominator = min(size_l, size_r) + epsilon
    return sum_p / denominator


def calculate_score_prob_ratio(sum_p, size_l, size_r, u_base, epsilon=1e-6):
    """
    Calculate score probability expected ratio: log((sum_p + ε) / (expected_random_matches + ε))
    where expected_random_matches = size_l * size_r * u_base

    Args:
        sum_p: Sum of match probabilities for entity pair
        size_l: Size of left entity (number of records)
        size_r: Size of right entity (number of records)
        u_base: Base m-probability (probability of random match)
        epsilon: Small value to prevent division by zero and log(0)

    Returns:
        Log ratio score
    """
    numerator = sum_p + epsilon
    expected_random_matches = size_l * size_r * u_base
    denominator = expected_random_matches + epsilon
    return np.log(numerator / denominator)


def aggregate_entity_pair_matched_pairs(
    df_predict, id_columns=["local_source_id_l", "local_source_id_r"]
):
    """
    Aggregate matched pairs by entity IDs to compute various aggregation strategies.
    Mirrors the groupby.agg logic from splink_low_card.ipynb.

    Args:
        df_predict: DataFrame with match predictions (must have match_probability and match_weight columns)
        id_columns: List containing [left_id_column, right_id_column]

    Returns:
        DataFrame with aggregated scores for each entity pair
    """
    # Make a copy to avoid modifying original
    df_agg = df_predict.copy()

    # Ensure we have the required columns
    if "match_probability" not in df_agg.columns:
        raise ValueError("DataFrame must contain 'match_probability' column")
    if "match_weight" not in df_agg.columns:
        raise ValueError("DataFrame must contain 'match_weight' column")

    left_id_col, right_id_col = id_columns

    # Group by entity pairs and aggregate
    aggregated = (
        df_agg.groupby([left_id_col, right_id_col])
        .agg(
            sum_p=("match_probability", "sum"),
            max_w=("match_weight", "max"),
            observed_edges=("match_probability", "size"),  # count of comparisons
        )
        .reset_index()
    )

    # Apply different aggregation strategies
    # For each group, we need to apply the aggregation functions to the original probabilities/weights

    # Helper function to apply aggregation to grouped data
    def apply_agg_functions(group):
        probs = group["match_probability"].values
        weights = group["match_weight"].values

        return pd.Series(
            {
                "top_3_mean_w": calc_top_k_mean_weight(weights, k=3),
                "score_noisy_or": calc_noisy_or_probs(probs),
                "score_softmax_p": calc_softmax_prob(probs),
            }
        )

    # Apply aggregation functions to each group
    agg_results = (
        df_agg.groupby([left_id_col, right_id_col])
        .apply(apply_agg_functions)
        .reset_index()
    )

    # Merge the aggregated results
    result = pd.merge(aggregated, agg_results, on=[left_id_col, right_id_col])

    # Rename columns for clarity
    result = result.rename(
        columns={
            "sum_p": "sum_p",
            "max_w": "max_w",
            "observed_edges": "observed_edges",
            "top_3_mean_w": "top_3_mean_w",
            "score_noisy_or": "score_noisy_or",
            "score_softmax_p": "score_softmax_p",
        }
    )

    return result


def create_uniform_quartile_bins(df, features):
    """Discretizes features using uniform 4-bin quartile boundaries

    and returns both the binned DataFrame and the exact boundary definitions.
    """
    binned_df = df.copy()
    bin_edges_map = {}

    for col in features:
        clean_col = df[col].dropna()
        if clean_col.empty:
            binned_df[f"{col}_bin"] = 0
            bin_edges_map[col] = {0: "Empty Data"}
            continue

        min_v = float(clean_col.min())
        max_v = float(clean_col.max())

        if min_v == max_v:
            binned_df[f"{col}_bin"] = 0
            bin_edges_map[col] = {0: f"[{min_v:.4f}]"}
            continue

        # Enforce exact quartile bins for ALL features uniformly
        quantiles = [0, 25, 50, 75, 100]
        bins = np.percentile(clean_col, quantiles)

        # Force structural boundaries
        bins = np.array(bins, dtype=float)
        bins[0] = min_v
        bins[-1] = max_v

        # Deduplicate overlapping tiers
        bins = np.unique(bins)
        bins = np.sort(bins)

        # Pad boundaries slightly to absorb rounding drift securely
        padded_bins = bins.copy()
        padded_bins[0] -= 1e-5
        padded_bins[-1] += 1e-5

        if len(padded_bins) >= 2 and np.all(np.diff(padded_bins) > 0):
            # Categorise rows into 4 distinct ordinal bins (0, 1, 2, 3)
            binned_df[f"{col}_bin"] = pd.cut(
                df[col], bins=padded_bins, labels=False, include_lowest=True
            )

            # Store the real boundary definitions for tracking/plotting
            bin_labels = {}
            for i in range(len(bins) - 1):
                bin_labels[i] = f"[{bins[i]:.3f} to {bins[i + 1]:.3f}]"
            bin_edges_map[col] = bin_labels
        else:
            binned_df[f"{col}_bin"] = 0
            bin_edges_map[col] = {0: f"[{min_v:.4f} to {max_v:.4f}]"}

    return binned_df, bin_edges_map


def train_fellegi_sunter_parameters_unsupervised(df, features, max_iter=50):
    """Computes conditional m and u distributions using unsupervised EM."""
    param_dict = {}
    p_match = 0.1
    eps = 1e-6

    feature_bins = {}
    for col in features:
        bin_col = f"{col}_bin"
        if bin_col in df.columns:
            feature_bins[col] = np.sort(df[bin_col].dropna().unique())

    for iteration in range(max_iter):
        # E-STEP
        if iteration == 0:
            responsibilities = np.full(len(df), p_match)
        else:
            prior_odds = p_match / (1.0 - p_match + eps)
            log_posterior_odds = np.full(len(df), np.log(prior_odds))

            for col in features:
                bin_col = f"{col}_bin"
                if bin_col not in df.columns:
                    continue

                m_probs = param_dict[col]["m"]
                u_probs = param_dict[col]["u"]

                m_val = df[bin_col].map(m_probs).fillna(eps).astype(float)
                u_val = df[bin_col].map(u_probs).fillna(eps).astype(float)
                log_posterior_odds += np.log(m_val / u_val)

            posterior_odds = np.exp(np.clip(log_posterior_odds, -20, 20))
            responsibilities = posterior_odds / (1.0 + posterior_odds)

        # M-STEP
        new_param_dict = {}
        for col, active_bins in feature_bins.items():
            bin_col = f"{col}_bin"
            m_weighted = np.zeros(len(active_bins))
            u_weighted = np.zeros(len(active_bins))

            for i, bin_val in enumerate(active_bins):
                mask = df[bin_col] == bin_val
                m_weighted[i] = np.sum(responsibilities[mask])
                u_weighted[i] = np.sum((1.0 - responsibilities)[mask])

                m_weighted[i] = max(m_weighted[i], eps)
                u_weighted[i] = max(u_weighted[i], eps)

            m_probs = m_weighted / np.sum(m_weighted)
            u_probs = u_weighted / np.sum(u_weighted)
            fs_weights = dict(zip(active_bins, np.log2(m_probs / u_probs)))

            new_param_dict[col] = {
                "m": dict(zip(active_bins, m_probs)),
                "u": dict(zip(active_bins, u_probs)),
                "w": fs_weights,
            }

        param_dict = new_param_dict
        p_match = np.clip(float(np.mean(responsibilities)), eps, 1.0 - eps)

    return param_dict, p_match


def predict_adjusted_match_probabilities(df, params, p_match):
    """Predicts match probabilities and sums up total match weights."""
    df_result = df.copy()
    eps = 1e-6

    prior_odds = p_match / (1.0 - p_match + eps)
    total_log_odds = np.full(len(df_result), np.log(prior_odds))
    df_result["final_match_weight"] = 0.0

    for col in params.keys():
        bin_col = f"{col}_bin"
        if bin_col not in df_result.columns:
            continue

        m_probs = params[col]["m"]
        u_probs = params[col]["u"]
        w_weights = params[col]["w"]

        m_val = df_result[bin_col].map(m_probs).fillna(eps).astype(float)
        u_val = df_result[bin_col].map(u_probs).fillna(eps).astype(float)

        total_log_odds += np.log(m_val / u_val)
        df_result["final_match_weight"] += df_result[bin_col].map(w_weights).fillna(0.0)

    odds = np.exp(np.clip(total_log_odds, -20, 20))
    df_result["final_adjusted_prob"] = odds / (1.0 + odds)
    return df_result


def second_layer_splink_evaluation(
    aggregated_features, feature_columns=None, max_iter=15
):
    """Evaluation wrapper using a pure quartile-percentile unsupervised FS structure."""
    df_meta = aggregated_features.copy()

    if feature_columns is None:
        id_like_cols = [
            col
            for col in df_meta.columns
            if "id" in col.lower() or "source" in col.lower()
        ]
        feature_columns = [
            col
            for col in df_meta.columns
            if df_meta[col].dtype in ["float64", "int64", "float32", "int32"]
            and col not in id_like_cols
        ]

    # Step 1: Execute uniform 4-bin quartile splits
    binned_df, bin_boundaries = create_uniform_quartile_bins(df_meta, feature_columns)

    # Step 2: Unsupervised EM loop execution
    model_params, p_match = train_fellegi_sunter_parameters_unsupervised(
        binned_df, feature_columns, max_iter=max_iter
    )

    # Step 3: Map probabilities and append boundary text metadata to params dictionary
    final_scored_df = predict_adjusted_match_probabilities(
        binned_df, model_params, p_match
    )

    for col in model_params.keys():
        model_params[col]["intervals"] = bin_boundaries[col]

    df_meta["final_adjusted_prob"] = final_scored_df["final_adjusted_prob"]
    df_meta["final_match_weight"] = final_scored_df["final_match_weight"]

    return df_meta, model_params


# def create_skewed_bins(df, features):
#     """Discretizes continuous metrics using strictly monotonic, safe boundaries."""
#     binned_df = df.copy()

#     for col in features:
#         clean_col = df[col].dropna()
#         if clean_col.empty:
#             binned_df[f"{col}_bin"] = 0
#             continue

#         min_v = float(clean_col.min())
#         max_v = float(clean_col.max())

#         if min_v == max_v:
#             binned_df[f"{col}_bin"] = 0
#             continue

#         if "score" in col or "weighted" in col:
#             base_bins = [0.0, 0.2, 0.5, 0.75, 0.90, 0.95, 0.98, 1.0]
#             bins = [b for b in base_bins if min_v <= b <= max_v]
#             if len(bins) < 2:
#                 bins = [min_v, max_v]
#         else:
#             quantiles = [0, 20, 40, 60, 80, 90, 95, 100]
#             bins = np.percentile(clean_col, quantiles)

#         bins = np.array(bins, dtype=float)
#         bins[0] = min_v
#         bins[-1] = max_v

#         bins = np.unique(bins)
#         bins = np.sort(bins)

#         bins[0] -= 1e-5
#         bins[-1] += 1e-5

#         if len(bins) >= 2 and np.all(np.diff(bins) > 0):
#             binned_df[f"{col}_bin"] = pd.cut(
#                 df[col], bins=bins, labels=False, include_lowest=True
#             )
#         else:
#             binned_df[f"{col}_bin"] = 0

#     return binned_df


# def train_fellegi_sunter_parameters_unsupervised(df, features, max_iter=50):
#     """Computes conditional m and u distributions using unsupervised EM."""
#     param_dict = {}
#     p_match = 0.1
#     eps = 1e-6

#     feature_bins = {}
#     for col in features:
#         bin_col = f"{col}_bin"
#         if bin_col in df.columns:
#             feature_bins[col] = np.sort(df[bin_col].dropna().unique())

#     for iteration in range(max_iter):
#         # E-STEP
#         if iteration == 0:
#             responsibilities = np.full(len(df), p_match)
#         else:
#             prior_odds = p_match / (1.0 - p_match + eps)
#             log_posterior_odds = np.full(len(df), np.log(prior_odds))

#             for col in features:
#                 bin_col = f"{col}_bin"
#                 if bin_col not in df.columns:
#                     continue

#                 m_probs = param_dict[col]["m"]
#                 u_probs = param_dict[col]["u"]

#                 m_val = df[bin_col].map(m_probs).fillna(eps).astype(float)
#                 u_val = df[bin_col].map(u_probs).fillna(eps).astype(float)
#                 log_posterior_odds += np.log(m_val / u_val)

#             posterior_odds = np.exp(np.clip(log_posterior_odds, -20, 20))
#             responsibilities = posterior_odds / (1.0 + posterior_odds)

#         # M-STEP
#         new_param_dict = {}
#         for col, active_bins in feature_bins.items():
#             bin_col = f"{col}_bin"
#             m_weighted = np.zeros(len(active_bins))
#             u_weighted = np.zeros(len(active_bins))

#             for i, bin_val in enumerate(active_bins):
#                 mask = df[bin_col] == bin_val
#                 m_weighted[i] = np.sum(responsibilities[mask])
#                 u_weighted[i] = np.sum((1.0 - responsibilities)[mask])

#                 m_weighted[i] = max(m_weighted[i], eps)
#                 u_weighted[i] = max(u_weighted[i], eps)

#             m_probs = m_weighted / np.sum(m_weighted)
#             u_probs = u_weighted / np.sum(u_weighted)

#             # Pre-calculate log2 match weights for each bin tier
#             fs_weights = dict(zip(active_bins, np.log2(m_probs / u_probs)))

#             new_param_dict[col] = {
#                 "m": dict(zip(active_bins, m_probs)),
#                 "u": dict(zip(active_bins, u_probs)),
#                 "w": fs_weights,
#             }

#         param_dict = new_param_dict
#         p_match = np.clip(float(np.mean(responsibilities)), eps, 1.0 - eps)

#     return param_dict, p_match


# def predict_adjusted_match_probabilities(df, params, p_match):
#     """Predicts probabilities and compiles absolute Fellegi-Sunter match weights."""
#     df_result = df.copy()
#     eps = 1e-6

#     prior_odds = p_match / (1.0 - p_match + eps)
#     total_log_odds = np.full(len(df_result), np.log(prior_odds))

#     # Initialize a column for the cumulative Fellegi-Sunter weight sum
#     df_result["final_match_weight"] = 0.0

#     for col in params.keys():
#         bin_col = f"{col}_bin"
#         if bin_col not in df_result.columns:
#             continue

#         m_probs = params[col]["m"]
#         u_probs = params[col]["u"]
#         w_weights = params[col]["w"]

#         m_val = df_result[bin_col].map(m_probs).fillna(eps).astype(float)
#         u_val = df_result[bin_col].map(u_probs).fillna(eps).astype(float)

#         total_log_odds += np.log(m_val / u_val)

#         # Add the log2 weight directly into our tracking column
#         df_result["final_match_weight"] += df_result[bin_col].map(w_weights).fillna(0.0)

#     odds = np.exp(np.clip(total_log_odds, -20, 20))
#     df_result["final_adjusted_prob"] = odds / (1.0 + odds)
#     return df_result


# def second_layer_splink_evaluation(
#     aggregated_features, feature_columns=None, max_iter=15
# ):
#     """Evaluation wrapper returning parameters alongside raw dataframe predictions."""
#     df_meta = aggregated_features.copy()

#     if feature_columns is None:
#         id_like_cols = [
#             col
#             for col in df_meta.columns
#             if "id" in col.lower() or "source" in col.lower()
#         ]
#         feature_columns = [
#             col
#             for col in df_meta.columns
#             if df_meta[col].dtype in ["float64", "int64", "float32", "int32"]
#             and col not in id_like_cols
#         ]

#     binned_df = create_skewed_bins(df_meta, feature_columns)
#     model_params, p_match = train_fellegi_sunter_parameters_unsupervised(
#         binned_df, feature_columns, max_iter=max_iter
#     )
#     final_scored_df = predict_adjusted_match_probabilities(
#         binned_df, model_params, p_match
#     )

#     df_meta["final_adjusted_prob"] = final_scored_df["final_adjusted_prob"]
#     df_meta["final_match_weight"] = final_scored_df["final_match_weight"]

#     return df_meta, model_params
