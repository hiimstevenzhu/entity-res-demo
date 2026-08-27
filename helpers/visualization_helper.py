# Visualization helper functions for entity resolution aggregation analysis.
# Split into two parts: heatmap visualization and threshold evaluation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def plot_aggregation_heatmaps(
    aggregated_df: pd.DataFrame,
    score_columns: list,
    notebook_dir,
    id_columns: list | None = None,
    figsize_per_plot: tuple = (6, 5),
    dpi: int = 150,
    max_entities: int = 50,
    cmap: str = "viridis",
) -> None:
    """
    Create heatmaps showing aggregated scores between entities for each score column.
    Each heatmap has left entity IDs as rows, right entity IDs as columns,
    and cell values represent the aggregated score.

    Parameters:
    -----------
    aggregated_df : pd.DataFrame
        DataFrame containing aggregation scores and entity IDs (output from aggregate_entity_pair_matched_pairs)
    score_columns : list
        List of score columns to visualize as heatmaps
    id_columns : list | None
        Columns containing [left_entity_id, right_entity_id]. If None, uses ["local_source_id_l", "local_source_id_r"]
    figsize_per_plot : tuple
        Size of each individual heatmap plot
    dpi : int
        Resolution for saved figures
    max_entities : int
        Maximum number of unique entities to display per axis (to avoid overly large plots).
        If exceeded, entities are sampled.
    cmap : str
        Colormap for heatmaps

    Returns:
    --------
    None (displays and saves plots)
    """
    if id_columns is None:
        id_columns = ["local_source_id_l", "local_source_id_r"]
    left_id_col, right_id_col = id_columns

    # Get unique entity IDs
    left_ids = aggregated_df[left_id_col].unique()
    right_ids = aggregated_df[right_id_col].unique()

    # Optionally limit number of entities for readability
    if len(left_ids) > max_entities:
        left_ids = np.random.choice(left_ids, size=max_entities, replace=False)
        left_ids = np.sort(left_ids)
        print(f"Limited left entities to {max_entities} (random sample)")
    if len(right_ids) > max_entities:
        right_ids = np.random.choice(right_ids, size=max_entities, replace=False)
        right_ids = np.sort(right_ids)
        print(f"Limited right entities to {max_entities} (random sample)")

    n_scores = len(score_columns)
    # Determine grid layout
    n_cols = min(3, n_scores)
    n_rows = (n_scores + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        constrained_layout=True,
        dpi=dpi,
    )
    # Ensure axes is 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each score as a heatmap
    for idx, score_col in enumerate(score_columns):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # Filter data to selected entities
        mask = aggregated_df[left_id_col].isin(left_ids) & aggregated_df[
            right_id_col
        ].isin(right_ids)
        df_plot = aggregated_df.loc[mask, [left_id_col, right_id_col, score_col]].copy()

        if df_plot.empty:
            ax.text(
                0.5,
                0.5,
                "No data for selected entities",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{score_col}\n(No Data)")
            continue

        # Pivot to create heatmap matrix
        try:
            heatmap_data = df_plot.pivot(
                index=left_id_col, columns=right_id_col, values=score_col
            )
            # Reindex to ensure all selected entities are present (fill missing with NaN)
            heatmap_data = heatmap_data.reindex(index=left_ids, columns=right_ids)
        except Exception:
            # Fallback if pivot fails (e.g., duplicates)
            heatmap_data = (
                df_plot.groupby([left_id_col, right_id_col])[score_col]
                .mean()
                .unstack(fill_value=np.nan)
                .reindex(index=left_ids, columns=right_ids)
            )

        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            center=0 if heatmap_data.min().min() < 0 else None,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_title(f"{score_col}", fontsize=12)
        ax.set_xlabel("Right Entity ID")
        ax.set_ylabel("Left Entity ID")

        # Rotate x labels if many entities
        if len(right_ids) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        if len(left_ids) > 10:
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    # Hide unused subplots
    for idx in range(n_scores, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Aggregated Score Heatmaps: Left Entity vs Right Entity",
        fontsize=16,
        y=1.02,
    )

    # Show and save
    plt.show()

    try:
        from notebook_helper import ensure_image_dir

        img_dir = ensure_image_dir(notebook_dir)
        import os

        img_path = os.path.join(img_dir, "aggregation_score_heatmaps.png")
        fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {img_path}")
    except ImportError:
        fig.savefig("aggregation_score_heatmaps.png", dpi=dpi, bbox_inches="tight")
        print("Figure saved as: aggregation_score_heatmaps.png")

    plt.close(fig)


def plot_threshold_evaluation(
    aggregated_df: pd.DataFrame,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    score_columns: list,
    notebook_dir,
    id_columns: list | None = None,
    bins: int = 30,
    figsize_per_plot: tuple = (6, 4),
    dpi: int = 150,
) -> None:
    """
    Create precision-recall-F1 curves binned into equal-width bins for each score column.
    For each bin (threshold range), computes precision, recall, and F1.

    Parameters:
    -----------
    aggregated_df : pd.DataFrame
        DataFrame containing aggregation scores and entity IDs
    df1 : pd.DataFrame
        First dataset with latent_event_id column
    df2 : pd.DataFrame
        Second dataset with latent_event_id column
    score_columns : list
        List of score columns to evaluate
    id_columns : list | None
        Columns containing [left_entity_id, right_entity_id]. If None, uses ["local_source_id_l", "local_source_id_r"]
    bins : int
        Number of equal-width bins to divide the score range into
    figsize_per_plot : tuple
        Size of each individual plot
    dpi : int
        Resolution for saved figures

    Returns:
    --------
    None (displays and saves plots)
    """
    if id_columns is None:
        id_columns = ["local_source_id_l", "local_source_id_r"]
    left_id_col, right_id_col = id_columns

    # Compute entity-level ground truth
    print("Computing entity-level ground truth for threshold evaluation...")
    latent_map1 = df1.groupby("local_source_id")["latent_event_id"].apply(set).to_dict()
    latent_map2 = df2.groupby("local_source_id")["latent_event_id"].apply(set).to_dict()

    def compute_entity_ground_truth(row):
        l_id = row[left_id_col]
        r_id = row[right_id_col]
        set_l = latent_map1.get(l_id, set())
        set_r = latent_map2.get(r_id, set())
        return len(set_l & set_r) > 0

    # Work on a copy
    df_eval = aggregated_df.copy()
    df_eval["is_true_match_entity"] = df_eval.apply(compute_entity_ground_truth, axis=1)

    total_true_matches = df_eval["is_true_match_entity"].sum()
    total_pairs = len(df_eval)
    print(
        f"Found {total_true_matches} true entity pairs out of {total_pairs} total pairs "
        f"({100 * total_true_matches / total_pairs:.2f}%)"
    )

    n_scores = len(score_columns)
    n_cols = min(3, n_scores)
    n_rows = (n_scores + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
        constrained_layout=True,
        dpi=dpi,
    )
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # For each score column, compute binned metrics
    for idx, score_col in enumerate(score_columns):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        scores = df_eval[score_col].values
        y_true = df_eval["is_true_match_entity"].astype(int).values

        # Handle constant scores
        if np.all(scores == scores[0]):
            ax.text(
                0.5,
                0.5,
                "Constant scores\n(std=0)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{score_col}\n(No Discrimination)")
            ax.set_xlabel("Score Bin")
            ax.set_ylabel("Metric Value")
            ax.grid(True, alpha=0.3)
            continue

        # Create bins based on score range (equal width)
        score_min, score_max = np.min(scores), np.max(scores)
        bin_edges = np.linspace(score_min, score_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign each score to a bin
        bin_indices = np.digitize(scores, bin_edges) - 1
        # Ensure bin_indices within [0, bins-1]
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        # Initialize arrays for metrics per bin
        precision_bin = np.zeros(bins)
        recall_bin = np.zeros(bins)
        f1_bin = np.zeros(bins)
        count_bin = np.zeros(bins, dtype=int)

        # Compute histogram of scores and true positives per bin
        tp_hist = np.zeros(bins)
        total_hist = np.zeros(bins)
        for i in range(len(scores)):
            b = bin_indices[i]
            total_hist[b] += 1
            if y_true[i] == 1:
                tp_hist[b] += 1

        # Now compute cumulative from high bin to low bin
        tp_cum = 0
        fp_cum = 0
        # Start from highest bin
        for b in range(bins - 1, -1, -1):
            tp_cum += tp_hist[b]
            fp_cum += total_hist[b] - tp_hist[b]
            fn_cum = (
                total_true_matches - tp_cum
            )  # fn_cum assigned but not used - keeping for clarity
            precision = tp_cum / (tp_cum + fp_cum + 1e-10)
            recall = tp_cum / (total_true_matches + 1e-10)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
            else:
                f1 = 0.0
            precision_bin[b] = precision
            recall_bin[b] = recall
            f1_bin[b] = f1
            count_bin[b] = np.sum(
                total_hist[b:]
            )  # number of samples with score >= bin_edges[b]

        # Plot precision, recall, f1 vs bin center
        ax.plot(
            bin_centers,
            precision_bin,
            "b-",
            label="Precision",
            linewidth=2,
        )
        ax.plot(
            bin_centers,
            recall_bin,
            "g-",
            label="Recall",
            linewidth=2,
        )
        ax.plot(
            bin_centers,
            f1_bin,
            "r-",
            label="F1",
            linewidth=2,
        )
        ax.set_xlabel("Score Value")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{score_col}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([score_min, score_max])
        ax.set_ylim([0, 1.05])

        # Optionally annotate best F1
        best_idx = np.argmax(f1_bin)
        best_f1 = f1_bin[best_idx]
        best_thresh = bin_centers[best_idx]
        ax.plot(
            best_thresh,
            best_f1,
            "ro",
            markersize=8,
            label=f"Best F1={best_f1:.3f}@thresh={best_thresh:.3f}",
        )
        ax.legend(loc="best", fontsize=9)

    # Hide unused subplots
    for idx in range(n_scores, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Threshold Evaluation: Precision, Recall, F1 binned into equal-width intervals",
        fontsize=16,
        y=1.02,
    )

    # Show and save
    plt.show()

    try:
        from notebook_helper import ensure_image_dir

        img_dir = ensure_image_dir(notebook_dir)
        import os

        img_path = os.path.join(img_dir, "threshold_evaluation_binned.png")
        fig.savefig(img_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {img_path}")
    except ImportError:
        fig.savefig("threshold_evaluation_binned.png", dpi=dpi, bbox_inches="tight")
        print("Figure saved as: threshold_evaluation_binned.png")

    plt.close(fig)


def plot_fellegi_sunter_weights(params):
    """Generates a professional horizontal bar chart showing learned weights

    mapped directly to their true numeric column interval boundaries.
    """
    plot_data = []
    for feature, values in params.items():
        for bin_idx, weight in values["w"].items():
            # Extract real calculated boundary labels securely
            interval_str = values["intervals"].get(bin_idx, f"Bin {bin_idx}")
            plot_data.append(
                {
                    "Label": f"{feature}\n{interval_str}",
                    "Weight (Bits)": weight,
                }
            )

    df_plot = pd.DataFrame(plot_data)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, len(params) * 2.2))

    colors = ["#2ecc71" if w >= 0 else "#e74c3c" for w in df_plot["Weight (Bits)"]]

    bars = ax.barh(
        y=df_plot["Label"],
        width=df_plot["Weight (Bits)"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )

    ax.axvline(x=0, color="#2c3e50", linestyle="-", linewidth=1.2, alpha=0.7)
    ax.set_title(
        "Fellegi-Sunter Weight Summary (Uniform Quartile Bin Distributions)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Match Weight Importance ($\log_2(m/u)$ Bits)", fontsize=11)
    ax.set_ylabel("Metrics and Calculated Quartile Ranges", fontsize=11)

    for bar in bars:
        width = bar.get_width()
        x_pos = width + 0.2 if width >= 0 else width - 0.2
        align = "left" if width >= 0 else "right"
        ax.text(
            x=x_pos,
            y=bar.get_y() + bar.get_height() / 2,
            s=f"{width:+.2f}",
            va="center",
            ha=align,
            fontsize=9.5,
            fontweight="bold",
            color="#2c3e50",
        )

    plt.tight_layout()
    plt.show()
