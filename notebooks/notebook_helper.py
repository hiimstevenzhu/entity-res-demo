import os
import sys
# Determine notebook directory: if __file__ is defined (running as script), use its dir;
# otherwise assume we are in an interactive notebook and use current working directory.
if "__file__" in globals():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
else:
    notebook_dir = os.getcwd()
# Add project root (parent of notebooks) and helpers directory to sys.path
project_root = os.path.dirname(notebook_dir)
helpers_dir = os.path.join(project_root, 'helpers')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import aggregation helper functions
from helpers.entity_resolution_aggregation import (
    calc_noisy_or_probs,
    calc_softmax_prob,
    calc_top_k_mean_weight,
    compute_source_sizes,
    calculate_match_density,
    calculate_weighted_jaccard,
    calculate_weighted_overlap,
    calculate_score_prob_ratio,
    aggregate_entity_pair_matched_pairs,
    second_layer_splink_evaluation
)


def setup_notebook_environment(notebook_dir=None):
    """
    Set up the notebook environment: configure paths and ensure requirements are met.
    This function prepares the environment for running entity resolution notebooks.

    After calling this function:
    - The project root is added to sys.path (allows importing notebooks.helper)
    - The helpers directory is added to sys.path (allows importing em_pipeline.*)
    - Dataset paths can be resolved using get_dataset_path()
    - Datasets will be automatically generated if missing

    Returns:
        Simple object with:
            - get_dataset_path: function to resolve dataset paths by friendly name
            - ensure_dataset_exists: function to check/generate dataset if missing
    """
    if notebook_dir is None:
        # Get the directory of the currently running notebook/script
        notebook_dir = (
            os.path.dirname(os.path.abspath("__file__"))
            if "__file__" in globals()
            else os.getcwd()
        )

    # ADD PROJECT ROOT TO SYS.PATH
    # This makes "notebooks.notebook_helper" importable
    # Assuming notebook_dir is .../notebooks, we add .../ (project root)
    project_root = (
        os.path.dirname(notebook_dir)
        if notebook_dir.endswith("notebooks")
        else notebook_dir
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # ADD HELPERS DIRECTORY TO SYS.PATH
    # This allows importing em_pipeline.* as top-level modules
    # Structure: project_root/helpers/em_pipeline/ -> after adding helpers to path,
    #            we can import em_pipeline (which resolves to helpers/em_pipeline)
    helpers_dir = os.path.join(project_root, "helpers")
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)

    # Load configuration from notebooks directory
    config_path = os.path.join(notebook_dir, "notebook_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Prepare dataset path resolver (data_dir is relative to notebooks dir per config)
    data_dir = os.path.join(notebook_dir, config["data_dir"])
    dataset_mapping = config["dataset_mapping"]

    def get_dataset_path(pair_name, dataset_letter):
        """Resolve the full path to a dataset file."""
        filename = dataset_mapping[pair_name][dataset_letter]
        return os.path.join(data_dir, f"{filename}.parquet")

    def ensure_dataset_exists(pair_name):
        """
        Check if dataset pair exists, and generate it if missing.

        Args:
            pair_name: Friendly name for the dataset pair (e.g., 'smaller')

        Returns:
            tuple: (path_to_dataset_a, path_to_dataset_b)
        """
        # Get paths for both datasets
        path_a = get_dataset_path(pair_name, "a")
        path_b = get_dataset_path(pair_name, "b")

        # Check if both files exist
        if os.path.exists(path_a) and os.path.exists(path_b):
            return path_a, path_b

        # If not, try to generate the dataset
        print(f"Dataset pair '{pair_name}' not found. Attempting to generate...")

        try:
            # Import the generate function (avoid circular imports)
            # We need to import from the project root
            sys.path.insert(0, project_root)
            from generate_data import generate_dataset

            # Generate the dataset
            generated_path_a, generated_path_b = generate_dataset(pair_name)

            # Verify the generated files exist
            if os.path.exists(generated_path_a) and os.path.exists(generated_path_b):
                print(f"Successfully generated dataset pair '{pair_name}'")
                return generated_path_a, generated_path_b
            else:
                raise FileNotFoundError("Generated files not found")

        except Exception as e:
            print(f"Failed to generate dataset pair '{pair_name}': {e}")
            print("Please ensure you have the required dependencies installed.")
            print(
                "You can manually run: python generate_data.py (after setting the size)"
            )
            raise

    # Return object with path resolution and dataset assurance
    class Helper:
        def __init__(self, get_dataset_path, ensure_dataset_exists):
            self.get_dataset_path = get_dataset_path
            self.ensure_dataset_exists = ensure_dataset_exists

    return Helper(get_dataset_path, ensure_dataset_exists)


def ensure_image_dir(notebook_name):
    """
    Ensure the image directory exists for a given notebook.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)

    Returns:
        Path to the image directory for this notebook
    """
    # Get the notebooks directory
    notebooks_dir = os.path.dirname(os.path.abspath(__file__))
    # Create images directory if it doesn't exist
    images_dir = os.path.join(notebooks_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    # Create notebook-specific subdirectory
    notebook_images_dir = os.path.join(images_dir, notebook_name)
    os.makedirs(notebook_images_dir, exist_ok=True)
    return notebook_images_dir


def save_chart(chart, notebook_name, chart_name, dpi=150, bbox_inches='tight', format='png'):
    """
    Save a chart object (Altair, matplotlib, etc.) to the notebook's image directory.

    Args:
        chart: The chart object to save (Altair chart, matplotlib figure, etc.)
        notebook_name: Name of the notebook (without .ipynb extension)
        chart_name: Name for the saved chart file (without extension)
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image
        format: Image format ('png', 'jpg', 'svg', etc.)

    Returns:
        Path to the saved image file
    """
    # Ensure image directory exists
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, f"{chart_name}.{format}")

    # Save the chart based on its type
    if hasattr(chart, 'save'):
        # Altair/Vega-Lite charts have a save method
        try:
            chart.save(img_path, format=format, scale_factor=dpi/100)
        except ValueError as e:
            if "vl-convert" in str(e):
                # Handle vl-convert errors for PNG, SVG, PDF formats
                if format.lower() == 'png':
                    # Try SVG first for PNG fallback
                    try:
                        svg_path = img_path.replace('.png', '.svg')
                        chart.save(svg_path, format='svg', scale_factor=dpi/100)
                        print(f"Warning: vl-convert-python not available for PNG. Saved as SVG instead: {svg_path}")
                        return svg_path
                    except ValueError as e2:
                        if "vl-convert" in str(e2):
                            # SVG also failed, fall back to HTML
                            html_path = img_path.replace('.png', '.html')
                            chart.save(html_path, format='html')
                            print(f"Warning: vl-convert-python not available for PNG or SVG. Saved as HTML instead: {html_path}")
                            return html_path
                        else:
                            raise
                elif format.lower() == 'svg':
                    # Fall back to HTML for SVG
                    html_path = img_path.replace('.svg', '.html')
                    chart.save(html_path, format='html')
                    print(f"Warning: vl-convert-python not available for SVG. Saved as HTML instead: {html_path}")
                    return html_path
                elif format.lower() == 'pdf':
                    # Try SVG first for PDF fallback, then HTML
                    try:
                        svg_path = img_path.replace('.pdf', '.svg')
                        chart.save(svg_path, format='svg', scale_factor=dpi/100)
                        print(f"Warning: vl-convert-python not available for PDF. Saved as SVG instead: {svg_path}")
                        return svg_path
                    except ValueError as e2:
                        if "vl-convert" in str(e2):
                            # SVG also failed, fall back to HTML
                            html_path = img_path.replace('.pdf', '.html')
                            chart.save(html_path, format='html')
                            print(f"Warning: vl-convert-python not available for PDF or SVG. Saved as HTML instead: {html_path}")
                            return html_path
                        else:
                            raise
                else:
                    # For other formats, just re-raise
                    raise
            else:
                raise
    elif hasattr(chart, 'figure') and hasattr(chart.figure, 'savefig'):
        # Some chart objects might have a figure attribute
        chart.figure.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches, format=format)
    elif hasattr(chart, 'savefig'):
        # Matplotlib figures
        chart.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches, format=format)
    else:
        # Try to treat it as a matplotlib figure
        plt.figure(chart)
        plt.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches, format=format)
        plt.close()

    return img_path


def save_match_weights_chart(notebook_name, linker, dpi=150, bbox_inches='tight'):
    """
    Save the match weights chart from a Splink linker.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image

    Returns:
        Path to the saved image file
    """
    chart = linker.visualisations.match_weights_chart()
    return save_chart(chart, notebook_name, "match_weights_chart", dpi, bbox_inches)


def save_match_weight_distribution(notebook_name, linker, dpi=150, bbox_inches='tight'):
    """
    Save the match weight distribution chart from a Splink linker.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image

    Returns:
        Path to the saved image file
    """
    chart = linker.visualisations.match_weight_distribution()
    return save_chart(chart, notebook_name, "match_weight_distribution", dpi, bbox_inches)


def save_threshold_curve(notebook_name, linker, df_predict, ground_truth_col, dpi=150, bbox_inches='tight'):
    """
    Save the threshold selection curve (precision, recall, F1) from Splink predictions.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object
        df_predict: Pandas DataFrame with prediction results (or Splink Tbl object)
        ground_truth_col: Column name containing ground truth IDs for evaluation
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image

    Returns:
        Path to the saved image file
    """
    # Ensure we're working with a pandas DataFrame
    # If df_predict is a Splink Tbl object, convert it to pandas
    if hasattr(df_predict, 'as_pandas_dataframe'):
        df_predict = df_predict.as_pandas_dataframe()

    # Flag true matches inside the scored pool
    df_predict["is_true_match"] = (
        df_predict[ground_truth_col + "_l"] == df_predict[ground_truth_col + "_r"]
    ).astype(int)

    # Get the total true match ceiling via a fast 1-to-1 merge count
    # This catches matches that were blocked, ensuring Recall remains accurate
    total_true_matches = (
        df_predict[[ground_truth_col + "_l"]]
        .merge(df_predict[[ground_truth_col + "_r"]], left_on=ground_truth_col + "_l", right_on=ground_truth_col + "_r")
        .shape[0]
    )

    # Generate metrics across thresholds (0.0 to 1.0)
    thresholds = np.linspace(0.0, 1.0, 101)
    curve_data = []

    for t in thresholds:
        # Filter predictions at current probability threshold
        pred_positive = df_predict["match_probability"] >= t

        tp = ((pred_positive) & (df_predict["is_true_match"] == 1)).sum()
        fp = ((pred_positive) & (df_predict["is_true_match"] == 0)).sum()

        # Calculate metrics with standard division-by-zero safeguards
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_true_matches if total_true_matches > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        curve_data.append(
            {"threshold": t, "precision": precision, "recall": recall, "f1_score": f1}
        )

    df_curve = pd.DataFrame(curve_data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_curve["threshold"], df_curve["precision"], label="Precision", color="blue", lw=2)
    plt.plot(df_curve["threshold"], df_curve["recall"], label="Recall", color="orange", lw=2)
    plt.plot(df_curve["threshold"], df_curve["f1_score"], label="F1-Score", color="green", linestyle="--")

    plt.title("Custom Model Evaluation Curve (By Match Probability)")
    plt.xlabel("Match Probability Threshold")
    plt.ylabel("Score")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, "threshold_curve.png")
    plt.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

    return img_path


def save_accuracy_analysis_chart(notebook_name, linker, ground_truth_col, dpi=150, bbox_inches='tight'):
    """
    Save the accuracy analysis chart from Splink evaluation.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object
        ground_truth_col: Column name containing ground truth IDs for evaluation
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image

    Returns:
        Path to the saved image file
    """
    chart = linker.evaluation.accuracy_analysis_from_labels_column(
        ground_truth_col,
        output_type="threshold_selection",
        add_metrics=["f1", "accuracy"]
    )
    return save_chart(chart, notebook_name, "accuracy_analysis", dpi, bbox_inches)


def save_unlinkables_chart(notebook_name, linker, dpi=150, bbox_inches='tight'):
    """
    Save the unlinkables chart from Splink evaluation.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image

    Returns:
        Path to the saved image file
    """
    chart = linker.evaluation.unlinkables_chart()
    return save_chart(chart, notebook_name, "unlinkables_chart", dpi, bbox_inches)


def save_blocking_analysis_chart(notebook_name, df1, df2, blocking_rules, db_api, link_type="link_only", unique_id_column_name="row_id", dpi=150, bbox_inches='tight'):
    """
    Save the blocking analysis chart from Splink.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        df1, df2: Input dataframes
        blocking_rules: List of blocking rules
        db_api: Splink DBCAPI object
        link_type: Type of linkage ("link_only" or "dedupe_only")
        unique_id_column_name: Column name for unique IDs
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image

    Returns:
        Path to the saved image file
    """
    # Ensure image directory exists
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, "blocking_analysis.png")

    # Import the blocking analysis function
    from splink.blocking_analysis import (
        cumulative_comparisons_to_be_scored_from_blocking_rules_chart
    )

    # Generate and save the chart
    chart = cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
        table_or_tables=[df1, df2],
        blocking_rules=blocking_rules,
        db_api=db_api,
        link_type=link_type,
        unique_id_column_name=unique_id_column_name
    )
    # Save the chart - assuming it's an Altair/Vega-Lite chart
    chart.save(img_path, scale_factor=dpi/100)  # Altair uses scale_factor for DPI

    return img_path


# NEW "WRAPPER" FUNCTIONS - These directly wrap the specific function calls to save images
# as requested by the user: "wrap these up to save the image when we call that specific function"

def save_match_weights_chart_wrapper(notebook_name, linker, chart_name="match_weights_chart", dpi=150):
    """
    Wrapper that calls the match weights chart function and saves the result.
    This is the simplest wrapping approach - just call this instead of the original method
    if you want to automatically save the chart.

    Usage:
        # Instead of: chart = linker.visualisations.match_weights_chart()
        # Use: chart = save_match_weights_chart_wrapper(notebook_name, linker)
        # Or: save_match_weights_chart_wrapper(notebook_name, linker, "custom_name")

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object
        chart_name: Name for the saved chart file (default: "match_weights_chart")
        dpi: Resolution for saved image

    Returns:
        The chart object (so you can still use it if needed)
    """
    chart = linker.visualisations.match_weights_chart()
    save_chart(chart, notebook_name, chart_name, dpi)
    return chart


def save_match_weight_distribution_wrapper(notebook_name, linker, chart_name="match_weight_distribution", dpi=150):
    """
    Wrapper that calls the match weight distribution function and saves the result.

    Usage:
        chart = save_match_weight_distribution_wrapper(notebook_name, linker)
    """
    chart = linker.visualisations.match_weight_distribution()
    save_chart(chart, notebook_name, chart_name, dpi)
    return chart


def save_threshold_curve_wrapper(notebook_name, linker, df_predict, ground_truth_col,
                                chart_name="threshold_curve", dpi=150):
    """
    Wrapper that generates and saves the threshold selection curve.

    Usage:
        save_threshold_curve_wrapper(notebook_name, linker, df_predict, ground_truth_col)
    """
    # Flag true matches inside the scored pool
    df_predict["is_true_match"] = (
        df_predict[ground_truth_col + "_l"] == df_predict[ground_truth_col + "_r"]
    ).astype(int)

    # Get the total true match ceiling via a fast 1-to-1 merge count
    total_true_matches = (
        df_predict[[ground_truth_col + "_l"]]
        .merge(df_predict[[ground_truth_col + "_r"]], left_on=ground_truth_col + "_l", right_on=ground_truth_col + "_r")
        .shape[0]
    )

    # Generate metrics across thresholds (0.0 to 1.0)
    thresholds = np.linspace(0.0, 1.0, 101)
    curve_data = []

    for t in thresholds:
        # Filter predictions at current probability threshold
        pred_positive = df_predict["match_probability"] >= t

        tp = ((pred_positive) & (df_predict["is_true_match"] == 1)).sum()
        fp = ((pred_positive) & (df_predict["is_true_match"] == 0)).sum()

        # Calculate metrics with standard division-by-zero safeguards
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_true_matches if total_true_matches > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        curve_data.append(
            {"threshold": t, "precision": precision, "recall": recall, "f1_score": f1}
        )

    df_curve = pd.DataFrame(curve_data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_curve["threshold"], df_curve["precision"], label="Precision", color="blue", lw=2)
    plt.plot(df_curve["threshold"], df_curve["recall"], label="Recall", color="orange", lw=2)
    plt.plot(df_curve["threshold"], df_curve["f1_score"], label="F1-Score", color="green", linestyle="--")

    plt.title("Custom Model Evaluation Curve (By Match Probability)")
    plt.xlabel("Match Probability Threshold")
    plt.ylabel("Score")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, f"{chart_name}.png")
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return img_path


def save_accuracy_analysis_chart_wrapper(notebook_name, linker, ground_truth_col,
                                       chart_name="accuracy_analysis", dpi=150):
    """
    Wrapper that calls the accuracy analysis function and saves the result.

    Usage:
        save_accuracy_analysis_chart_wrapper(notebook_name, linker, ground_truth_col)
    """
    chart = linker.evaluation.accuracy_analysis_from_labels_column(
        ground_truth_col,
        output_type="threshold_selection",
        add_metrics=["f1", "accuracy"]
    )
    save_chart(chart, notebook_name, chart_name, dpi)
    return chart


def save_unlinkables_chart_wrapper(notebook_name, linker, chart_name="unlinkables_chart", dpi=150):
    """
    Wrapper that calls the unlinkables chart function and saves the result.

    Usage:
        save_unlinkables_chart_wrapper(notebook_name, linker)
    """
    chart = linker.evaluation.unlinkables_chart()
    save_chart(chart, notebook_name, chart_name, dpi)
    return chart


def save_blocking_analysis_chart_wrapper(notebook_name, df1, df2, blocking_rules, db_api,
                                        link_type="link_only", unique_id_column_name="row_id",
                                        chart_name="blocking_analysis", dpi=150):
    """
    Wrapper that calls the blocking analysis chart function and saves the result.

    Usage:
        save_blocking_analysis_chart_wrapper(notebook_name, df1, df2, blocking_rules, db_api)
    """
    # Import the blocking analysis function
    from splink.blocking_analysis import (
        cumulative_comparisons_to_be_scored_from_blocking_rules_chart
    )

    # Generate and save the chart
    chart = cumulative_comparisons_to_be_scored_from_blocking_rules_chart(
        table_or_tables=[df1, df2],
        blocking_rules=blocking_rules,
        db_api=db_api,
        link_type=link_type,
        unique_id_column_name=unique_id_column_name
    )
    # Save the chart - assuming it's an Altair/Vega-Lite chart
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, f"{chart_name}.png")
    chart.save(img_path, scale_factor=dpi/100)  # Altair uses scale_factor for DPI

    return img_path


def save_aggregation_heatmap(notebook_name, similarity_matrix, row_labels, col_labels,
                           title="Aggregated Similarity Heatmap",
                           xlabel="Column Entities", ylabel="Row Entities",
                           dpi=150, bbox_inches='tight', cmap='vlag', chart_name="aggregation_heatmap"):
    """
    Save a heatmap visualization of aggregated similarity scores.
    Useful for visualizing global aggregation results.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        similarity_matrix: 2D numpy array of similarity scores
        row_labels: Labels for rows (entities from first dataset)
        col_labels: Labels for columns (entities from second dataset)
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image
        cmap: Colormap for the heatmap
        chart_name: Name for the saved chart file

    Returns:
        Path to the saved image file
    """
    # Ensure image directory exists
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, f"{chart_name}.png")

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        cbar_kws={'label': 'Similarity Score'}
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Save the figure
    plt.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

    return img_path


def save_score_distribution_plot(notebook_name, scores, title="Distribution of Match Scores",
                               xlabel="Match Score", ylabel="Density",
                               dpi=150, bbox_inches='tight', bins=50, chart_name="score_distribution"):
    """
    Save a distribution plot of match scores.
    Useful for visualizing the output of local or global matching.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        scores: Array-like of match scores
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image
        bins: Number of bins for histogram
        chart_name: Name for the saved chart file

    Returns:
        Path to the saved image file
    """
    # Ensure image directory exists
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, f"{chart_name}.png")

    # Create the distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

    return img_path


def save_precision_recall_curve(notebook_name, precision_vals, recall_vals,
                              title="Precision-Recall Curve",
                              dpi=150, bbox_inches='tight', chart_name="precision_recall_curve"):
    """
    Save a precision-recall curve.
    Useful for evaluating matching performance across thresholds.

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        precision_vals: Array-like of precision values
        recall_vals: Array-like of recall values
        title: Title for the plot
        dpi: Resolution for saved image
        bbox_inches: Bounding box for saved image
        chart_name: Name for the saved chart file

    Returns:
        Path to the saved image file
    """
    # Ensure image directory exists
    img_dir = ensure_image_dir(notebook_name)
    img_path = os.path.join(img_dir, f"{chart_name}.png")

    # Create the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, 'b-', linewidth=2)
    plt.fill_between(recall_vals, precision_vals, alpha=0.2)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

    return img_path


# Classes for chaining visualization calls with immediate saving
class SavableChart:
    """
    A wrapper around chart objects (Altair, matplotlib, etc.) that adds a save_image method
    for easy saving to notebook-specific image directories.

    This allows chaining like:
        linker.visualisations.match_weights_chart().save_image("match_weights_chart")
    when used with get_savable_visualisations().
    """

    def __init__(self, chart, notebook_name):
        """
        Initialize a SavableChart.

        Args:
            chart: The chart object to wrap (Altair chart, matplotlib figure, etc.)
            notebook_name: Name of the notebook (without .ipynb extension)
        """
        self.chart = chart
        self.notebook_name = notebook_name

    def save_image(self, chart_name, dpi=150, format='png', bbox_inches='tight'):
        """
        Save the chart to the notebook's image directory.

        Args:
            chart_name: Name for the saved chart file (without extension)
            dpi: Resolution for saved image
            format: Image format ('png', 'jpg', 'svg', 'pdf', etc.)
            bbox_inches: Bounding box for saved image

        Returns:
            Path to the saved image file
        """
        # Ensure image directory exists
        img_dir = ensure_image_dir(self.notebook_name)
        img_path = os.path.join(img_dir, f"{chart_name}.{format}")

        # Save the chart based on its type
        if hasattr(self.chart, 'save'):
            # Altair/Vega-Lite charts have a save method
            try:
                self.chart.save(img_path, format=format, scale_factor=dpi/100)
            except ValueError as e:
                if "vl-convert" in str(e):
                    # Handle vl-convert errors for PNG, SVG, PDF formats
                    if format.lower() == 'png':
                        # Try SVG first for PNG fallback
                        try:
                            svg_path = img_path.replace('.png', '.svg')
                            self.chart.save(svg_path, format='svg', scale_factor=dpi/100)
                            print(f"Warning: vl-convert-python not available for PNG. Saved as SVG instead: {svg_path}")
                            return svg_path
                        except ValueError as e2:
                            if "vl-convert" in str(e2):
                                # SVG also failed, fall back to HTML
                                html_path = img_path.replace('.png', '.html')
                                self.chart.save(html_path, format='html')
                                print(f"Warning: vl-convert-python not available for PNG or SVG. Saved as HTML instead: {html_path}")
                                return html_path
                            else:
                                raise
                    elif format.lower() == 'svg':
                        # Fall back to HTML for SVG
                        html_path = img_path.replace('.svg', '.html')
                        self.chart.save(html_path, format='html')
                        print(f"Warning: vl-convert-python not available for SVG. Saved as HTML instead: {html_path}")
                        return html_path
                    elif format.lower() == 'pdf':
                        # Try SVG first for PDF fallback, then HTML
                        try:
                            svg_path = img_path.replace('.pdf', '.svg')
                            self.chart.save(svg_path, format='svg', scale_factor=dpi/100)
                            print(f"Warning: vl-convert-python not available for PDF. Saved as SVG instead: {svg_path}")
                            return svg_path
                        except ValueError as e2:
                            if "vl-convert" in str(e2):
                                # SVG also failed, fall back to HTML
                                html_path = img_path.replace('.pdf', '.html')
                                self.chart.save(html_path, format='html')
                                print(f"Warning: vl-convert-python not available for PDF or SVG. Saved as HTML instead: {html_path}")
                                return html_path
                            else:
                                raise
                    else:
                        # For other formats, just re-raise
                        raise
                else:
                    raise
        elif hasattr(self.chart, 'figure') and hasattr(self.chart.figure, 'savefig'):
            # Some chart objects might have a figure attribute
            self.chart.figure.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches, format=format)
        elif hasattr(self.chart, 'savefig'):
            # Matplotlib figures
            self.chart.savefig(img_path, dpi=dpi, bbox_inches=bbox_inches, format=format)
        else:
            # Try to treat it as a matplotlib figure by importing pyplot
            import matplotlib.pyplot as plt
            # If the chart is already a plt figure, we might need to get it differently
            # For now, we'll assume it's compatible or the user can use save_chart directly
            raise TypeError(f"Unsupported chart type: {type(self.chart)}. Use save_chart function instead.")

        return img_path


class SavableVisualisations:
    """
    A wrapper around linker.visualisations and linker.evaluation that returns
    SavableChart objects instead of raw chart objects, enabling chaining with .save_image().

    Usage:
        viz = get_savable_visualisations(notebook_name, linker)
        viz.match_weights_chart().save_image("match_weights_chart")
        viz.accuracy_analysis_from_labels_column("latent_event_id").save_image("accuracy_analysis")
    """

    def __init__(self, notebook_name, linker):
        """
        Initialize a SavableVisualisations wrapper.

        Args:
            notebook_name: Name of the notebook (without .ipynb extension)
            linker: Splink Linker object
        """
        self.notebook_name = notebook_name
        self.linker = linker

    def __getattr__(self, name):
        """
        Dynamically wrap methods from linker.visualisations and linker.evaluation
        to return SavableChart objects.

        This allows accessing visualization and evaluation methods and getting
        a SavableChart back instead of the raw chart.
        """
        # Try linker.visualisations first
        if hasattr(self.linker.visualisations, name):
            original_method = getattr(self.linker.visualisations, name)
        # Then try linker.evaluation
        elif hasattr(self.linker.evaluation, name):
            original_method = getattr(self.linker.evaluation, name)
        else:
            # If not found in either, raise AttributeError
            raise AttributeError(f"'{type(self.linker).__name__}' object has no attribute '{name}'")

        # Define a wrapper that calls the original method and wraps the result
        def wrapped_method(*args, **kwargs):
            chart = original_method(*args, **kwargs)
            return SavableChart(chart, self.notebook_name)

        return wrapped_method


def get_savable_visualisations(notebook_name, linker):
    """
    Get a wrapper around linker.visualisations that returns savable charts.

    This enables chaining visualization calls with immediate saving:
        get_savable_visualisations(notebook_name, linker).match_weights_chart().save_image("match_weights_chart")

    Args:
        notebook_name: Name of the notebook (without .ipynb extension)
        linker: Splink Linker object

    Returns:
        SavableVisualisations object
    """
    return SavableVisualisations(notebook_name, linker)