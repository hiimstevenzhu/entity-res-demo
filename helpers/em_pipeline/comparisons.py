"""
Comparison level definitions for the entity resolution pipeline.
Implements factory pattern for extensibility.
"""

from typing import Any
from splink.comparison_library import CustomComparison
from splink.comparison_level_library import (
    ExactMatchLevel,
    NullLevel,
    ElseLevel,
    PercentageDifferenceLevel,
    JaroWinklerLevel,
)
from .base import ComparisonBase, ComparisonRegistry


# ===========================================
# CONCRETE COMPARISON IMPLEMENTATIONS
# ===========================================


@ComparisonRegistry.register("numerical")
class NumericalComparison(ComparisonBase):
    """Numerical comparison using percentage difference levels."""

    def create_comparison(
        self,
        column_name: str = "numerical_noisy",
        thresholds: list[float] | None = None,
        include_exact_match: bool = False,
    ) -> CustomComparison:
        """
        Create numerical comparison with percentage difference levels.

        Args:
            column_name: Name of the column to compare
            thresholds: List of percentage difference thresholds (0-1 scale)
            include_exact_match: Whether to include exact match level

        Returns:
            CustomComparison object
        """
        if thresholds is None:
            thresholds = [0.05, 0.15, 0.4, 0.7]

        comparison_levels = []

        # Add null level
        comparison_levels.append(NullLevel(column_name))

        # Add exact match if requested
        if include_exact_match:
            comparison_levels.append(ExactMatchLevel(column_name))

        # Add percentage difference levels in ascending order (strictest first)
        # We need to sort thresholds in descending order for the levels (strictest first)
        sorted_thresholds = sorted(thresholds, reverse=True)
        for threshold in sorted_thresholds:
            comparison_levels.append(PercentageDifferenceLevel(column_name, threshold))

        # Add else level
        comparison_levels.append(ElseLevel())

        return CustomComparison(
            output_column_name=column_name, comparison_levels=comparison_levels
        )


@ComparisonRegistry.register("datetime")
class DateTimeComparison(ComparisonBase):
    """Datetime comparison with day thresholds."""

    def create_comparison(
        self,
        column_name: str = "datetime_noisy",
        day_thresholds: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Create datetime comparison with day thresholds.

        Args:
            column_name: Name of the datetime column to compare
            day_thresholds: List of day thresholds for comparison levels

        Returns:
            Dictionary defining the comparison for Splink
        """
        if day_thresholds is None:
            day_thresholds = [0, 1, 2, 15]

        # Template for date difference condition
        within_n_days_template = "abs(date_diff('day', {column}_l, {column}_r)) <= {n}"

        comparison_levels = []

        # Add null level
        comparison_levels.append(NullLevel(column_name))

        # Add exact match (same day)
        comparison_levels.append(
            {
                "sql_condition": within_n_days_template.format(
                    column=column_name, n=day_thresholds[0]
                ),
                "label_for_charts": "Same day",
            }
        )

        # Add other day thresholds
        for i, threshold in enumerate(day_thresholds[1:], 1):
            comparison_levels.append(
                {
                    "sql_condition": within_n_days_template.format(
                        column=column_name, n=threshold
                    ),
                    "label_for_charts": f"<={threshold} days",
                }
            )

        # Add else level
        comparison_levels.append(ElseLevel())

        return {
            "output_column_name": column_name,
            "comparison_levels": comparison_levels,
            "comparison_description": f"{column_name} days apart",
        }


@ComparisonRegistry.register("categorical")
class CategoricalComparison(ComparisonBase):
    """Categorical comparison."""

    def create_comparison(self, column_name: str = "categorical") -> CustomComparison:
        """
        Create categorical comparison.

        Args:
            column_name: Name of the categorical column to compare

        Returns:
            CustomComparison object
        """
        category_levels = [
            NullLevel(column_name),
            ExactMatchLevel(column_name),
            ElseLevel(),
        ]

        return CustomComparison(
            output_column_name=column_name, comparison_levels=category_levels
        )


@ComparisonRegistry.register("text")
class TextComparison(ComparisonBase):
    """Text comparison using string similarity metrics."""

    def create_comparison(
        self, column_name: str, thresholds: list[float] | None = None
    ) -> CustomComparison:
        """
        Create text comparison using string similarity metrics.

        Args:
            column_name: Name of the text column to compare
            thresholds: List of similarity thresholds (0-1 scale, higher = more similar)

        Returns:
            CustomComparison object
        """
        if thresholds is None:
            thresholds = [0.9, 0.8, 0.7]

        # Note: This is a simplified version. For actual text similarity,
        # you would need to implement custom comparison levels using
        # Jaro-Winkler, Levenshtein, or other string similarity functions
        # via SQL conditions in the comparison levels.

        text_levels = [
            NullLevel(column_name),
            ExactMatchLevel(column_name),
        ]

        # Add similarity levels (strictest first)
        for threshold in sorted(thresholds, reverse=True):
            text_levels.append(
                {
                    "sql_condition": f"jaro_winkler_similarity(l.{column_name}, r.{column_name}) >= {threshold}",
                    "label_for_charts": f"Similarity >= {threshold}",
                }
            )

        text_levels.append(ElseLevel())

        return CustomComparison(
            output_column_name=column_name, comparison_levels=text_levels
        )


@ComparisonRegistry.register("first_name")
class FirstNameComparison(ComparisonBase):
    """First name comparison."""

    def create_comparison(
        self,
        column_name: str = "first_name_noisy",
        similarity_thresholds: list[float] | None = None,
    ) -> CustomComparison:
        """
        Create first name comparison.

        Args:
            column_name: Name of the first name column to compare
            similarity_thresholds: List of similarity thresholds (0-1 scale, higher = more similar)

        Returns:
            CustomComparison object
        """
        if similarity_thresholds is None:
            similarity_thresholds = [0.8, 0.5, 0.3]

        comparison_levels = []

        # Add null level
        comparison_levels.append(NullLevel(column_name))

        # Add exact match level
        comparison_levels.append(ExactMatchLevel(column_name))

        # Add Jaro-Winkler similarity levels (strictest first)
        for threshold in sorted(similarity_thresholds, reverse=True):
            comparison_levels.append(
                JaroWinklerLevel(column_name, distance_threshold=threshold)
            )

        comparison_levels.append(ElseLevel())

        return CustomComparison(
            output_column_name=column_name, comparison_levels=comparison_levels
        )


@ComparisonRegistry.register("last_name")
class LastNameComparison(ComparisonBase):
    """Last name comparison."""

    def create_comparison(
        self,
        column_name: str = "last_name_noisy",
        similarity_thresholds: list[float] | None = None,
    ) -> CustomComparison:
        """
        Create last name comparison.

        Args:
            column_name: Name of the last name column to compare
            similarity_thresholds: List of similarity thresholds (0-1 scale, higher = more similar)

        Returns:
            CustomComparison object
        """
        if similarity_thresholds is None:
            similarity_thresholds = [0.8, 0.5, 0.3]

        comparison_levels = []

        # Add null level
        comparison_levels.append(NullLevel(column_name))

        # Add exact match level
        comparison_levels.append(ExactMatchLevel(column_name))

        # Add Jaro-Winkler similarity levels (strictest first)
        for threshold in sorted(similarity_thresholds, reverse=True):
            comparison_levels.append(
                JaroWinklerLevel(column_name, distance_threshold=threshold)
            )

        comparison_levels.append(ElseLevel())

        return CustomComparison(
            output_column_name=column_name, comparison_levels=comparison_levels
        )


@ComparisonRegistry.register("numerical_percentile")
class NumericalPercentileComparison(ComparisonBase):
    """Numerical comparison using percentile binning.

    Assumes input data is normalized to [0,1] range where 0=min percentile, 1=max percentile.
    """

    def create_comparison(
        self,
        column_name: str = "numerical_noisy",
        bin_boundaries: list[float] | None = None,
        include_exact_match: bool = False,
    ) -> CustomComparison:
        """
        Create numerical comparison with percentile binning.
        Treats bins as categorical - exact bin match or not.

        Args:
            column_name: Name of the numerical column to compare (normalized [0,1])
            bin_boundaries: List of bin boundaries (e.g., [0.2, 0.4, 0.6, 0.8] for 5 bins)
            include_exact_match: Whether to include exact match level (deprecated, kept for API compatibility)

        Returns:
            CustomComparison object
        """
        if bin_boundaries is None:
            # Default to 5 bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
            bin_boundaries = [0.2, 0.4, 0.6, 0.8]

        # Validate and sort boundaries
        bin_boundaries = sorted([b for b in bin_boundaries if 0 < b < 1])
        bin_edges = [0.0] + bin_boundaries + [1.0]
        num_bins = len(bin_edges) - 1

        comparison_levels = []

        # Add null level
        comparison_levels.append(NullLevel(column_name))

        # Add exact match if requested (for API compatibility, though not typically used with bins)
        if include_exact_match:
            comparison_levels.append(ExactMatchLevel(column_name))

        # Add same bin level (treat bins as categorical - exact match or not)
        # Build condition: value is in bin i AND other value is in same bin i, for any bin i
        same_bin_conditions = []
        for bin_idx in range(num_bins):
            lower = bin_edges[bin_idx]
            upper = bin_edges[bin_idx + 1]
            same_bin_conditions.append(
                f"(l.{column_name} >= {lower} AND l.{column_name} < {upper} AND "
                f"r.{column_name} >= {lower} AND r.{column_name} < {upper})"
            )

        same_bin_condition = " OR ".join(same_bin_conditions)
        comparison_levels.append(
            {
                "sql_condition": same_bin_condition,
                "label_for_charts": "Same bin",
            }
        )

        # Add else level (covers different bins and all other cases)
        comparison_levels.append(ElseLevel())

        return CustomComparison(
            output_column_name=column_name, comparison_levels=comparison_levels
        )


# ===========================================
# accessible layer.
# ===========================================


def create_numerical_comparison(
    column_name: str = "numerical_noisy",
    thresholds: list[float] | None = None,
    include_exact_match: bool = False,
) -> CustomComparison:
    """
    Create numerical comparison with percentage difference levels.
    Backward compatible wrapper.
    """
    if thresholds is None:
        thresholds = [0.05, 0.15, 0.4, 0.7]

    factory = ComparisonRegistry.get("numerical")
    return factory().create_comparison(
        column_name=column_name,
        thresholds=thresholds,
        include_exact_match=include_exact_match,
    )


def create_datetime_comparison(
    column_name: str = "datetime_noisy", day_thresholds: list[int] | None = None
) -> dict[str, Any]:
    """
    Create datetime comparison with day thresholds.
    Backward compatible wrapper.
    """
    if day_thresholds is None:
        day_thresholds = [0, 1, 2, 15]

    factory = ComparisonRegistry.get("datetime")
    return factory().create_comparison(
        column_name=column_name, day_thresholds=day_thresholds
    )


def create_categorical_comparison(column_name: str = "categorical") -> CustomComparison:
    """
    Create categorical comparison.
    Backward compatible wrapper.
    """
    factory = ComparisonRegistry.get("categorical")
    return factory().create_comparison(column_name=column_name)


def create_text_comparison(
    column_name: str, thresholds: list[float] | None = None
) -> CustomComparison:
    """
    Create text comparison using string similarity metrics.
    Backward compatible wrapper.
    """
    if thresholds is None:
        thresholds = [0.9, 0.8, 0.7]

    factory = ComparisonRegistry.get("text")
    return factory().create_comparison(column_name=column_name, thresholds=thresholds)


def create_first_name_comparison(
    column_name: str = "first_name_noisy",
    similarity_thresholds: list[float] | None = None,
) -> CustomComparison:
    """
    Create first name comparison.
    Backward compatible wrapper.
    """
    if similarity_thresholds is None:
        similarity_thresholds = [0.8, 0.5, 0.3]

    factory = ComparisonRegistry.get("first_name")
    return factory().create_comparison(
        column_name=column_name, similarity_thresholds=similarity_thresholds
    )


def create_last_name_comparison(
    column_name: str = "last_name_noisy",
    similarity_thresholds: list[float] | None = None,
) -> CustomComparison:
    """
    Create last name comparison.
    Backward compatible wrapper.
    """
    if similarity_thresholds is None:
        similarity_thresholds = [0.8, 0.5, 0.3]

    factory = ComparisonRegistry.get("last_name")
    return factory().create_comparison(
        column_name=column_name, similarity_thresholds=similarity_thresholds
    )


def create_custom_comparison_from_config(
    column_name: str, comparison_config: dict[str, Any]
) -> Any:
    """
    Create a comparison from configuration dictionary.
    Enhanced to use registry-based factory pattern.

    Args:
        column_name: Name of the column to compare
        comparison_config: Configuration dictionary specifying comparison type and parameters

    Returns:
        Comparison object suitable for Splink
    """
    comparison_type = comparison_config.get("type", "exact_match")

    # Map config types to registry types
    type_mapping = {
        "percentage_difference": "numerical",
        "datetime_thresholds": "datetime",
        "exact_match": "categorical",
        "text_similarity": "text",
        "first_name_similarity": "first_name",
        "last_name_similarity": "last_name",
        "numerical_percentile": "numerical_percentile",
    }

    registry_type = type_mapping.get(comparison_type, comparison_type)

    if registry_type not in ComparisonRegistry.list_types():
        raise ValueError(f"Unsupported comparison type: {comparison_type}")

    factory_class = ComparisonRegistry.get(registry_type)
    factory = factory_class()

    # Pass all config parameters to the create_comparison method
    # Remove 'type' from kwargs as it's used for lookup
    kwargs = {k: v for k, v in comparison_config.items() if k != "type"}
    kwargs["column_name"] = column_name

    return factory.create_comparison(**kwargs)


@ComparisonRegistry.register("numerical_percentile")
class NumericalPercentileComparison(ComparisonBase):
    """Numerical comparison using percentile binning.

    Assumes input data is normalized to [0,1] range where 0=min percentile, 1=max percentile.
    """

    def create_comparison(
        self,
        column_name: str = "numerical_noisy",
        bin_boundaries: list[float] | None = None,
        include_exact_match: bool = False,
    ) -> CustomComparison:
        """
        Create numerical comparison with percentile binning.

        Args:
            column_name: Name of the numerical column to compare (normalized [0,1])
            bin_boundaries: List of bin boundaries (e.g., [0.2, 0.4, 0.6, 0.8] for 5 bins)
            include_exact_match: Whether to include exact match level

        Returns:
            CustomComparison object
        """
        if bin_boundaries is None:
            # Default to 5 bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
            bin_boundaries = [0.2, 0.4, 0.6, 0.8]

        # Validate and sort boundaries
        bin_boundaries = sorted([b for b in bin_boundaries if 0 < b < 1])
        bin_edges = [0.0] + bin_boundaries + [1.0]
        num_bins = len(bin_edges) - 1

        comparison_levels = []

        # Add null level
        comparison_levels.append(NullLevel(column_name))

        # Add exact match if requested
        if include_exact_match:
            comparison_levels.append(ExactMatchLevel(column_name))

        # Add same bin levels
        for bin_idx in range(num_bins):
            lower = bin_edges[bin_idx]
            upper = bin_edges[bin_idx + 1]
            comparison_levels.append(
                {
                    "sql_condition": f"""
                    ({column_name}_l >= {lower} AND {column_name}_l < {upper}) AND
                    ({column_name}_r >= {lower} AND {column_name}_r < {upper})
                """.strip(),
                    "label_for_charts": f"Same bin [{lower:.0%}, {upper:.0%})",
                }
            )

        # Add adjacent bin levels (one bin apart)
        for bin_idx in range(num_bins - 1):
            lower1, upper1 = bin_edges[bin_idx], bin_edges[bin_idx + 1]
            lower2, upper2 = bin_edges[bin_idx + 1], bin_edges[bin_idx + 2]

            comparison_levels.append(
                {
                    "sql_condition": f"""
                    (({column_name}_l >= {lower1} AND {column_name}_l < {upper1}) AND
                    ({column_name}_r >= {lower2} AND {column_name}_r < {upper2})) OR
                    (({column_name}_l >= {lower2} AND {column_name}_l < {upper2}) AND
                    ({column_name}_r >= {lower1} AND {column_name}_r < {upper1}))
                """.strip(),
                    "label_for_charts": f"Adjacent bins [{lower1:.0%}-{upper1:.0%}) and [{lower2:.0%}-{upper2:.0%})",
                }
            )

        # Add else level (covers non-adjacent bins and all other cases)
        comparison_levels.append(ElseLevel())

        return CustomComparison(
            output_column_name=column_name, comparison_levels=comparison_levels
        )


def create_numerical_percentile_comparison(
    column_name: str = "numerical_noisy",
    bin_boundaries: list[float] | None = None,
    include_exact_match: bool = False,
) -> CustomComparison:
    """
    Create numerical comparison with percentile binning.
    Backward compatible wrapper.
    """
    if bin_boundaries is None:
        bin_boundaries = [0.2, 0.4, 0.6, 0.8]

    factory = ComparisonRegistry.get("numerical_percentile")
    return factory().create_comparison(
        column_name=column_name,
        bin_boundaries=bin_boundaries,
        include_exact_match=include_exact_match,
    )
