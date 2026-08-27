"""
Blocking rule definitions for the entity resolution pipeline.
Implements factory pattern for extensibility.
"""

from .base import BlockingRuleBase, BlockingRuleRegistry


# ===========================================
# CONCRETE BLOCKING RULE IMPLEMENTATIONS
# ===========================================

@BlockingRuleRegistry.register("general")
class GeneralBlockingRules(BlockingRuleBase):
    """General blocking rules for prediction."""

    def get_rules(self, **kwargs) -> list:
        """
        Get general blocking rules for prediction.

        These rules are designed to generate candidate pairs for scoring
        while maintaining a balance between recall and precision.

        Returns:
            List of blocking rule strings
        """
        return [
            """l.categorical = r.categorical OR l.categorical IS NULL OR r.categorical IS NULL""",
            """strftime(l.datetime_noisy, '%Y%m') = strftime(r.datetime_noisy, '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
            """((ABS(l.numerical_noisy - r.numerical_noisy) / NULLIF(CASE WHEN l.numerical_noisy > r.numerical_noisy THEN l.numerical_noisy ELSE r.numerical_noisy END, 0)) < 0.25) OR l.numerical_noisy IS NULL OR r.numerical_noisy IS NULL""",
            """strftime(CAST(l.datetime_noisy AS TIMESTAMP) + INTERVAL '15' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
            """strftime(CAST(l.datetime_noisy AS TIMESTAMP) - INTERVAL '15' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
        ]


@BlockingRuleRegistry.register("strict")
class StrictTrainingRules(BlockingRuleBase):
    """Strict blocking rules for training/estimation."""

    def get_rules(self, **kwargs) -> list:
        """
        Get strict blocking rules for training/estimation.

        These rules are more restrictive and are used to generate
        high-confidence matches for parameter estimation.

        Returns:
            List of blocking rule strings
        """
        return [
            """l.categorical = r.categorical AND (ABS(l.numerical_noisy - r.numerical_noisy) / NULLIF(CASE WHEN l.numerical_noisy > r.numerical_noisy THEN l.numerical_noisy ELSE r.numerical_noisy END, 0)) <= 0.10 AND ABS(epoch(CAST(l.datetime_noisy AS TIMESTAMP)) - epoch(CAST(r.datetime_noisy AS TIMESTAMP))) <= 86400""",
            """l.categorical = r.categorical AND ABS(epoch(CAST(l.datetime_noisy AS TIMESTAMP)) - epoch(CAST(r.datetime_noisy AS TIMESTAMP))) <= 86400""",
        ]


@BlockingRuleRegistry.register("datetime")
class DateTimeBlockingRules(BlockingRuleBase):
    """Datetime-specific blocking rules."""

    def get_rules(self, **kwargs) -> list:
        """
        Get datetime-specific blocking rules.

        Returns:
            List of blocking rule strings focused on datetime matching
        """
        return [
            """strftime(l.datetime_noisy, '%Y%m') = strftime(r.datetime_noisy, '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
            """strftime(CAST(l.datetime_noisy AS TIMESTAMP) + INTERVAL '15' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
            """strftime(CAST(l.datetime_noisy AS TIMESTAMP) - INTERVAL '15' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
        ]


@BlockingRuleRegistry.register("column_specific")
class ColumnSpecificBlockingRules(BlockingRuleBase):
    """Blocking rules specific to a column."""

    def get_rules(self, column_name: str, rule_type: str = "general", **kwargs) -> list:
        """
        Get blocking rules specific to a column.

        Args:
            column_name: Name of the column to get blocking rules for
            rule_type: Either "general" for prediction rules or "strict" for training rules

        Returns:
            List of blocking rule strings
        """
        if column_name == "categorical":
            if rule_type == "general":
                return [
                    """l.categorical = r.categorical OR l.categorical IS NULL OR r.categorical IS NULL"""
                ]
            elif rule_type == "strict":
                return ["""l.categorical = r.categorical"""]

        elif column_name == "datetime_noisy":
            if rule_type == "general":
                return [
                    """strftime(l.datetime_noisy, '%Y%m') = strftime(r.datetime_noisy, '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
                    """strftime(CAST(l.datetime_noisy AS TIMESTAMP) + INTERVAL '15' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
                    """strftime(CAST(l.datetime_noisy AS TIMESTAMP) - INTERVAL '15' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m') OR l.datetime_noisy IS NULL OR r.datetime_noisy IS NULL""",
                ]
            elif rule_type == "strict":
                return [
                    """ABS(epoch(CAST(l.datetime_noisy AS TIMESTAMP)) - epoch(CAST(r.datetime_noisy AS TIMESTAMP))) <= 86400"""
                ]

        elif column_name == "numerical_noisy":
            if rule_type == "general":
                return [
                    """((ABS(l.numerical_noisy - r.numerical_noisy) / NULLIF(CASE WHEN l.numerical_noisy > r.numerical_noisy THEN l.numerical_noisy ELSE r.numerical_noisy END, 0)) < 0.25) OR l.numerical_noisy IS NULL OR r.numerical_noisy IS NULL""",
                    """l.numerical_noisy + 50 = r.numerical_noisy AND (l.categorical = r.categorical) AND l.categorical IS NOT NULL AND yearweek(l.datetime_noisy) = yearweek(r.datetime_noisy)""",
                    """l.numerical_noisy - 50 = r.numerical_noisy AND (l.categorical = r.categorical) AND l.categorical IS NOT NULL AND yearweek(l.datetime_noisy) = yearweek(r.datetime_noisy)""",
                ]
            elif rule_type == "strict":
                return [
                    """(ABS(l.numerical_noisy - r.numerical_noisy) / NULLIF(CASE WHEN l.numerical_noisy > r.numerical_noisy THEN l.numerical_noisy ELSE r.numerical_noisy END, 0)) <= 0.10"""
                ]

        return []


# ===========================================
# BACKWARD COMPATIBLE FUNCTIONS
# ===========================================
# These maintain backward compatibility with existing code

def get_general_blocking_rules() -> list:
    """
    Get general blocking rules for prediction.
    Backward compatible wrapper.
    """
    factory = BlockingRuleRegistry.get("general")
    return factory().get_rules()


def get_strict_training_rules() -> list:
    """
    Get strict blocking rules for training/estimation.
    Backward compatible wrapper.
    """
    factory = BlockingRuleRegistry.get("strict")
    return factory().get_rules()


def get_datetime_blocking_rules() -> list:
    """
    Get datetime-specific blocking rules.
    Backward compatible wrapper.
    """
    factory = BlockingRuleRegistry.get("datetime")
    return factory().get_rules()


def get_blocking_rules_for_column(
    column_name: str, rule_type: str = "general"
) -> list:
    """
    Get blocking rules specific to a column.
    Backward compatible wrapper.
    """
    factory = BlockingRuleRegistry.get("column_specific")
    return factory().get_rules(column_name=column_name, rule_type=rule_type)


def get_blocking_rules_from_config(config: dict) -> dict:
    """
    Extract blocking rules from configuration.
    Enhanced to use registry-based factory pattern.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with 'general' and 'strict' blocking rules
    """
    blocking_rules_config = config.get("blocking_rules", {})

    general_rules = blocking_rules_config.get("general", get_general_blocking_rules())
    strict_rules = blocking_rules_config.get("strict", get_strict_training_rules())

    # Allow custom rule types from config
    custom_rules = {}
    for rule_type, rules in blocking_rules_config.items():
        if rule_type not in ["general", "strict"]:
            try:
                factory = BlockingRuleRegistry.get(rule_type)
                custom_rules[rule_type] = factory().get_rules()
            except ValueError:
                # If not in registry, use as-is (assuming it's already a list of rules)
                custom_rules[rule_type] = rules

    result = {"general": general_rules, "strict": strict_rules}
    result.update(custom_rules)
    return result