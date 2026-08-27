"""
Basic package for functions relevant to splink, abstractable.
Assumes we use duckdb.
"""

import duckdb
from splink import DuckDBAPI, block_on


def build_api_with_con(mem_limit: str, temp_dir: str):
    """
    Sets up a duckdb connection with these specific settings.
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{mem_limit}';")
    con.execute(f"SET temp_directory = '{temp_dir}';")
    db_api = DuckDBAPI(connection=con)
    return db_api


# TODO(): The idea behind this function is to build a lookup for different blocking rules that might be applicable, but
# in this case we really just care about the same set of rules, so for now we'll keep it as this.
# Build a lookup table for this when necessary.
def get_blocking_rules():
    """
    Returns a set of blocking rules as a list.
    """
    blocking_rule_category = """
        l.categorical = r.categorical 
    """

    # Rule 3: Robust percentage-based bounding OR allow if either side is missing
    blocking_rule_numerical = """
        (
            (ABS(l.numerical_noisy - r.numerical_noisy) / NULLIF(
                CASE WHEN l.numerical_noisy > r.numerical_noisy THEN l.numerical_noisy ELSE r.numerical_noisy END, 
                0
            )) < 0.25
        )
    """

    # Rule 4: Handle shifting date windows cleanly OR allow if either side is missing
    blocking_rule_date_shift_1 = """
        strftime(CAST(l.datetime_noisy AS TIMESTAMP) + INTERVAL '1' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m')
    """

    blocking_rule_date_shift_2 = """
        strftime(CAST(l.datetime_noisy AS TIMESTAMP) - INTERVAL '1' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m')
    """

    # Combined array for your settings dictionary
    brs = [
        blocking_rule_category,
        blocking_rule_numerical,
        blocking_rule_date_shift_1,
        blocking_rule_date_shift_2,
    ]

    return brs


def get_non_numerical_blocking_rules():
    """
    Returns a set of blocking rules as a list.
    """
    blocking_rule_category = """
        l.categorical = r.categorical 
    """

    # Rule 4: Handle shifting date windows cleanly OR allow if either side is missing
    blocking_rule_date_shift_1 = """
        strftime(CAST(l.datetime_noisy AS TIMESTAMP) + INTERVAL '1' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m')
    """

    blocking_rule_date_shift_2 = """
        strftime(CAST(l.datetime_noisy AS TIMESTAMP) - INTERVAL '1' DAY, '%Y%m') = strftime(CAST(r.datetime_noisy AS TIMESTAMP), '%Y%m')
    """

    # Combined array for your settings dictionary
    brs = [
        blocking_rule_category,
        blocking_rule_date_shift_1,
        blocking_rule_date_shift_2,
    ]

    return brs


def get_estimating_blocking_rules():
    strict_rule_category = "l.categorical = r.categorical"

    # Rule 2: Numbers are within 10% difference of each other (Symmetric, No Left/Right Bias)
    strict_rule_numerical = """
        (ABS(l.numerical_noisy - r.numerical_noisy) / NULLIF(
            CASE WHEN l.numerical_noisy > r.numerical_noisy THEN l.numerical_noisy ELSE r.numerical_noisy END, 
            0
        )) <= 0.10
    """

    # Rule 3: Datetime is within 1 day (24 hours) of each other
    strict_rule_datetime = """
        CAST(l.datetime_noisy AS DATE) = CAST(r.datetime_noisy AS DATE)
    """

    full = (
        strict_rule_category
        + " AND "
        + strict_rule_datetime
        + " AND "
        + strict_rule_numerical
    )

    return full


def get_non_numerical_estimating_blocking_rules():
    strict_rule_category = "l.categorical = r.categorical"

    # Rule 3: Datetime is within 1 day (24 hours) of each other
    strict_rule_datetime = """
        CAST(l.datetime_noisy AS DATE) = CAST(r.datetime_noisy AS DATE)
    """

    full = strict_rule_category + " AND " + strict_rule_datetime

    return full


import splink.comparison_level_library as cll
import splink.comparison_library as cl


def get_numerical_comparison(num_col: str):
    comparison_levels = [
        cll.NullLevel(num_col),
        cll.PercentageDifferenceLevel(num_col, 0.05),
        cll.PercentageDifferenceLevel(num_col, 0.10),
        cll.PercentageDifferenceLevel(num_col, 0.25),
        cll.PercentageDifferenceLevel(num_col, 0.50),
        cll.ElseLevel(),
    ]

    # 2. Wrap them into a CustomComparison creator
    numerical_noisy_comparison = cl.CustomComparison(
        output_column_name=num_col,
        comparison_levels=comparison_levels,  # Pass the list here
    )

    return numerical_noisy_comparison


def get_datetime_comparison(dt_col):
    base_template = (
        f"abs(datediff('day', CAST({dt_col}_l AS DATE), CAST({dt_col}_r AS DATE)))"
    )
    within_n_days_template = base_template + " <= {n}"

    comparison_date = {
        "output_column_name": dt_col,
        "comparison_levels": [
            cll.NullLevel(dt_col),
            {
                "sql_condition": within_n_days_template.format(n=0),
                "label_for_charts": "Same day",
            },
            {
                "sql_condition": within_n_days_template.format(n=1),
                "label_for_charts": "<=1 day",
            },
            {
                "sql_condition": within_n_days_template.format(n=5),
                "label_for_charts": "<=5 days",
            },
            cll.ElseLevel(),
        ],
        "comparison_description": "Datetime days apart",
    }

    return comparison_date


def get_category_comparison(cat_col: str):
    category_levels = [
        cll.NullLevel(cat_col),
        cll.ExactMatchLevel(cat_col),
        cll.ElseLevel(),
    ]

    category_comparison = cl.CustomComparison(
        output_column_name=cat_col, comparison_levels=category_levels
    )

    return category_comparison
