import pandas as pd
import operator


def filter_dataframe(df, cutoff_value, comparison_operator):
    """
    Filters a DataFrame based on a cutoff value and a comparison operator.

    Parameters:
    - df: pd.DataFrame - The DataFrame to filter.
    - cutoff_value: float - The cutoff value for filtering.
    - comparison_operator: str - A string representing the comparison operator
      (e.g., '<', '<=', '>', '>=', '==', '!=').

    Returns:
    - pd.DataFrame - The filtered DataFrame with rows removed if all values do not match the criteria.
    """
    # Define available operators
    operators = {
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
        "e": operator.eq,
        "note": operator.ne,
    }

    # Check if the provided operator is valid
    if comparison_operator not in operators:
        raise ValueError(
            f"Invalid comparison operator '{comparison_operator}'. Use one of {list(operators.keys())}."
        )

    # Apply the filter using the selected operator
    filtered_df = df[operators[comparison_operator](df, cutoff_value)].dropna(how="all")
    return filtered_df
