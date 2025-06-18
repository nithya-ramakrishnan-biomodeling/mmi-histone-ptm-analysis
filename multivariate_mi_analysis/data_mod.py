import operator
from typing import Literal
import pandas as pd

operators = {
    "lt": operator.lt,  # Less than
    "gt": operator.gt,  # greater than
    "le": operator.le,  # less than or equal to
    "ge": operator.ge,  # greater than or equal to
    "eq": operator.eq,  # equal to
    "ne": operator.ne,
}  # not equal to


def histone_comb_generator(histone_mods: list, r: int):
    """Generate histone combination for regression model

    Note: Only generate the 3 pair and 2 pair combinations.

    Parameters
    ----------
    hostone_mod : list
        _description_
    r : int
        elements in the each pair.

    Returns
    -------
    List
        Unique combinations of the histone modificaiton pair.
    """

    from itertools import combinations

    if r == 2:

        two_comb_list = list(combinations(histone_mods, r=2))
        two_comb_list[1] = two_comb_list[1][::-1]

        comb_list = two_comb_list

    elif r == 3:
        three_comb_list = []
        order = [1, 0, 2]
        for indx, histone in enumerate(histone_mods):
            comp_list = [i for i in histone_mods if i not in [histone]]
            comp_list.append(histone)
            three_comb_list.append(comp_list)

        comb_list = [three_comb_list[i] for i in order]

    else:

        comb_list = f"Please provide the r value <=3"

    return comb_list


def flatten_dict(nested_dict, parent_key="", sep="_"):
    """
    Flatten a nested dictionary by concatenating keys.

    Args:
        nested_dict (dict): The nested dictionary to flatten.
        parent_key (str): The base key for the current level.
        sep (str): The separator to use between keys."

    Returns:
        dict: A flattened dictionary.
    """
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))  # Recursive call
        else:
            items[new_key] = value
    return items


# * function for removing the Nan values and fileter the dataframe with condition
def three_var_mi_to_dict_filter(
    df: pd.DataFrame,
    cond: Literal["lt", "gt", "le", "ge", "eq", "ne"],
    value: float,
    order: bool = False,
):

    # Get only values from conditions and non-NaN values
    filtered_df = df[(operators[cond](df, value)) & df.notna()]

    # Convert to a dictionary while removing NaN values
    df_dict = filtered_df.apply(lambda x: x.dropna().to_dict(), axis=0).to_dict()
    df_dict_sortd = dict(
        sorted(flatten_dict(df_dict).items(), key=lambda x: x[1], reverse=order)
    )
    return df_dict_sortd


# Function for the getting the three_var mi and corresponfing covarince values based on the conditions
