import pandas as pd
from typing import List, Dict, Any
import json
import warnings
import os

warnings.filterwarnings("ignore")


def csv_loader(file_path: str, reset_indx=True) -> pd.DataFrame:
    """Load the csv file using panda dataframe utility function.
    It also handle the df columns and index formatting and
    give the new dataframe.

    Parameters
    ----------
    file_path : str
        Dataframe complete file path.

    Returns
    -------
    pd.DataFrame
        New pandas Dataframe.
    """

    raw_df = pd.read_csv(file_path, index_col=0)
    if reset_indx:

        raw_df.reset_index(drop=True)

    new_cols = [i.strip() for i in raw_df.keys()]

    raw_indx = raw_df.index

    if type(raw_indx[0]) is str:
        new_indx = [i.strip() for i in raw_indx if type(i) is str]
    else:
        new_indx = raw_indx

    return pd.DataFrame(raw_df.values, columns=new_cols, index=new_indx)


def df_slicer(file_path: str, histone_name: List[str]) -> pd.DataFrame:
    """sliced dataframe according to the requested
    column name list.

    Parameters
    ----------
    - file_path : str
        Dataframe path
    - histone_name : list
        Column names.

    Returns
    -------
    pd.DataFrame
        New pandas Dataframe.
    """

    df = csv_loader(file_path)
    histone_name = [name.strip().lower() for name in histone_name]

    histone_mod_name_lowed = dict(
        zip([histone_mod_name.lower() for histone_mod_name in df.keys()], df.keys())
    )
    histone_name_updt = [histone_mod_name_lowed[name] for name in histone_name]

    return df.loc[:, histone_name_updt]


def json_file_loader(path):
    with open(path, "r") as json_file:

        data = json.load(json_file)

    return data


def dict_order(raw_dict, order_keys):

    return {key: raw_dict[key] for key in order_keys}
