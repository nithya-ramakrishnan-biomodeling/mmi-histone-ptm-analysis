import pandas as pd


def csv_loader(file_path: str) -> pd.DataFrame:
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
    new_cols = [i.strip() for i in raw_df.keys()]

    raw_indx = raw_df.index

    if type(raw_indx[0]) is str:
        new_indx = [i.strip() for i in raw_df.index]

    else:
        new_indx = raw_df.index

    return pd.DataFrame(raw_df.values, columns=new_cols, index=new_indx)


def df_slicer(file_path: str, histone_name: list) -> pd.DataFrame:
    """It gives sliced dataframe according to the requested
    column name list.

    Parameters
    ----------
    file_path : str
        Dataframe path
    histone_name : list
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


def df_indx_selectr(file_path: str, pos: list) -> pd.DataFrame:
    """Slice the dataframe based on the given slicing positions

    Parameters
    ----------
    file_path : str
        Dataframe file path.
    pos : list
        Column position in the pandas dataframe.

    Returns
    -------
    pd.DataFrame

    """
    df = csv_loader(file_path=file_path)
    return df.iloc[:, pos]


def feature_traget_gen(hist_file_path: str, histone_name: list, target_col_num: int):
    """Feature and target columns selection

    Parameters
    ----------
    hist_file_path : str
        Dataframe file path.
    histone_name : list
        Column names for the feature columns
    target_col_num : int
        Position of the target column.


    Returns
    -------
    _type_
        _description_
    """

    indx_list = [0, (1, 2, 0), (0, 2, 1), (0, 1, 2)]
    order_hist = [histone_name[i] for i in indx_list[target_col_num]]
    df = df_slicer(hist_file_path, order_hist)
    print(f"Features:{order_hist[:2]}, Target: {list(order_hist[2:])}")

    feature = df.iloc[:, :2]
    target = df.iloc[:, 2:]

    return [(order_hist[:2], order_hist[2:]), (feature, target)]
