import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import warnings
import os
import gc

warnings.filterwarnings("ignore")


def csv_loader(
    file_path: str, reset_indx: bool = True, optimize_memory: bool = True
) -> pd.DataFrame:
    """Load the csv file using pandas dataframe utility function.
    It also handles the df columns and index formatting and
    gives the new dataframe.

    Parameters
    ----------
    file_path : str
        Dataframe complete file path.
    reset_indx : bool, optional
        Whether to reset the index, by default True
    optimize_memory : bool, optional
        Whether to optimize memory usage, by default True

    Returns
    -------
    pd.DataFrame
        New pandas DataFrame.

    Raises
    ----
    FileNotFoundError
        If the file path does not exist
    pd.errors.EmptyDataError
        If the CSV file is empty
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load CSV with error handling
        raw_df = pd.read_csv(file_path, index_col=0)

        if raw_df.empty:
            raise pd.errors.EmptyDataError("CSV file is empty")

        # Reset index if requested
        if reset_indx:
            raw_df = raw_df.reset_index(drop=True)  # Fixed: actually reset the index

        # Clean column names - remove whitespace and handle duplicates
        new_cols = [str(col).strip() for col in raw_df.columns]

        # Handle duplicate column names
        seen_cols = {}
        unique_cols = []
        for col in new_cols:
            if col in seen_cols:
                seen_cols[col] += 1
                unique_cols.append(f"{col}_{seen_cols[col]}")
            else:
                seen_cols[col] = 0
                unique_cols.append(col)

        # Clean index names
        raw_indx = raw_df.index
        if len(raw_indx) > 0 and isinstance(raw_indx[0], str):
            new_indx = [str(idx).strip() for idx in raw_indx]
        else:
            new_indx = raw_indx

        # Create new DataFrame with cleaned names
        result_df = pd.DataFrame(raw_df.values, columns=unique_cols, index=new_indx)

        # Optimize memory usage if requested
        if optimize_memory:
            result_df = _optimize_dataframe_memory(result_df)

        # Clean up
        del raw_df
        gc.collect()

        return result_df

    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The CSV file {file_path} is empty")
    except Exception as e:
        raise Exception(f"Error loading CSV file {file_path}: {str(e)}")


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types"""
    try:
        # Downcast float columns
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float", errors="ignore")

        # Downcast integer columns
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer", errors="ignore")

        return df
    except Exception:
        # If optimization fails, return original DataFrame
        return df


def df_slicer(file_path: str, histone_name: List[str]) -> pd.DataFrame:
    """Slice dataframe according to the requested column name list.

    Parameters
    ----------
    file_path : str
        Dataframe path
    histone_name : List[str]
        Column names to extract

    Returns
    -------
    pd.DataFrame
        New pandas DataFrame with selected columns

    Raises
    ------
    ValueError
        If none of the requested columns are found
    """
    if not histone_name:
        raise ValueError("histone_name list cannot be empty")

    df = csv_loader(file_path)

    # Normalize requested column names
    histone_name_normalized = [name.strip().lower() for name in histone_name]

    # Create mapping of lowercase column names to actual column names
    column_mapping = {col.lower(): col for col in df.columns}

    # Find matching columns
    found_columns = []
    missing_columns = []

    for requested_col in histone_name_normalized:
        if requested_col in column_mapping:
            found_columns.append(column_mapping[requested_col])
        else:
            missing_columns.append(requested_col)

    if not found_columns:
        raise ValueError(
            f"None of the requested columns {histone_name} found in DataFrame"
        )

    if missing_columns:
        print(f"Warning: The following columns were not found: {missing_columns}")

    return df.loc[:, found_columns]


def json_file_loader(path: str) -> Dict[str, Any]:
    """Load JSON file and return as dictionary

    Parameters
    ----------
    path : str
        Path to JSON file

    Returns
    -------
    Dict[str, Any]
        Dictionary loaded from JSON file

    Raises
    ------
    FileNotFoundError
        If the file path does not exist
    json.JSONDecodeError
        If the file is not valid JSON
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading JSON file {path}: {str(e)}")


def json_file_saver(data_dict: Dict[str, Any], file_path: str) -> None:
    """Save dictionary to JSON file with pretty formatting

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Dictionary to save
    file_path : str
        Path to output file

    Raises
    ------
    TypeError
        If data_dict is not serializable to JSON
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(data_dict, json_file, indent=4, ensure_ascii=False)
    except TypeError as e:
        raise TypeError(f"Data is not JSON serializable: {str(e)}")
    except Exception as e:
        raise Exception(f"Error saving JSON file {file_path}: {str(e)}")


def dict_order(raw_dict: Dict[str, Any], order_keys: List[str]) -> Dict[str, Any]:
    """Reorder dictionary according to specified key order

    Parameters
    ----------
    raw_dict : Dict[str, Any]
        Original dictionary
    order_keys : List[str]
        List of keys in desired order

    Returns
    -------
    Dict[str, Any]
        Reordered dictionary

    Raises
    ------
    KeyError
        If any key in order_keys is not found in raw_dict
    """
    if not order_keys:
        return raw_dict.copy()

    missing_keys = [key for key in order_keys if key not in raw_dict]
    if missing_keys:
        raise KeyError(f"Keys not found in dictionary: {missing_keys}")

    return {key: raw_dict[key] for key in order_keys if key in raw_dict}


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame for common issues

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate

    Returns
    -------
    bool
        True if DataFrame passes validation
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False

    # Check for all NaN columns
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Columns with all NaN values: {all_nan_cols}")

    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Warning: Duplicate column names: {duplicate_cols}")

    return True


def memory_usage_report(df: pd.DataFrame) -> None:
    """Print memory usage report for DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / (1024 * 1024)  # Convert to MB

    print(f"DataFrame Memory Usage:")
    print(f"Shape: {df.shape}")
    print(f"Total Memory: {total_memory:.2f} MB")
    print(f"Index Memory: {memory_usage.iloc[0] / (1024 * 1024):.2f} MB")
    print(f"Data Memory: {memory_usage.iloc[1:].sum() / (1024 * 1024):.2f} MB")
