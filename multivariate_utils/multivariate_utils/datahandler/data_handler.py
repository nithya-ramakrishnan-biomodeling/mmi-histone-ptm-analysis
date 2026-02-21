import functools
import glob
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import pandas as pd
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Loader Functions
def csv_loader(file_path: str, **kwargs: Any) -> Union[pd.DataFrame, Any]:
    """Load a CSV file into a DataFrame with optional index cleaning."""
    clean_index = kwargs.pop("clean_index", True)
    try:
        df = pd.read_csv(file_path, **kwargs)
        if clean_index and df.index.dtype == "object":
            df.index = df.index.str.strip()
        df.columns = [col.strip() for col in df.columns]
        logger.info(f"Successfully loaded CSV file: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Invalid CSV format: {file_path}")
        raise ValueError(f"Invalid CSV format: {file_path}")


def excel_loader(file_path: str, **kwargs: Any) -> Union[pd.DataFrame, Any]:
    """Load an Excel file into a DataFrame with optional index cleaning."""
    clean_index = kwargs.pop("clean_index", True)
    try:
        df = pd.read_excel(file_path, **kwargs)
        if clean_index and df.index.dtype == "object":
            df.index = df.index.str.strip()
        df.columns = [col.strip() for col in df.columns]
        logger.info(f"Successfully loaded Excel file: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Excel file not found: {file_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid Excel file: {file_path} - {str(e)}")
        raise ValueError(f"Invalid Excel file: {file_path}")


def json_loader(file_path: str, **kwargs: Any) -> Union[Dict[str, Any], Any]:
    """Load a JSON file into a dictionary."""
    try:
        with open(file_path, "r", encoding=kwargs.get("encoding", "utf-8")) as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON file: {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format: {file_path}")
        raise ValueError(f"Invalid JSON format: {file_path}")


def pickle_loader(file_path: str, **kwargs: Any) -> Any:
    """Load a pickle file and return the deserialized object."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded pickle file: {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Pickle file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Invalid pickle file: {file_path} - {str(e)}")
        raise ValueError(f"Invalid pickle file: {file_path}")


# Generalized Data Loader
def load_data(
    file_path: str, validate: Optional[Callable[[Any], None]] = None, **kwargs: Any
) -> Any:
    """
    Load data from a file based on its extension with optional validation.

    Parameters
    ----------
    file_path : str
        Path to the file.
    validate : callable, optional
        Function to validate the loaded data.
    **kwargs : additional keyword arguments
        Passed to the specific loader function.

    Returns
    -------
    pd.DataFrame or dict or any
        Loaded data, type depends on file extension.

    Raises
    ------
    ValueError
        If the file extension is unsupported or validation fails.
    """
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".csv": csv_loader,
        ".xlsx": excel_loader,
        ".xls": excel_loader,
        ".json": json_loader,
        ".pkl": pickle_loader,
        ".pickle": pickle_loader,
    }
    if ext not in loaders:
        logger.error(f"Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported file extension: {ext}")

    data = loaders[ext](file_path, **kwargs)
    # Remove 'Unnamed' columns if DataFrame
    if isinstance(data, pd.DataFrame):
        unnamed_cols = [
            col for col in data.columns if col.lower().startswith("unnamed")
        ]
        if unnamed_cols:
            data = data.drop(columns=unnamed_cols)
    if validate:
        try:
            validate(data)
            logger.info("Data validation passed")
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise ValueError(f"Data validation failed: {str(e)}")
    return data


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


# DataFrame Slicer
def df_slicer(file_path: str, columns: List[str], **kwargs: Any) -> pd.DataFrame:
    """
    Load a DataFrame and slice it to include only specified columns.

    Parameters
    ----------
    file_path : str
        Path to the file.
    columns : List[str]
        List of column names to extract.
    **kwargs : additional keyword arguments
        Passed to load_data (e.g., clean_index, sheet_name).

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame with requested columns.
    """
    data = load_data(file_path, **kwargs)
    if not isinstance(data, pd.DataFrame):
        logger.error(f"Loaded data is not a DataFrame: {file_path}")
        raise ValueError("Loaded data must be a pandas DataFrame")

    columns = [col.strip().lower() for col in columns]
    available_cols = {col.lower(): col for col in data.columns}
    valid_cols = [available_cols[col] for col in columns if col in available_cols]

    if not valid_cols:
        logger.warning(f"No valid columns found in {file_path}")
        return pd.DataFrame()

    logger.info(f"Slicing DataFrame to columns: {valid_cols}")
    return data.loc[:, valid_cols]


# JSON File Saver
def json_file_saver(data: Any, file_path: str, indent: int = 4, **kwargs: Any) -> None:
    """
    Save data to a JSON file.

    Parameters
    ----------
    data : Any
        Data to save.
    file_path : str
        Path to the output file.
    indent : int, optional
        JSON indentation level.
    **kwargs : additional keyword arguments
        Passed to json.dump (e.g., encoding).
    """
    try:
        with open(file_path, "w", encoding=kwargs.get("encoding", "utf-8")) as f:
            json.dump(data, f, indent=indent, **kwargs)
        logger.info(f"Data saved to JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {file_path} - {str(e)}")
        raise


# Dictionary Utilities
def dict_order(
    raw_dict: Dict[str, Any], order_keys: List[str], include_all: bool = False
) -> Dict[str, Any]:
    """
    Reorder a dictionary based on a list of keys.

    Parameters
    ----------
    raw_dict : Dict[str, Any]
        Original dictionary.
    order_keys : List[str]
        Desired key order.
    include_all : bool, optional
        If True, include all keys, with unspecified ones appended.

    Returns
    -------
    Dict[str, Any]
        Reordered dictionary.
    """
    if include_all:
        ordered = {k: raw_dict[k] for k in order_keys if k in raw_dict}
        remaining = {k: v for k, v in raw_dict.items() if k not in order_keys}
        return {**ordered, **remaining}
    return {k: raw_dict[k] for k in order_keys if k in raw_dict}


def dict_sort(
    raw_dict: Dict[str, Any], by: str = "key", reverse: bool = False
) -> Dict[str, Any]:
    """
    Return a new dictionary sorted by keys or values.

    Parameters
    ----------
    raw_dict : Dict[str, Any]
        The dictionary to sort.
    by : str, optional
        Sort by 'key' or 'value'. Default is 'key'.
    reverse : bool, optional
        If True, sort in descending order.

    Returns
    -------
    Dict[str, Any]
        Sorted dictionary.

    Raises
    ------
    ValueError
        If 'by' is not 'key' or 'value'.
    """
    if by == "key":
        sorted_items = sorted(
            raw_dict.items(), key=lambda item: item[0], reverse=reverse
        )
    elif by == "value":
        sorted_items = sorted(
            raw_dict.items(), key=lambda item: item[1], reverse=reverse
        )
    else:
        raise ValueError("Parameter 'by' must be either 'key' or 'value'.")
    return dict(sorted_items)


# Example Validation Function
def validate_df_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Validate that a DataFrame has all required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def memory_usage_report(df: pd.DataFrame) -> None:
    """Print memory usage report for DataFrame"""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / (1024 * 1024)  # Convert to MB

    print(f"DataFrame Memory Usage:")
    print(f"Shape: {df.shape}")
    print(f"Total Memory: {total_memory:.2f} MB")
    print(f"Index Memory: {memory_usage.iloc[0] / (1024 * 1024):.2f} MB")
    print(f"Data Memory: {memory_usage.iloc[1:].sum() / (1024 * 1024):.2f} MB")


def log_header(title: str, width: int = 70):
    """
    Standard function to print a double-line styled header.
    Using a single logger call ensures the header stays as one block in logs.

    """
    # import logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # logger = logging.getLogger(__name__)

    divider = "=" * width
    # Using a single f-string with \n is the standard for atomic logging
    header = f"\n{divider}\n{title.upper()}\n{divider}"
    print(header)
