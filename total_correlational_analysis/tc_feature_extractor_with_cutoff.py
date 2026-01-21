import pandas as pd
import json
import os
import argparse
import ast
import notebook__init__b
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths

"""
# Total Correlation Feature Extractor

This module extracts features from total correlation analysis csv -
file based on cutoff values and operators.

## Overview
- Loads total correlation results from CSV file
- Applies cutoff filters with specified operators
- Extracts unique histone names from filtered results
- Saves feature lists as JSON files

## Main Components

### 1. Command Line Interface
Required arguments:
- `-o, --organism`: Target organism (e.g., 'human', 'yeast')
- `-c, --cutoff`: Cutoff value for filtering
- `-op, --operator`: Comparison operator ('ge', 'le', 'gt', 'lt', 'eq')

### 2. Functions
- get_triples(): Filter DataFrame based on cutoff and operator
- get_unique_names(): Extract unique histone names from filtered data

## Usage
```bash
python total_correlation_feature_extractor.py -o yeast -c 2.2 -op ge
python total_correlation_feature_extractor.py -o yeast -c 0.28 -op le
"""


def get_triples(df, cutoff, operator):
    """
    Filter DataFrame based on cutoff value and operator.

    Parameters
    ----------
    df : pd.DataFrame
        Total correlation results DataFrame
    cutoff : float
        Cutoff value for filtering
    operator : str
        Comparison operator ('ge', 'le', 'gt', 'lt', 'eq')

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with NaN rows/columns removed
    """
    if operator == "ge":
        filtered_df = df[df >= cutoff]
    elif operator == "le":
        filtered_df = df[df <= cutoff]
    elif operator == "gt":
        filtered_df = df[df > cutoff]
    elif operator == "lt":
        filtered_df = df[df < cutoff]
    elif operator == "eq":
        filtered_df = df[df == cutoff]
    else:
        raise ValueError("Unsupported operator. Use 'ge', 'le', 'gt', 'lt', or 'eq'.")

    # Drop rows and columns that are all NaN
    filtered_df = filtered_df.dropna(how="all")
    filtered_df = filtered_df.dropna(axis=1, how="all")

    return filtered_df


def get_unique_names(df):
    """
    Extract unique histone names from DataFrame index and columns.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered total correlation DataFrame

    Returns
    -------
    list
        List of unique histone names
    """
    # Get unique names from the index
    index_names = df.index.unique().tolist()

    # Split the names and get unique elements
    unique_names = set()
    for name in index_names:
        parts = name.split("_")
        unique_names.update(parts)

    # Get unique names from the columns
    column_names = df.columns.unique().tolist()
    unique_names.update(column_names)

    return list(unique_names)


def json_file_saver(data_list: list, absolute_file_name: str):
    """
    Save list to JSON file with pretty formatting.

    Parameters
    ----------
    data_list : list
        List to save
    absolute_file_name : str
        Path to output file
    """
    with open(absolute_file_name, "w") as json_file:
        json.dump(data_list, json_file, indent=4)


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Extract features from total correlation analysis"
    )

    # Add arguments
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="The organism to analyze",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        required=True,
        help="Cutoff value for filtering",
    )
    parser.add_argument(
        "-op",
        "--operator",
        type=str,
        required=True,
        choices=["ge", "le", "gt", "lt", "eq"],
        help="Comparison operator for filtering",
    )

    # Parse arguments
    args = parser.parse_args()

    # Access variable values
    organism = args.organism
    cutoff = args.cutoff
    operator = args.operator

    # Configure directories
    total_corr_dir = "total correlation analysis"

    # Input file path
    input_file_path = os.path.join(
        ProjectPaths.get_output_files_dir(),
        total_corr_dir,
        organism,
        "total_correlation_results.csv",
    )

    # Output directory
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), total_corr_dir, organism
    )

    # Create output directory if it doesn't exist
    dir_handler.dir_maker(output_dir)

    print(f"Loading total correlation results for {organism}")
    print(f"Input file: {input_file_path}")

    # Load the DataFrame
    try:
        df = pd.read_csv(input_file_path, index_col=0)
        print(f"Loaded DataFrame with shape: {df.shape}")

        # Display basic statistics
        print("\nDataFrame statistics:")
        print(f"Max value: {df.max().max():.3f}")
        print(f"Min value: {df.min().min():.3f}")

    except Exception as e:
        print(f"Error loading file: {e}")
        exit(1)

    # Apply filtering
    print(f"\nApplying filter: {operator} {cutoff}")
    filtered_df = get_triples(df, cutoff, operator)
    print(f"Filtered DataFrame shape: {filtered_df.shape}")

    # Extract unique names
    unique_features = get_unique_names(filtered_df)
    print(f"Found {len(unique_features)} unique features")
    print(f"Features: {unique_features}")

    # # Generate output filename
    # cutoff_str = str(cutoff).replace('.', '_').replace('-', 'neg')
    # output_filename = f"{organism}_total_corr_{operator}_{cutoff_str}_features.json"
    # output_file_path = os.path.join(output_dir, output_filename)

    # # Save features to JSON
    # json_file_saver(unique_features, output_file_path)
    # print(f"\nFeatures saved to: {output_file_path}")

    # # Also create standardized max/min feature files if appropriate
    # if operator in ['ge', 'gt'] and len(unique_features) >= 10:
    #     max_features_path = os.path.join(output_dir, "max_10_features.json")
    #     json_file_saver(unique_features[:10], max_features_path)
    #     print(f"Max 10 features saved to: {max_features_path}")

    # elif operator in ['le', 'lt'] and len(unique_features) >= 10:
    #     min_features_path = os.path.join(output_dir, "min_10_features.json")
    #     json_file_saver(unique_features[:10], min_features_path)
    #     print(f"Min 10 features saved to: {min_features_path}")


# ------------------------------------------------------------------------------
# This section shows how to load a JSON file containing unique triplet data.
# Load the JSON file
# with open('total_correlation_triplets.json', 'r') as f:
#     data = json.load(f)

# # Convert string keys back to tuples if needed
# triplet_data = {ast.literal_eval(k): v for k, v in data.items()}
