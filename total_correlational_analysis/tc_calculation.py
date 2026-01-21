import numpy as np
import pandas as pd
import json
from typing import List, Optional
from itertools import combinations
import notebook__init__b

from tc_utils import entropy_mutualinfo
from pathlib import Path
import argparse
import os
from joblib import Parallel, delayed
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths


def generate_unique_triplets(elements: List[str]) -> List[tuple]:
    """Generate unique triplets efficiently using itertools."""
    return list(combinations(elements, 3))


def create_bin_spacing(
    min_value: float, max_value: float, bin_number: int
) -> np.ndarray:
    """Creates equally spaced bins between minimum and maximum values."""
    return np.linspace(min_value, max_value, bin_number)


def compute_total_correlation_triplet(
    triplet_indices: tuple, histone_data: np.ndarray, bin_edges: np.ndarray
) -> tuple:
    """Compute total correlation for a single triplet using indices directly."""
    idx1, idx2, idx3, triplet_names = triplet_indices
    arr1, arr2, arr3 = (
        histone_data[:, idx1],
        histone_data[:, idx2],
        histone_data[:, idx3],
    )
    total_corr_value = entropy_mutualinfo.total_correlation(arr1, arr2, arr3, bin_edges)

    # Return both formats: original format and sorted triplet format
    original_format = (
        f"{triplet_names[0]}_{triplet_names[1]}",
        triplet_names[2],
        total_corr_value,
    )
    sorted_triplet = tuple(sorted(triplet_names))  # Sort to reduce ambiguity
    triplet_format = (sorted_triplet, total_corr_value)

    return original_format, triplet_format


def create_optimized_dataframe(triplets: List[tuple]) -> pd.DataFrame:
    """Create DataFrame structure more efficiently."""
    # Get unique row and column labels
    row_labels = [f"{t[0]}_{t[1]}" for t in triplets]
    col_labels = list(set(t[2] for t in triplets))

    # Pre-allocate DataFrame with NaN values
    df = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
    return df


def create_triplet_combinations_dict(triplet_results: List[tuple]) -> dict:
    """Create dictionary with sorted triplet tuples as keys (converted to strings for JSON)."""
    triplet_dict = {}
    for sorted_triplet, total_corr_value in triplet_results:
        # Convert tuple to string representation for JSON compatibility
        key = str(sorted_triplet)
        triplet_dict[key] = round(total_corr_value, 6)  # Round to 6 decimal places

    # Sort by keys to maintain consistent ordering
    sorted_triplet_dict = dict(sorted(triplet_dict.items()))

    return sorted_triplet_dict


def json_file_saver(data_dict: dict, absolute_file_name: str):
    """Save dictionary to JSON file with pretty formatting."""
    with open(absolute_file_name, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)


if __name__ == "__main__":
    # Set up argument parser for organism specification
    parser = argparse.ArgumentParser(
        description="Total correlation analysis in parallel"
    )
    parser.add_argument(
        "-o", "--organism", type=str, required=True, help="The organism to analyze"
    )
    args = parser.parse_args()

    # Configure directories
    out_folder_name = "total correlation analysis"
    main_dir = ProjectPaths.get_project_root()
    cleaned_dir = ProjectPaths.get_cleaned_data_dir()
    output_file_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), out_folder_name, args.organism
    )
    output_img_dir = os.path.join(
        ProjectPaths.get_output_images_dir(), out_folder_name, args.organism
    )

    # Create output directories if they don't exist
    dir_handler.dir_maker(output_file_dir)
    dir_handler.dir_maker(output_img_dir)

    # Load input data
    input_file_path = os.path.join(
        cleaned_dir, args.organism, f"{args.organism}_zero_mean.csv"
    )
    histone_df = data_handler.csv_loader(input_file_path)

    # Load binning information
    mi_bin_info_dict = data_handler.json_file_loader(
        os.path.join(cleaned_dir, "mi_bin_info.json")
    )[args.organism]

    # Create bin edges
    bin_edges = create_bin_spacing(
        min_value=mi_bin_info_dict["global_min"],
        max_value=mi_bin_info_dict["global_max"],
        bin_number=mi_bin_info_dict["bin_num"],
    )

    # Generate unique triplets from the histone marks
    columns = histone_df.columns.tolist()
    list_of_triplets = generate_unique_triplets(columns)

    # Convert DataFrame to numpy array for faster access
    histone_data = histone_df.values

    # Pre-compute indices and prepare picklable data
    triplet_indices = []
    for triplet in list_of_triplets:
        idx1 = columns.index(triplet[0])
        idx2 = columns.index(triplet[1])
        idx3 = columns.index(triplet[2])
        triplet_indices.append((idx1, idx2, idx3, triplet))

    # Create optimized DataFrame structure for original format
    df_original = create_optimized_dataframe(list_of_triplets)

    # Parallel computation of total correlations
    n_jobs = min(os.cpu_count() - 1, len(triplet_indices))  # Leave one core free

    print(f"Processing {len(triplet_indices)} triplets using {n_jobs} cores...")

    # Use threading backend to avoid pickling issues
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=1)(
        delayed(compute_total_correlation_triplet)(triplet_idx, histone_data, bin_edges)
        for triplet_idx in triplet_indices
    )

    print("Processing results...")

    # Separate original format and triplet format results
    original_results = []
    triplet_results = []

    for original_format, triplet_format in results:
        original_results.append(original_format)
        triplet_results.append(triplet_format)

    print("Filling original format DataFrame...")
    # Fill original DataFrame efficiently using batch assignment
    for row_label, col_label, value in original_results:
        df_original.at[row_label, col_label] = value

    print("Creating triplet combinations dictionary...")
    # Create triplet combinations dictionary
    triplet_dict = create_triplet_combinations_dict(triplet_results)

    # Save both outputs
    # Original format CSV file
    output_file_path_original = os.path.join(
        output_file_dir, "total_correlation_results.csv"
    )
    df_original.to_csv(output_file_path_original)

    # Triplet combinations JSON file
    output_file_path_triplets = os.path.join(
        output_file_dir, "total_correlation_triplets.json"
    )
    json_file_saver(triplet_dict, output_file_path_triplets)

    print(f"Total correlation analysis completed for {args.organism}")
    print(f"Original format results saved to: {output_file_path_original}")
    print(f"Triplet combinations results saved to: {output_file_path_triplets}")

    # Print summary statistics
    values = list(triplet_dict.values())
    print(f"\nSummary:")
    print(f"Total triplets processed: {len(triplet_indices)}")
    print(f"Original DataFrame shape: {df_original.shape}")
    print(f"Triplet dictionary entries: {len(triplet_dict)}")
    print(f"Total correlation range: {min(values):.6f} to {max(values):.6f}")
