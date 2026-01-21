"""
Two-Variable Mutual Information Calculation for All Histone PTMs Based on Varying Bin Numbers

This script calculates mutual information (MI) values for all pairs of histone PTMs across a range of bin numbers
for both human and yeast organisms, helping to determine the optimal bin count for MI calculations.
"""

import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse

import notebook__init__b
from path_utils.path_handler import ProjectPaths
from datahandler import data_handler
import entropy_mutualinfo


def process_organism_mi(organism, use_global_minmax, bin_num_list):
    """Process a single organism, calculating MI for all feature pairs."""

    # File paths
    input_file = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )
    input_mi_info_file = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), "mi_bin_info.json"
    )

    # Load the CSV file
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, index_col=0)
    print(f"Data loaded with shape {data.shape}")

    # Loading the MI info file
    mi_dict_file = data_handler.json_file_loader(input_mi_info_file)
    mi_dict = mi_dict_file[organism]

    # Global min/max values
    global_min = mi_dict["global_min"]
    global_max = mi_dict["global_max"]

    # Print which min/max values are being used
    if use_global_minmax:
        print(
            f"Using global min ({global_min:.4f}) and max ({global_max:.4f}) for all features"
        )

    # Create dictionary to store MI values for each feature pair and bin number
    mi_values = {}
    features = data.columns
    # Process each feature pair
    for i, feat1 in enumerate(
        tqdm(features, desc=f"Processing {organism} feature pairs")
    ):
        mi_values[feat1] = {}
        feat1_values = data[feat1].values
        for j, feat2 in enumerate(features):
            if j < i:
                continue  # Avoid duplicate pairs and self-pairing
            feat2_values = data[feat2].values
            # Initialize dictionary for this feature pair
            pair_key = f"{feat1}__{feat2}"
            mi_values[feat1][pair_key] = {}

            # Process each bin number
            for bin_num in bin_num_list:
                bin_edges = np.linspace(global_min, global_max, bin_num + 1)
                mi = entropy_mutualinfo.mi_betwn_uandy(
                    feat1_values, feat2_values, bin_edges
                )
                mi_values[feat1][pair_key][bin_num] = mi

    # Create output directory
    minmax_type = "global" if use_global_minmax else "individual"
    output_dir_name = "mi_values"
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), output_dir_name, organism
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the MI values to a JSON file
    output_file = os.path.join(
        output_dir, f"all_ptms_bin_based_mi_values_{minmax_type}_minmax.json"
    )
    print(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(mi_values, f, indent=4)

    return mi_values


def main():
    """Main function to calculate MI values for all histone PTM pairs in both organisms."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate MI values for histone PTM pairs"
    )
    parser.add_argument(
        "-o",
        "--organisms",
        nargs="+",
        default=["human", "yeast"],
        help="Organisms to process (default: both human and yeast)",
    )
    parser.add_argument(
        "-g",
        "--global_minmax",
        action="store_true",
        help="Use global min/max values for all features",
    )
    parser.add_argument(
        "-b",
        "--start_bin",
        type=int,
        default=10,
        help="Starting bin number (default: 10)",
    )
    parser.add_argument(
        "-e",
        "--end_bin",
        type=int,
        default=500,
        help="Ending bin number (default: 500)",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=10,
        help="Step size for bin numbers (default: 10)",
    )
    args = parser.parse_args()

    organisms = args.organisms
    use_global_minmax = args.global_minmax
    start_bin = args.start_bin
    end_bin = args.end_bin
    step = args.step

    bin_num_list = list(range(start_bin, end_bin + 1, step))
    all_results = {}

    for i, organism in enumerate(organisms, 1):
        print(f"\n[{i}/{len(organisms)}] Processing organism: {organism}")
        all_results[organism] = process_organism_mi(
            organism, use_global_minmax, bin_num_list
        )

    # Save consolidated results if multiple organisms were processed
    if len(organisms) > 1:
        minmax_type = "global" if use_global_minmax else "individual"
        output_dir = os.path.join(ProjectPaths.get_output_files_dir(), "mi_values")
        os.makedirs(output_dir, exist_ok=True)
        consolidated_file = os.path.join(
            output_dir, f"consolidated_mi_values_{minmax_type}_minmax.json"
        )
        print(f"Saving consolidated results to {consolidated_file}")
        with open(consolidated_file, "w") as f:
            json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
