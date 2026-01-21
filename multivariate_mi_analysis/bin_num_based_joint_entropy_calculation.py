"""
Joint Entropy Calculation for Random Feature Pairs Covering All Histones

This script generates random feature pairs that ensure every histone PTM appears at least once,
then calculates joint entropy values across varying bin numbers for both human and yeast organisms.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import time
import argparse

import notebook__init__b
from path_utils.path_handler import ProjectPaths
from datahandler import data_handler
import entropy_mutualinfo


def generate_covering_pairs(num_features, random_seed=None):
    """Generate random pairs ensuring every feature appears at least once.

    Parameters
    ----------
    num_features : int
        Total number of features available
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list
        List of tuples containing feature index pairs
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Minimum number of pairs needed to cover all features
    # For n features, we need at least ceil(n/2) pairs
    min_pairs_needed = int(np.ceil(num_features / 2))

    # Create a list to track which features have been used
    features_used = set()
    pairs = []

    # First pass: ensure all features are covered
    available_features = list(range(num_features))
    np.random.shuffle(available_features)

    for i in range(0, len(available_features) - 1, 2):
        pair = (available_features[i], available_features[i + 1])
        pairs.append(pair)
        features_used.add(available_features[i])
        features_used.add(available_features[i + 1])

    # Handle odd number of features - pair the last one with a random feature
    if len(available_features) % 2 == 1:
        last_feature = available_features[-1]
        random_partner = np.random.choice(
            [f for f in range(num_features) if f != last_feature]
        )
        pairs.append((last_feature, random_partner))
        features_used.add(last_feature)
        features_used.add(random_partner)

    # Second pass: generate additional random pairs to increase diversity
    # This ensures we have more random pairings while maintaining the covering property
    additional_pairs_needed = max(0, min_pairs_needed - len(pairs))

    for _ in range(additional_pairs_needed):
        pair = tuple(np.random.choice(num_features, size=2, replace=False))
        if pair not in pairs and (pair[1], pair[0]) not in pairs:
            pairs.append(pair)

    return pairs


def process_organism_covering_pairs(
    organism, use_global_minmax, bin_num_list, random_seed=None
):
    """Process a single organism, calculating joint entropy for covering feature pairs.

    Parameters
    ----------
    organism : str
        Name of the organism to process (e.g., 'human', 'yeast')
    use_global_minmax : bool
        Whether to use global min/max values for all features
    bin_num_list : list
        List of bin numbers to test
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing pair information and joint entropy values
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    print(f"\n{'='*60}")
    print(f"PROCESSING ORGANISM: {organism.upper()}")
    print(f"{'='*60}")

    # File paths
    input_file = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )
    # input MI info file
    # path example:data\processed\mi_bin_info.json

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

    print(f"Global min: {global_min:.4f}, Global max: {global_max:.4f}")

    # Print which min/max values are being used
    if use_global_minmax:
        print(
            f"Using global min ({global_min:.4f}) and max ({global_max:.4f}) for all features"
        )
    else:
        print("Using individual min/max values for each feature")

    # Generate covering pairs
    num_features = len(data.columns)
    print(f"\nTotal features available: {num_features}")

    pair_indices = generate_covering_pairs(num_features, random_seed)
    print(f"Generated {len(pair_indices)} pairs covering all {num_features} features")

    # Create a list of feature pairs
    histone_pairs = []
    for idx1, idx2 in pair_indices:
        feature1 = data.columns[idx1]
        feature2 = data.columns[idx2]
        histone_pairs.append((feature1, feature2))

    print(f"\nSelected pairs:")
    for i, (f1, f2) in enumerate(histone_pairs, 1):
        print(f"  Pair {i}: {f1} <-> {f2}")

    # Verify coverage
    all_features_in_pairs = set()
    for f1, f2 in histone_pairs:
        all_features_in_pairs.add(f1)
        all_features_in_pairs.add(f2)

    print(f"\nCoverage verification:")
    print(f"  Total unique features in pairs: {len(all_features_in_pairs)}")
    print(f"  Total features in dataset: {num_features}")
    print(f"  All features covered: {len(all_features_in_pairs) == num_features}")

    if len(all_features_in_pairs) < num_features:
        uncovered = set(data.columns) - all_features_in_pairs
        print(
            f"  WARNING: {len(uncovered)} features not covered: {list(uncovered)[:5]}..."
        )

    # Create dictionary to store joint entropy values for each pair and bin number
    joint_entropy_values = {}

    # Track start time for performance reporting
    start_time = time.time()

    # Process each histone pair
    for pair_idx, (feature1, feature2) in enumerate(
        tqdm(histone_pairs, desc=f"Processing {organism} pairs"), 1
    ):
        feature1_values = data[feature1].values
        feature2_values = data[feature2].values

        # Calculate individual min/max for the current features
        feature1_min = np.min(feature1_values)
        feature1_max = np.max(feature1_values)
        feature2_min = np.min(feature2_values)
        feature2_max = np.max(feature2_values)

        # Select which min/max to use based on the boolean option
        min1 = global_min if use_global_minmax else feature1_min
        max1 = global_max if use_global_minmax else feature1_max
        min2 = global_min if use_global_minmax else feature2_min
        max2 = global_max if use_global_minmax else feature2_max

        # Initialize dictionary for this pair
        pair_name = f"pair_{pair_idx}_{feature1}_vs_{feature2}"
        joint_entropy_values[pair_name] = {
            "feature1": feature1,
            "feature2": feature2,
            "entropy_values": {},
        }

        # Process each bin number
        for bin_num in bin_num_list:
            # Create bin edges using the selected min/max values
            bin_edges = np.linspace(min1, max1, bin_num + 1)

            # Calculate joint probability
            joint_prob_matrix = entropy_mutualinfo.joint_prob(
                feature1_values, feature2_values, bin_edges
            )

            # Calculate joint entropy H(X, Y)
            joint_entropy = entropy_mutualinfo.entropy_calculator(joint_prob_matrix)

            # Store the joint entropy value
            joint_entropy_values[pair_name]["entropy_values"][bin_num] = joint_entropy

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(
        f"Processed {len(histone_pairs)} pairs in {int(minutes)} minutes and {seconds:.2f} seconds"
    )

    # Create output directory
    minmax_type = "global" if use_global_minmax else "individual"
    output_dir_name = "covering_pairs_joint_entropy_values"
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), output_dir_name, organism
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the joint entropy values to a JSON file
    output_file = os.path.join(
        output_dir, f"covering_pairs_joint_entropy_{minmax_type}_minmax.json"
    )

    print(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(joint_entropy_values, f, indent=4)

    # Generate a summary of joint entropy values at key bin counts
    summary_bins = [10, 50, 100, 200, 300, 400, 500]
    summary_bins = [b for b in summary_bins if b in bin_num_list]

    print("\nJoint Entropy Summary for Selected Bin Numbers:")
    print("Pair".ljust(40), end="")
    for bin_num in summary_bins:
        print(f"{bin_num:>10}", end="")
    print()

    for pair_name in list(joint_entropy_values.keys())[:10]:
        feature_pair = f"{joint_entropy_values[pair_name]['feature1']} vs {joint_entropy_values[pair_name]['feature2']}"
        print(feature_pair[:40].ljust(40), end="")
        for bin_num in summary_bins:
            if bin_num in joint_entropy_values[pair_name]["entropy_values"]:
                print(
                    f"{joint_entropy_values[pair_name]['entropy_values'][bin_num]:>10.3f}",
                    end="",
                )
            else:
                print(f"{'N/A':>10}", end="")
        print()

    if len(joint_entropy_values) > 10:
        print("... plus", len(joint_entropy_values) - 10, "more pairs")

    # Also save a CSV summary for easier analysis
    summary_data = []
    for pair_name, pair_data in joint_entropy_values.items():
        row = {
            "pair_name": pair_name,
            "feature1": pair_data["feature1"],
            "feature2": pair_data["feature2"],
        }
        for bin_num in summary_bins:
            row[f"bin_{bin_num}"] = pair_data["entropy_values"].get(bin_num, np.nan)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(
        output_dir, f"covering_pairs_joint_entropy_summary_{minmax_type}_minmax.csv"
    )
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary CSV saved to {summary_file}")

    # Save pair information for reference
    pair_info = {
        "organism": organism,
        "total_features": num_features,
        "total_pairs": len(histone_pairs),
        "all_features_covered": len(all_features_in_pairs) == num_features,
        "features_covered_count": len(all_features_in_pairs),
        "pairs": [
            {"pair_id": i + 1, "feature1": f1, "feature2": f2}
            for i, (f1, f2) in enumerate(histone_pairs)
        ],
    }

    pair_info_file = os.path.join(
        output_dir, f"covering_pairs_info_{minmax_type}_minmax.json"
    )
    with open(pair_info_file, "w") as f:
        json.dump(pair_info, f, indent=4)
    print(f"Pair information saved to {pair_info_file}")

    return joint_entropy_values


def main():
    """Main function to calculate joint entropy values for covering feature pairs in both organisms."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate joint entropy values for random feature pairs covering all histones"
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
        default="--global_minmax",
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
    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Use command line arguments
    organisms = args.organisms
    use_global_minmax = args.global_minmax
    random_seed = args.random_seed

    # Configure bin number list based on command line options
    start_bin = args.start_bin
    end_bin = args.end_bin
    step = args.step

    bin_num_list = list(range(start_bin, end_bin + 1, step))

    # Program start information
    print(f"\n{'='*80}")
    print("RANDOM FEATURE PAIRS COVERING ALL HISTONES - JOINT ENTROPY CALCULATION")
    print(f"{'='*80}")
    print(f"Organisms to process: {', '.join(organisms)}")
    print(f"Using global min/max: {use_global_minmax}")
    print(f"Random seed: {random_seed}")
    print(
        f"Testing {len(bin_num_list)} bin configurations from {start_bin} to {end_bin} with step {step}"
    )

    # Track overall start time
    overall_start_time = time.time()

    # Store results for all organisms
    all_results = {}

    # Process each organism
    for i, organism in enumerate(organisms, 1):
        print(f"\n[{i}/{len(organisms)}] Processing organism: {organism}")
        all_results[organism] = process_organism_covering_pairs(
            organism, use_global_minmax, bin_num_list, random_seed
        )

    # Calculate overall elapsed time
    overall_elapsed_time = time.time() - overall_start_time
    minutes, seconds = divmod(overall_elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)

    # Final summary
    print(f"\n{'='*80}")
    print("COVERING PAIRS JOINT ENTROPY CALCULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(organisms)} organisms: {', '.join(organisms)}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Save consolidated results if multiple organisms were processed
    if len(organisms) > 1:
        minmax_type = "global" if use_global_minmax else "individual"
        output_dir = os.path.join(
            ProjectPaths.get_output_files_dir(), "covering_pairs_joint_entropy_values"
        )
        os.makedirs(output_dir, exist_ok=True)
        consolidated_file = os.path.join(
            output_dir,
            f"consolidated_covering_pairs_joint_entropy_{minmax_type}_minmax.json",
        )

        print(f"Saving consolidated results to {consolidated_file}")
        with open(consolidated_file, "w") as f:
            json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
