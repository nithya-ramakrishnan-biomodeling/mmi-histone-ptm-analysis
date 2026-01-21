"""
Joint Entropy Calculation for Random Feature Triplets Covering All Histones

This script generates random feature triplets that ensure every histone PTM appears at least once,
then calculates joint entropy values H(X,Y,Z) across varying bin numbers for both human and yeast organisms.
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


def generate_covering_triplets(num_features, random_seed=None):
    """Generate random triplets ensuring every feature appears at least once.

    Parameters
    ----------
    num_features : int
        Total number of features available
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list
        List of tuples containing feature index triplets
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Minimum number of triplets needed to cover all features
    # For n features, we need at least ceil(n/3) triplets
    min_triplets_needed = int(np.ceil(num_features / 3))

    # Create a list to track which features have been used
    features_used = set()
    triplets = []

    # First pass: ensure all features are covered
    available_features = list(range(num_features))
    np.random.shuffle(available_features)

    # Group features into triplets
    for i in range(0, len(available_features) - 2, 3):
        triplet = (
            available_features[i],
            available_features[i + 1],
            available_features[i + 2],
        )
        triplets.append(triplet)
        features_used.add(available_features[i])
        features_used.add(available_features[i + 1])
        features_used.add(available_features[i + 2])

    # Handle remaining features (1 or 2)
    remaining = len(available_features) % 3
    if remaining == 1:
        # Last feature needs 2 random partners
        last_feature = available_features[-1]
        partners = np.random.choice(
            [f for f in range(num_features) if f != last_feature], size=2, replace=False
        )
        triplet = (last_feature, partners[0], partners[1])
        triplets.append(triplet)
        features_used.add(last_feature)
        features_used.add(partners[0])
        features_used.add(partners[1])
    elif remaining == 2:
        # Last 2 features need 1 random partner each
        last_two = available_features[-2:]
        partner1 = np.random.choice(
            [f for f in range(num_features) if f != last_two[0] and f != last_two[1]]
        )
        partner2 = np.random.choice(
            [f for f in range(num_features) if f != last_two[1] and f != last_two[0]]
        )

        triplet1 = (last_two[0], last_two[1], partner1)
        triplets.append(triplet1)
        features_used.add(last_two[0])
        features_used.add(last_two[1])
        features_used.add(partner1)

    # Second pass: generate additional random triplets to increase diversity
    # This ensures we have more random triplets while maintaining the covering property
    additional_triplets_needed = max(0, min_triplets_needed - len(triplets))

    for _ in range(additional_triplets_needed):
        triplet = tuple(np.random.choice(num_features, size=3, replace=False))
        if triplet not in triplets:
            triplets.append(triplet)

    return triplets


def process_organism_covering_triplets(
    organism, use_global_minmax, bin_num_list, random_seed=None
):
    """Process a single organism, calculating joint entropy for covering feature triplets.

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
        Dictionary containing triplet information and joint entropy values
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
    else:
        print("Using individual min/max values for each feature")

    # Generate covering triplets
    num_features = len(data.columns)
    print(f"\nTotal features available: {num_features}")

    triplet_indices = generate_covering_triplets(num_features, random_seed)
    print(
        f"Generated {len(triplet_indices)} triplets covering all {num_features} features"
    )

    # Create a list of feature triplets
    histone_triplets = []
    for idx1, idx2, idx3 in triplet_indices:
        feature1 = data.columns[idx1]
        feature2 = data.columns[idx2]
        feature3 = data.columns[idx3]
        histone_triplets.append((feature1, feature2, feature3))

    print(f"\nSelected triplets:")
    for i, (f1, f2, f3) in enumerate(histone_triplets, 1):
        print(f"  Triplet {i}: {f1} <-> {f2} <-> {f3}")

    # Verify coverage
    all_features_in_triplets = set()
    for f1, f2, f3 in histone_triplets:
        all_features_in_triplets.add(f1)
        all_features_in_triplets.add(f2)
        all_features_in_triplets.add(f3)

    print(f"\nCoverage verification:")
    print(f"  Total unique features in triplets: {len(all_features_in_triplets)}")
    print(f"  Total features in dataset: {num_features}")
    print(f"  All features covered: {len(all_features_in_triplets) == num_features}")

    if len(all_features_in_triplets) < num_features:
        uncovered = set(data.columns) - all_features_in_triplets
        print(
            f"  WARNING: {len(uncovered)} features not covered: {list(uncovered)[:5]}..."
        )

    # Create dictionary to store joint entropy values for each triplet and bin number
    joint_entropy_values = {}

    # Track start time for performance reporting
    start_time = time.time()

    # Process each histone triplet
    for triplet_idx, (feature1, feature2, feature3) in enumerate(
        tqdm(histone_triplets, desc=f"Processing {organism} triplets"), 1
    ):
        feature1_values = data[feature1].values
        feature2_values = data[feature2].values
        feature3_values = data[feature3].values

        # Calculate individual min/max for the current features
        feature1_min = np.min(feature1_values)
        feature1_max = np.max(feature1_values)
        feature2_min = np.min(feature2_values)
        feature2_max = np.max(feature2_values)
        feature3_min = np.min(feature3_values)
        feature3_max = np.max(feature3_values)

        min_val = global_min
        max_val = global_max

        # # Select which min/max to use based on the boolean option
        # if use_global_minmax:
        #     min_val = global_min
        #     max_val = global_max
        # else:
        #     # Use the overall min/max across all three features
        #     min_val = min(feature1_min, feature2_min, feature3_min)
        #     max_val = max(feature1_max, feature2_max, feature3_max)

        # Initialize dictionary for this triplet
        triplet_name = f"triplet_{triplet_idx}_{feature1}_vs_{feature2}_vs_{feature3}"
        joint_entropy_values[triplet_name] = {
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "entropy_values": {},
        }

        # Process each bin number
        for bin_num in bin_num_list:
            # Create bin edges using the selected min/max values
            bin_edges = np.linspace(min_val, max_val, bin_num + 1)

            # Calculate joint probability for three variables
            joint_prob_matrix = entropy_mutualinfo.joint_prob(
                feature1_values,
                feature2_values,
                bin_edges,
                input_list_3=feature3_values,
            )

            # Calculate joint entropy H(X, Y, Z)
            joint_entropy = entropy_mutualinfo.entropy_calculator(joint_prob_matrix)

            # Store the joint entropy value
            joint_entropy_values[triplet_name]["entropy_values"][
                bin_num
            ] = joint_entropy

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(
        f"Processed {len(histone_triplets)} triplets in {int(minutes)} minutes and {seconds:.2f} seconds"
    )

    # Create output directory
    minmax_type = "global" if use_global_minmax else "individual"
    output_dir_name = "covering_triplets_joint_entropy_values"
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), output_dir_name, organism
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the joint entropy values to a JSON file
    output_file = os.path.join(
        output_dir, f"covering_triplets_joint_entropy_{minmax_type}_minmax.json"
    )

    print(f"Saving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(joint_entropy_values, f, indent=4)

    # Generate a summary of joint entropy values at key bin counts
    summary_bins = [10, 50, 100, 200, 300, 400, 500]
    summary_bins = [b for b in summary_bins if b in bin_num_list]

    print("\nJoint Entropy (3-Variable) Summary for Selected Bin Numbers:")
    print("Triplet".ljust(60), end="")
    for bin_num in summary_bins:
        print(f"{bin_num:>10}", end="")
    print()

    for triplet_name in list(joint_entropy_values.keys())[:10]:
        feature_triplet = f"{joint_entropy_values[triplet_name]['feature1']} - {joint_entropy_values[triplet_name]['feature2']} - {joint_entropy_values[triplet_name]['feature3']}"
        print(feature_triplet[:60].ljust(60), end="")
        for bin_num in summary_bins:
            if bin_num in joint_entropy_values[triplet_name]["entropy_values"]:
                print(
                    f"{joint_entropy_values[triplet_name]['entropy_values'][bin_num]:>10.3f}",
                    end="",
                )
            else:
                print(f"{'N/A':>10}", end="")
        print()

    if len(joint_entropy_values) > 10:
        print("... plus", len(joint_entropy_values) - 10, "more triplets")

    # Also save a CSV summary for easier analysis
    summary_data = []
    for triplet_name, triplet_data in joint_entropy_values.items():
        row = {
            "triplet_name": triplet_name,
            "feature1": triplet_data["feature1"],
            "feature2": triplet_data["feature2"],
            "feature3": triplet_data["feature3"],
        }
        for bin_num in summary_bins:
            row[f"bin_{bin_num}"] = triplet_data["entropy_values"].get(bin_num, np.nan)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(
        output_dir, f"covering_triplets_joint_entropy_summary_{minmax_type}_minmax.csv"
    )
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary CSV saved to {summary_file}")

    # Save triplet information for reference
    triplet_info = {
        "organism": organism,
        "total_features": num_features,
        "total_triplets": len(histone_triplets),
        "all_features_covered": len(all_features_in_triplets) == num_features,
        "features_covered_count": len(all_features_in_triplets),
        "triplets": [
            {"triplet_id": i + 1, "feature1": f1, "feature2": f2, "feature3": f3}
            for i, (f1, f2, f3) in enumerate(histone_triplets)
        ],
    }

    triplet_info_file = os.path.join(
        output_dir, f"covering_triplets_info_{minmax_type}_minmax.json"
    )
    with open(triplet_info_file, "w") as f:
        json.dump(triplet_info, f, indent=4)
    print(f"Triplet information saved to {triplet_info_file}")

    return joint_entropy_values


def main():
    """Main function to calculate joint entropy values for covering feature triplets in both organisms."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate joint entropy values for random feature triplets covering all histones"
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
    print("RANDOM FEATURE TRIPLETS COVERING ALL HISTONES - JOINT ENTROPY CALCULATION")
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
        all_results[organism] = process_organism_covering_triplets(
            organism, use_global_minmax, bin_num_list, random_seed
        )

    # Calculate overall elapsed time
    overall_elapsed_time = time.time() - overall_start_time
    minutes, seconds = divmod(overall_elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)

    # Final summary
    print(f"\n{'='*80}")
    print("COVERING TRIPLETS JOINT ENTROPY CALCULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(organisms)} organisms: {', '.join(organisms)}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Save consolidated results if multiple organisms were processed
    if len(organisms) > 1:
        minmax_type = "global" if use_global_minmax else "individual"
        output_dir = os.path.join(
            ProjectPaths.get_output_files_dir(),
            "covering_triplets_joint_entropy_values",
        )
        os.makedirs(output_dir, exist_ok=True)
        consolidated_file = os.path.join(
            output_dir,
            f"consolidated_covering_triplets_joint_entropy_{minmax_type}_minmax.json",
        )

        print(f"Saving consolidated results to {consolidated_file}")
        with open(consolidated_file, "w") as f:
            json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
