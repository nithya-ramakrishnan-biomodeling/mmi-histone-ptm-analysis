# random_feature_main.py
# idea:
"""Main script to run random feature selection"""
import os
import argparse
import notebook__init__b
import pandas as pd
from path_utils.path_handler import ProjectPaths
from datahandler import dir_handler, data_handler
from random_utils.random_feature_selector import random_n_feature_selector


# getting arguments from command line
def parse_arguments():
    parser = argparse.ArgumentParser(description="Random Feature Selection")
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Target organism (e.g., 'human', 'yeast')",
    )
    parser.add_argument(
        "-mmi",
        "--mmi_cutoff_value",
        type=str,
        required=True,
        help="MMI cutoff value for feature selection",
    )
    parser.add_argument(
        "-c",
        "--num_copies",
        type=int,
        default=3,
        help="Number of random feature copies to generate (default: 3)",
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


# checking if the file path exist or raise error
def check_file_path(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")


# target and feature set selection from the given input total feature
def select_target_and_features(total_features: set, target_feature: str):
    if target_feature not in total_features:
        raise ValueError(
            f"Target feature '{target_feature}' not found in the total features."
        )

    features = total_features - {target_feature}
    return target_feature, features


# example path
# multivariate_ptms_analysis-\data\processed\human\human_zero_mean.csv
# getting the csv file path
# loading the csv file columns as total features set
# Iteration
#    for each target histone generate random n features
#   call random feature selector function
# create output directory if not exists:
# directory structure: random_feature_selection_results, for each organism, number of features
# saving the selected features as a json file for each target histone
#  saving file name as: target_histone_name_random_features.json (lower case)

if __name__ == "__main__":
    args = parse_arguments()

    organism = args.organism
    num_copies = args.num_copies
    random_seed = args.random_seed
    mmi_cutoff_value = args.mmi_cutoff_value

    # Define file paths
    input_file_path = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "ptm_prediction",
        organism,
        "mmi_r2",
        f"{organism}_mmi_cutoff_{mmi_cutoff_value}_le_feature_target.json",
    )

    # Check if input file exists
    check_file_path(input_file_path)
    input_file = data_handler.json_file_loader(input_file_path)

    feature_list = input_file["feature"]
    target_list = input_file["target"]
    total_features = set(feature_list).union(set(target_list))
    num_features = len(feature_list)

    # Output directory
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "random_feature_selection_results",
        organism,
        f"{mmi_cutoff_value}_cutoff_{num_features}_features",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each feature as target
    for target_feature in total_features:
        target, features = select_target_and_features(total_features, target_feature)

        # Generate multiple copies of random features
        for copy_num in range(1, num_copies + 1):
            # Use different random seed for each copy
            copy_random_seed = random_seed + copy_num

            # Select n random features from available features
            selected_features = random_n_feature_selector(
                features, num_features, copy_random_seed
            )

            # Save selected features with copy number in filename
            output_file = os.path.join(
                output_dir, f"{target.lower()}_random_features_copy_{copy_num}.txt"
            )
            with open(output_file, "w") as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")

            print(
                f"Selected {num_features} features for target '{target}' (copy {copy_num}/{num_copies})"
            )

    print(f"\n✓ Random feature selection completed!")
    print(f"  Total targets: {len(total_features)}")
    print(f"  Copies per target: {num_copies}")
    print(f"  Features per selection: {num_features}")
    print(f"  Output directory: {output_dir}")
