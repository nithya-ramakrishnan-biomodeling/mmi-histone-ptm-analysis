""" 
This file is used for random one feature selection 
"""

# idea:
# argument parameters:
#   organism, num_copies, random_seed,  number of features to select (n), defult is 1
# loading the input file:
# from mmi cut off file:
# get the target histone files
# get the total features set
# for each target histone:
#   select n random features from the total features set excluding the target histone
# save the selected features as a json file with:
# target histone name as key, selected features as value
# with file name based on each copy number

import os
import argparse
import notebook__init__b
import pandas as pd
from path_utils.path_handler import ProjectPaths
from datahandler import dir_handler, data_handler
from random_utils.random_feature_selector import random_n_feature_selector


# argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Random One Feature Selection")
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Organism name (e.g., human, yeast)",
    )
    parser.add_argument(
        "-n",
        "--features_number",
        type=int,
        default=1,
        help="Number of features to select (default: 1)",
    )
    parser.add_argument(
        "--num_copies",
        type=int,
        default=3,
        help="Number of random feature sets to generate",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    organism = args.organism
    n = args.features_number
    num_copies = args.num_copies
    random_seed = args.random_seed

    # Define file paths
    input_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )
    # Check if input file exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(
            f"The specified file path does not exist: {input_file_path}"
        )

    # Load the input JSON file containing features and target histones
    feature_target_data = data_handler.csv_loader(input_file_path)

    total_features = set(feature_target_data.columns)

    # Create output directory
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "random_feature_selection_results",
        organism,
        f"{n}_features",
    )
    dir_handler.dir_maker(output_dir)

    total_histone_copy = total_features.copy()

    #  I want to save the json file as a copy number where:
    # copy number has to have all the target histones as key  with n random features selected as a list value
    # in a json file

    for copy_num in range(1, num_copies + 1):
        random_feature_selection_dict = {}
        for target_histone in total_histone_copy:
            features_pool = total_features - {target_histone}
            selected_features = random_n_feature_selector(
                features_pool,
                n,
                random_seed
                + copy_num,  # Change seed for each copy to ensure different selections
            )
            random_feature_selection_dict[target_histone] = selected_features

        # Save the selected features to a JSON file
        output_file_path = os.path.join(
            output_dir, f"random_{n}_features_selection_copy_{copy_num}.json"
        )
        data_handler.json_file_saver(random_feature_selection_dict, output_file_path)
