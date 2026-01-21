"""
This file used to predict that target histone PTM intensities
"""

""" 
idea:

load the MMI Histone PTM data for a given organism:
 extract feature histone PTMs and target histone PTM

 using each target histone PTM:
 get random n features from the target histone random feature set text file
 predict the target histone PTM using the random n features
 calculate the prediction performance using R2 score
 save the prediction performance for each target histone PTM into json file
 with target histone PTM as a key and R2 score as value

 Note: Key consideration:
 - make the input as command line argument
 - try to run in parallel mode using joblib if the resource available,
   otherwise run in sequential mode
 - make the ml prediction by considering memory optimization and cpu optimization
"""

import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from utils import histone_mod_predictor
from path_utils.path_handler import ProjectPaths
import psutil


# getting arguments from command line with input: organism name, mmi cut off value, parallel option


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict target histone PTM from random feature selection"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="The organism to analyze",
    )

    parser.add_argument(
        "-c",
        "--mmi_cutoff_value",
        type=str,
        required=True,
        help="provide the MMI values to get the random features for target histone prediction",
    )

    parser.add_argument(
        "-g",
        "--copy_num",
        type=int,
        required=True,
        help="alternate copy number of random feature set file to use for prediction",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Run predictions in parallel (default: False)",
    )

    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


# output directory setup

randmon_ptm_prediction_dir_name = "randmom_ptm_prediction"
random_feature_dir = os.path.join(
    ProjectPaths.get_output_files_dir(), randmon_ptm_prediction_dir_name
)

# Create output directory if it doesn't exist
os.makedirs(random_feature_dir, exist_ok=True)

if __name__ == "__main__":
    args = parse_arguments()

    organism = args.organism
    mmi_cutoff_value = args.mmi_cutoff_value
    parallel_mode = args.parallel
    random_seed = args.random_seed
    num_copies = args.copy_num

    # Define file paths
    # example: output\files\ptm_prediction\human\mmi_r2\human_mmi_cutoff_-0.6_le_feature_target.json
    # 1. mmi_r2 file path with cutoff value
    mmi_file_path_with_cutoff = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "ptm_prediction",
        organism,
        "mmi_r2",
        f"{organism}_mmi_cutoff_{mmi_cutoff_value}_le_feature_target.json",
    )
    # 2. loading the processed histone modification data file path
    # path example: multivariate_ptms_analysis-\data\processed\yeast\yeast_zero_mean.csv
    input_data_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )
    # Check if input file exists
    if not os.path.exists(input_data_file_path):
        raise FileNotFoundError(f"Input data file not found: {input_data_file_path}")

    # loading the histone modification data
    feature_histone_name = data_handler.json_file_loader(mmi_file_path_with_cutoff)[
        "feature"
    ]
    # getting target histones from mmi r2 file
    target_histone_name = data_handler.json_file_loader(mmi_file_path_with_cutoff)[
        "target"
    ]

    target_r2 = {}
    # Load data ONCE before the loop
    input_data = pd.read_csv(input_data_file_path)

    # Function to predict and save R2 score for a single target histone
    def predict_and_save_r2(target_histone, verbose=False):

        # examplepath : output\files\random_feature_selection_results\yeast\-0.41_cutoff_12_features\h2ak5ac_random_features_copy_1.txt
        random_feature_file = os.path.join(
            ProjectPaths.get_output_files_dir(),
            "random_feature_selection_results",
            organism,
            f"{mmi_cutoff_value}_cutoff_{len(feature_histone_name)}_features",
            f"{target_histone.lower()}_random_features_copy_{num_copies}.txt",
        )

        # Load random features
        with open(random_feature_file, "r") as f:
            random_features = [line.strip() for line in f.readlines()]

        print(f"random features for {target_histone}: {random_features}")

        # Use pre-loaded data instead of reading file again
        random_feature_data = input_data[random_features]
        target_data = input_data[[target_histone]]

        r2_regressor = histone_mod_predictor.Histone_Regressor(
            feature=random_feature_data.to_numpy(), target=target_data.to_numpy()
        )

        r2_score = r2_regressor.error_calculator().round(3)
        if verbose:
            print(f"R2 score for {target_histone}: {r2_score}")
        return target_histone, r2_score

    # Choose between parallel and sequential processing
    print(f"Processing in {'parallel' if parallel_mode else 'sequential'} mode")
    # Calculate optimal number of workers
    n_workers = max(1, psutil.cpu_count(logical=False) - 1)  # Leave 1 core free

    if parallel_mode:
        # Use Parallel and delayed to run the loop in parallel
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(predict_and_save_r2)(target_, verbose=False)
            for target_ in target_histone_name
        )
        # Update the target_r2 dictionary with results
        target_r2 = dict(results)
    else:
        # Process sequentially
        for target_ in target_histone_name:
            target_name, score = predict_and_save_r2(target_, verbose=True)
            target_r2[target_name] = score

    # Print the final dictionary
    print("Final R2 scores:", target_r2)

    # Sort the results dictionary by R2 score in descending order
    sorted_target_r2 = dict(sorted(target_r2.items(), key=lambda x: x[1], reverse=True))

    # Generate filename with standardized naming convention including operator
    output_r2_file_path = os.path.join(
        ProjectPaths.get_output_files_dir(),  # Use ProjectPaths
        randmon_ptm_prediction_dir_name,
        organism,
        f"random_feature_mmi_cutoff_{mmi_cutoff_value}_r2_copy{num_copies}.json",
    )

    data_handler.json_file_saver(sorted_target_r2, output_r2_file_path)
    print(f"Results saved to {output_r2_file_path}")
