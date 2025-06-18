# --------------------------------------------
""" This script uses pre-defined features from pseudo_inverse json file for the calculation 
of regression score using Xgboost. Features are selected from pseudo-inverse analysis 
instead of using Orthogonal matching pursuit.
"""

import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from ptms_utils.model.histone_mod_predictor import Histone_Regressor
from path_utils.path_handler import ProjectPaths

# Define the argument parser
parser = argparse.ArgumentParser(description="R² prediction using pseudo-inverse based features")

# Add the arguments
parser.add_argument(
    "-o",
    "--organism",
    type=str,
    default="yeast",
    help="The target organism (default: yeast)",
)
parser.add_argument(
    "-p",
    "--parallel",
    action="store_true",
    help="Run predictions in parallel (default: False)",
)

# Parse the arguments
args = parser.parse_args()

# Access the variable values
organism = args.organism
parallel = args.parallel

# Standard function to use in all files
def json_file_saver(data_dict: dict, absolute_file_name: str):
    """Save dictionary to JSON file with pretty formatting

    Parameters
    ----------
    data_dict : dict
        Dictionary to save
    file_path : str
        Path to output file
    """
    with open(absolute_file_name, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)# indent=4 for pretty formatting


def append_or_create_json(file_path, new_data):
    """
    Append dictionary to a JSON file or create the file if it doesn't exist.
    
    Args:
        file_path (str): The path to the JSON file.
        new_data (dict): The dictionary to append to the JSON file.
    """
    data = {}

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            # If the file is empty or corrupted, start with an empty dictionary
            data = {}

    # Merge the existing data with the new data
    data.update(new_data)

    # Write the updated data to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Successfully updated {file_path} with new data.")


# Define file paths
dir_name = r"ptm_prediction"
r2_dir_name = f"pseudo_inverse_xgb_r2"
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

# Define file paths
# Getting input histone mod data files 
histone_mod_file_path = os.path.join(cleaned_dir, organism, f"{organism}_zero_mean.csv")

# Load the histone modification data
histone_mod_df = data_handler.csv_loader(histone_mod_file_path)

# Load the pseudo-inverse feature selection data
pseudo_inverse_file = os.path.join(
    ProjectPaths.get_output_files_dir(), "pseudo_inverse", organism, f"{organism}_pseudo_inverse.json"
)

with open(pseudo_inverse_file, 'r') as f:
    pseudo_inverse_features = json.load(f)


# Create a function for the regression prediction using pre-defined features
def regression_prediction(target_histone: str):
    
    # Check if target_histone is in pseudo_inverse_features
    if target_histone not in pseudo_inverse_features:
        print(f"Warning: {target_histone} not found in pseudo_inverse_features. Skipping...")
        return None
    
    # Get the pre-defined features for this target histone
    feature_histone = pseudo_inverse_features[target_histone]
    
    # Check if all features exist in the histone_mod_df
    missing_features = [f for f in feature_histone if f not in histone_mod_df.columns]
    if missing_features:
        print(f"Warning: Features {missing_features} for {target_histone} not found in data. Skipping...")
        return None
    
    # Getting feature histone df and target histone df
    feature_df_cp = histone_mod_df[feature_histone].copy()
    target_df_cp = histone_mod_df[target_histone].copy()
    feature_names = feature_df_cp.columns

    # Create and train the model
    model = Histone_Regressor(
        feature=feature_df_cp, 
        target=target_df_cp,
    )
    
    # Evaluate model
    r2 = model.error_calculator()
    print(f"R² score for {target_histone}: {r2:.3f}")
    return r2


# Run the regression prediction for all the histones in the pseudo_inverse_features
# Two options: if parallel is true then it will run in parallel
# if parallel is false then it will run in sequence
results = {}
parallel = False

if parallel:
    results_list = Parallel(n_jobs=-1)(
        delayed(regression_prediction)(target_histone)
        for target_histone in pseudo_inverse_features.keys()
    )
    # Convert results to dictionary
    results = {h: r for h, r in zip(pseudo_inverse_features.keys(), results_list) if r is not None}
else:
    for target_histone in pseudo_inverse_features.keys():
        r2 = regression_prediction(target_histone)
        if r2 is not None:
            results[target_histone] = r2

# Sorting dictionary by R² score in descending order
results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

# Save the results in a json file
output_file = os.path.join(
    output_dir_full_path, 
    f"{organism}_pseudo_inverse_xgb_r2.json"
)
json_file_saver(results, output_file)

print(f"Analysis completed. Results saved to {output_file}")