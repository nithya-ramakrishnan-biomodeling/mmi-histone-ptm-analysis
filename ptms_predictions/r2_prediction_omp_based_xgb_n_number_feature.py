# --------------------------------------------
""" This script used for the calculation of regression score using Xgboost.
Feature histone set are selected by the Orthogonal matching pursuit
"""

import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from ptms_utils.model import omp_with_sorted_feature_imp_regressor
from path_utils.path_handler import ProjectPaths

# Define the argument parser
parser = argparse.ArgumentParser(description="R² prediction using OMP-selected features and XGBoost")
parser.add_argument("-o", "--organism", type=str, default="yeast", help="Target organism (default: yeast)")
parser.add_argument("-n", "--n_features", type=int, default=8, help="Number of features to select (default: 8)")
parser.add_argument("-p", "--parallel", action="store_true", help="Run in parallel mode (default: False)")
args = parser.parse_args()

# Access the variable values
organism = args.organism
n_features_to_select = args.n_features
parallel_mode = args.parallel

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
        json.dump(data_dict, json_file, indent=4)

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
r2_dir_name = f"omp_xgb_sorted_{n_features_to_select}_feature_importance"
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

# Define file paths
# getting input histone mod data files 
histone_mod_file_path = os.path.join(cleaned_dir, organism, f"{organism}_zero_mean.csv")

# Load the histone modification data
histone_mod_df = data_handler.csv_loader(histone_mod_file_path) 

# create a function for the regression prediction 
def regression_prediction(target_histone:str):
    
    # getting feature histone,
    # that is histone set except target histone
    feature_histone = [histone for histone in histone_mod_df.columns if histone != target_histone]
    # getting feature histone df and target histone df
    feature_df_cp = histone_mod_df[feature_histone].copy()
    target_df_cp = histone_mod_df[target_histone].copy()
    feature_names = feature_df_cp.columns

    # Create and train the model
    model = omp_with_sorted_feature_imp_regressor.HistoneRegressor(
        features=feature_df_cp, 
        target=target_df_cp,
        feature_names=feature_names,
        n_features_to_select=n_features_to_select
    )
    
    # Evaluate model
    r2 = model.calculate_error(method="r2")

    # Feature importance image
    model.plot_feature_importance(
        filename=f"{target_histone}_feature_importance.png", 
        output_dir=output_dir_full_path
    )

    # Getting feature names
    feature_names = model.get_feature_summary()
    # Save the feature summary in a JSON file
    feature_summary_file_path = os.path.join(output_dir_full_path, f"{organism}_{n_features_to_select}_feature_summary.json")
    # Append more key-value pair on the same json file 
    feature_names = {target_histone: feature_names["coefficients"]}
    append_or_create_json(feature_summary_file_path, feature_names)

    # Print feature summary
    model.print_feature_summary()
    print(f"{target_histone}: {r2:.3f}")

    return r2

print(f"Running OMP-XGB prediction for {organism} with {n_features_to_select} features")
print(f"Mode: {'Parallel' if parallel_mode else 'Sequential'}")

# Run the regression prediction for all the remaining target histones
results = {}

if parallel_mode:
    targets = list(histone_mod_df.columns)
    r2_scores = Parallel(n_jobs=-1)(
        delayed(regression_prediction)(target_histone)
        for target_histone in targets
    )
    results = dict(zip(targets, r2_scores))
else:
    for target_histone in histone_mod_df.columns:
        results[target_histone] = regression_prediction(target_histone)

# Sorting dictionary by R² scores
results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

# Save the results in a json file
# For OMP XGBoost script
output_r2_file_path = os.path.join(
    output_dir_full_path,
    f"{organism}_omp_{n_features_to_select}_features_r2.json"
)

json_file_saver(results, output_r2_file_path)
print(f"Results saved to {output_r2_file_path}")








