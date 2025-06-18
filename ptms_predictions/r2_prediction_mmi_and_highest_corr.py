import os
import argparse
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from ptms_utils.model import histone_mod_predictor, histone_mod_helper
from ptms_utils.data import data_manipulator
from path_utils.path_handler import ProjectPaths

"""
This file carries out the prediction of histone modifications using multivariate analysis.
which includes the following steps:
- load the histone modification data
- load the feature histones with the specified cut-off MMI value
- include highest correlated histones into the feature histones set
- load the remaining histones as target histones
- predict the histone modifications 
- save the results into a JSON file

The code uses the following arguments:
- organism: The organism (yeast/human)
- cut_off_value: The cut-off value for MMI
- operator: The logical operator cutting of the dataframe value (e.g, "le", "ge") 
- parallel: Run predictions in parallel (optional)

script is run as follows:
python new_mmi_r2.py -o yeast -c -0.41 -op le -p

or 
python new_mmi_r2.py -o yeast -c 0.41 -op ge
"""


# Define the argument parser
parser = argparse.ArgumentParser(description="Pass variables as command-line arguments")

# Add the arguments (with short and long names)
parser.add_argument(
    "-o",
    "--organism",
    type=str,
    default="human",
    help="The organism",
)
parser.add_argument(
    "-c",
    "--cut_off_value",
    type=float,
    default=-0.44,
    help="The cut-off value",
)
parser.add_argument(
    "-op",
    "--operator",
    type=str,
    required=True,
    help="The logical operator cutting of the dataframe value",
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
cut_off_value = args.cut_off_value
operator_value = args.operator
parallel_mode = args.parallel

dir_name = r"ptm_prediction"
r2_dir_name = r"mmi_and_highest_corr_r2"
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

# Define file paths
large_neg_three_mi_file = os.path.join(
    ProjectPaths.get_output_files_dir(),
    "multivariate analysis",
    organism,
    f"masked_{organism}_three_mi.csv",
)

histone_mod_file_path = os.path.join(
    cleaned_dir,
    organism,
    f"{organism}_zero_mean.csv",
)

# Load data
histone_mod_df = data_handler.csv_loader(histone_mod_file_path)
large_neg_three_mi_df = data_handler.csv_loader(large_neg_three_mi_file)

# Get the correlation dataframe and mask diagonal elements
corr_df = histone_mod_df.corr().abs()
masked_corr_df = corr_df.where(~np.eye(corr_df.shape[0], dtype=bool))
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
# Filter the dataframe with the specified cut-off value
filetered_df = data_manipulator.filter_dataframe(
    df=large_neg_three_mi_df,
    cutoff_value=cut_off_value,
    comparison_operator=operator_value,
)
print(filetered_df)

# Convert the filtered dataframe into a dictionary format
non_nan_dict = {
    col: filetered_df[col].dropna().to_dict() for col in filetered_df.columns
}
non_nan_dict = {key: value for key, value in non_nan_dict.items() if value}

# Extract unique histones from the dictionary
large_neg_histone_mods = histone_mod_helper.extract_unique_histones(non_nan_dict)

# Slice the dataframe to get feature data
feature_data_df = data_handler.df_slicer(
    file_path=histone_mod_file_path, histone_name=large_neg_histone_mods
)

# Get remaining histones that are not present in the feature histone set
target_names = set(histone_mod_df.columns) - set(feature_data_df.columns)

# Get the highest Pearson correlation coefficient for each target histone
max_target_histone_corr_feature = {}
for histone in target_names:
    max_value_excluding_diagonal = masked_corr_df.idxmax()
    max_value_histone = max_value_excluding_diagonal[histone]
    max_value = corr_df[histone][max_value_histone]
    max_target_histone_corr_feature[histone] = {max_value_histone: max_value}

# Save the highest Pearson correlation histone modification file into JSON file
max_target_histone_corr_feature_file_path = os.path.join(
    output_dir_full_path,
    f"{organism}_mmi_and_highest_corr_cutoff_{cut_off_value}_target_max_corr.json",
)
json_file_saver(max_target_histone_corr_feature, max_target_histone_corr_feature_file_path)

# Create a dictionary for feature and target histones
feature_target_dict = {
    "feature": list(feature_data_df.columns),
    "target": list(target_names),
}
print(feature_target_dict)

# Save the feature and target histones dictionary into JSON file
feature_target_file_path = os.path.join(
    output_dir_full_path,
    f"{organism}_mmi_and_highest_corr_cutoff_{cut_off_value}_feature_target.json",
)

# Initialize the results dictionary
target_r2 = {}

# Define a function to process each target
def process_target(feature_, target_):
    updated_feature = list(feature_data_df.columns)
    updated_feature.append(feature_)
    print(updated_feature)
    feature = pd.DataFrame(histone_mod_df.loc[:, updated_feature]).to_numpy()
    target = pd.DataFrame(histone_mod_df.loc[:, target_]).to_numpy()
    regressor_predictor = histone_mod_predictor.Histone_Regressor(
        feature=feature, target=target
    )
    score = regressor_predictor.error_calculator().round(2)
    print(f"Target: {target_}  R2 score: {score}")
    return target_, score

# Choose between parallel and sequential processing
print(f"Processing in {'parallel' if parallel_mode else 'sequential'} mode")
if parallel_mode:
    # Use Parallel and delayed to run the loop in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_target)(list(new_feature.keys())[0], target)
        for target, new_feature in max_target_histone_corr_feature.items()
    )
    # Update the target_r2 dictionary with results
    target_r2 = dict(results)
else:
    # Process sequentially
    for target, new_feature in max_target_histone_corr_feature.items():
        target_name, score = process_target(list(new_feature.keys())[0], target)
        target_r2[target_name] = score

# Print the final dictionary
print("Final R2 scores:", target_r2)

# Sort the results dictionary by R2 score in descending order
sorted_target_r2 = dict(sorted(target_r2.items(), key=lambda x: x[1], reverse=True))

# Generate filename with parallel/sequential indicator
mode_suffix = "parallel" if parallel_mode else "sequential"
# For MMI and highest correlation script
output_r2_file_path = os.path.join(
    output_dir_full_path,
    f"{organism}_mmi_highest_corr_cutoff_{cut_off_value}_r2.json"
)

json_file_saver(sorted_target_r2, output_r2_file_path)
