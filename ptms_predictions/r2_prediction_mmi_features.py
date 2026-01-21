import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from ptms_utils.model import histone_mod_predictor
from path_utils.path_handler import ProjectPaths

"""
# MMI Regressor Module

This module implements MMI-based regression for histone modification prediction,
using features extracted from the 3MMI feature extractor.

## Overview
- Uses pre-extracted features based on MMI from 3mmi_feature_extractor file
- Predicts histone modifications using these features
- Supports parallel processing for efficiency (optional)

## Main Components

### 1. Command Line Interface
Required arguments:
- `-o, --organism`: Target organism (e.g., 'human', 'yeast')
- `-c, --cut_off_value`: Threshold for feature selection
- `-op, --operator`: Comparison operator for filtering, (eg., "le", "ge")

Optional arguments:
- `-p, --parallel`: Enable parallel processing (default: False)

### 2. Data Flow
1. Loads zero-mean normalized data and pre-extracted MMI features
2. Processes features and targets
3. Predicts target modifications
4. Calculates and saves R² scores

### 3. Output Files
- Feature-target pairs JSON
- R² scores for predictions
- Sorted results by performance

## Usage
```bash
python r2_prediction_mmi_features.py -o human -c -0.5 -op "le" -p
"""

# Define the argument parser
parser = argparse.ArgumentParser(
    description="MMI-based histone modification prediction"
)

# Add the arguments (with short and long names)
parser.add_argument(
    "-o",
    "--organism",
    type=str,
    required=True,
    help="The organism to analyze",
)
parser.add_argument(
    "-c",
    "--cut_off_value",
    type=float,
    required=True,
    help="The cut-off value for MMI feature selection",
)
parser.add_argument(
    "-op",
    "--operator",
    type=str,
    required=True,
    help="The logical operator for filtering ('le', 'ge')",
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

# Directory structure setup with standardized naming
dir_name = r"ptm_prediction"
r2_dir_name = r"mmi_r2"  # Output directory for R² results
mmi_feature_dir = r"mmi_feature"  # Directory where 3MMI features are stored
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

# Define file paths
histone_mod_file_path = os.path.join(cleaned_dir, organism, f"{organism}_zero_mean.csv")

# Path to 3MMI features file - corrected name
mmi_features_path = os.path.join(
    ProjectPaths.get_output_files_dir(),
    dir_name,
    organism,
    mmi_feature_dir,
    f"{organism}_3mmi_features_{operator_value}.json",  # Standardized naming
)


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


# Load data
print(f"Loading data for organism {organism}")
histone_mod_df = data_handler.csv_loader(histone_mod_file_path)

# Load pre-extracted 3MMI features
print(f"Loading 3MMI features from {mmi_features_path}")
try:
    mmi_features_dict = data_handler.json_file_loader(mmi_features_path)

    # Get features for the specified cutoff
    cut_off_str = str(cut_off_value)
    if cut_off_str not in mmi_features_dict:
        print(f"Warning: Cutoff {cut_off_str} not found in features file")
        print(f"Available cutoffs: {list(mmi_features_dict.keys())}")
        print(f"Using closest available cutoff")
        # Find closest available cutoff
        closest = min(
            mmi_features_dict.keys(), key=lambda x: abs(float(x) - cut_off_value)
        )
        print(f"Selected cutoff: {closest}")
        cut_off_str = closest

    large_neg_histone_mods = mmi_features_dict[cut_off_str]
    print(f"Found {len(large_neg_histone_mods)} features for cutoff {cut_off_str}")

except Exception as e:
    print(f"Error loading 3MMI features: {e}")
    print("Make sure to run 3mmi_feature_exratcor_with_mmicutoff.py first")
    exit(1)

# Use the features to create a dataframe
feature_data_df = histone_mod_df[large_neg_histone_mods]

# Get remaining histones that are not present in the feature histone set
target_names = set(histone_mod_df.columns) - set(feature_data_df.columns)

# Create a dictionary for feature and target histones
feature_target_dict = {
    "feature": list(feature_data_df.columns),
    "target": list(target_names),
}
print(f"Features: {len(feature_target_dict['feature'])}")
print(f"Targets: {len(feature_target_dict['target'])}")

# Save the feature and target histones dictionary into JSON file with standardized naming
feature_target_file_path = os.path.join(
    output_dir_full_path,
    f"{organism}_mmi_cutoff_{cut_off_str}_{operator_value}_feature_target.json",
)
json_file_saver(feature_target_dict, feature_target_file_path)
print(f"Feature-target pairs saved to {feature_target_file_path}")

# Initialize the results dictionary
target_r2 = {}


# Define a function to process each target
def process_target(target_):
    feature = feature_data_df.to_numpy()
    target = pd.DataFrame(histone_mod_df.loc[:, target_]).to_numpy()
    regressor_predictor = histone_mod_predictor.Histone_Regressor(
        feature=feature, target=target
    )
    score = regressor_predictor.error_calculator().round(3)
    print(f"Target: {target_}  R2 score: {score}")
    return target_, score


# Choose between parallel and sequential processing
print(f"Processing in {'parallel' if parallel_mode else 'sequential'} mode")
if parallel_mode:
    # Use Parallel and delayed to run the loop in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_target)(target_) for target_ in target_names
    )
    # Update the target_r2 dictionary with results
    target_r2 = dict(results)
else:
    # Process sequentially
    for target_ in target_names:
        target_name, score = process_target(target_)
        target_r2[target_name] = score

# Print the final dictionary
print("Final R2 scores:", target_r2)

# Sort the results dictionary by R2 score in descending order
sorted_target_r2 = dict(sorted(target_r2.items(), key=lambda x: x[1], reverse=True))

# Generate filename with standardized naming convention including operator
output_r2_file_path = os.path.join(
    output_dir_full_path,
    f"{organism}_mmi_cutoff_{cut_off_str}_{operator_value}_r2.json",
)
json_file_saver(sorted_target_r2, output_r2_file_path)
print(f"Results saved to {output_r2_file_path}")
