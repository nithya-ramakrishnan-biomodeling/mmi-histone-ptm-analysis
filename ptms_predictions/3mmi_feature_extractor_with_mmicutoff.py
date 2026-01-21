import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
from itertools import combinations, chain
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from ptms_utils.model import histone_mod_predictor, histone_mod_helper
from ptms_utils.data import data_manipulator
from path_utils.path_handler import ProjectPaths

# Define the argument parser
parser = argparse.ArgumentParser(
    description="3MMI Feature extraction with cutoff values"
)
parser.add_argument(
    "-o",
    "--organism",
    type=str,
    default="human",
    help="Target organism (default: human)",
)
parser.add_argument(
    "-op",
    "--operator",
    type=str,
    default="le",
    help="Comparison operator (le/ge) (default: le)",
)
args = parser.parse_args()

# Access the variable values
organism = args.organism
operator_value = args.operator

# Define cut-off dictionaries for different organisms
cut_off_dict_all = {
    "yeast": [
        "-0.44",
        "-0.43",
        "-0.41",
    ],
    "human": [
        "-0.6",
        "-0.7",
        "-0.9",
    ],
}

dir_name = r"ptm_prediction"
r2_dir_name = r"mmi_feature"
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


histone_mod_file_path = os.path.join(cleaned_dir, organism, f"{organism}_zero_mean.csv")
histone_mod_df = data_handler.csv_loader(histone_mod_file_path)
large_neg_three_mi_df = data_handler.csv_loader(large_neg_three_mi_file)


# Standard function to use in all files
def json_file_saver(data_dict: dict, file_path: str):
    """Save dictionary to JSON file with pretty formatting

    Parameters
    ----------
    data_dict : dict
        Dictionary to save
    file_path : str
        Path to output file
    """
    with open(file_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)


def get_features_for_cutoff(df, cutoff_value, operator):
    """Extract features for a specific cutoff value

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cutoff_value : float
        Cutoff value for filtering
    operator : str
        Comparison operator ('le', 'ge', etc.)

    Returns
    -------
    list
        List of feature histone modifications
    """
    # Filter the dataframe with the specified cut-off value
    filtered_df = data_manipulator.filter_dataframe(
        df=df,
        cutoff_value=cutoff_value,
        comparison_operator=operator,
    )

    # Convert the filtered dataframe into a dictionary format
    non_nan_dict = {
        col: filtered_df[col].dropna().to_dict() for col in filtered_df.columns
    }
    non_nan_dict = {key: value for key, value in non_nan_dict.items() if value}

    # Getting all the histone mods as feature set from key and nested dict keys
    non_nan_dict_outer_keys = non_nan_dict.keys()
    non_nan_dict_inner_keys = list(chain.from_iterable(non_nan_dict.values()))
    non_nan_dict_inner_keys = list(set(non_nan_dict_inner_keys))

    # Splitting the non_nan_dict_inner_keys with "_" and combine
    non_nan_dict_inner_keys = [i.split("_") for i in non_nan_dict_inner_keys]
    non_nan_dict_inner_keys = set(chain.from_iterable(non_nan_dict_inner_keys))

    # Combine outer and inner histone mods
    non_nan_dict_outer_keys = set(non_nan_dict_outer_keys)
    non_nan_dict_outer_keys = non_nan_dict_outer_keys.union(non_nan_dict_inner_keys)

    return list(non_nan_dict_outer_keys)


# Process all cutoff values for the selected organism
print(f"Processing organism: {organism} with operator: {operator_value}")
cut_off_values = cut_off_dict_all.get(
    organism, ["-0.7"]
)  # Default to -0.7 if organism not found
feature_histone_dict = {}

for cut_off_value in cut_off_values:
    cut_off_float = float(cut_off_value)
    feature_list = get_features_for_cutoff(
        large_neg_three_mi_df, cut_off_float, operator_value
    )
    feature_histone_dict[cut_off_value] = feature_list
    print(f"Found {len(feature_list)} features for cutoff {cut_off_value}")

# Main output file - standardized naming convention
output_file = f"{organism}_3mmi_features_{operator_value}.json"
output_path = os.path.join(output_dir_full_path, output_file)

# Saving all feature histones as json file
json_file_saver(feature_histone_dict, output_path)
print(f"Features saved to {output_path}")

# Backward compatibility file - standardized naming convention
if "-0.7" in feature_histone_dict:
    back_compat_path = os.path.join(
        output_dir_full_path, f"{organism}_3mmi_features_0_7_{operator_value}.json"
    )
    json_file_saver(
        {"-0.7": feature_histone_dict["-0.7"]},
        back_compat_path,
    )
    print(f"Backward compatibility file saved to {back_compat_path}")
