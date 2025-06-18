import os
import pandas as pd
import json
import argparse
import time
from joblib import Parallel, delayed
import notebook__init__b
from ptms_utils.model import histone_mod_predictor
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths

"""
# All Remaining Predictor Module

This module implements prediction of histone modifications using all other modifications.

## Overview
- Predicts each histone modification using all others as features, (n-1) features, 
  where n is the total number of histone modifications
- Can run in parallel or sequential mode
- Calculates and stores R² scores
- Supports multiple organisms

## Usage
```bash
python r2_prediction_all_remain_features.py -o <organism> [-p]
```
"""
# Define the argument parser
parser = argparse.ArgumentParser(description="Pass variables as command-line arguments")

# Add the arguments (with short and long names)
parser.add_argument(
    "-o",
    "--organism",
    type=str,
    required=True,
    help="The organism",
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
parallel_mode = args.parallel

dir_name = r"ptm_prediction"
r2_dir_name = r"remaining_r2"
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

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

def cutoff_r2_score_file_loader(r2_json_file_path: str):
    """Load and return keys from a JSON file containing R² scores

    Parameters
    ----------
    r2_json_file_path : str
        Path to the JSON file containing R² scores

    Returns
    -------
    list
        List of keys from the R² score dictionary
    """
    cut_off_r2_score_dict = data_handler.json_file_loader(r2_json_file_path)
    return list(cut_off_r2_score_dict.keys())


def predict_target_histone(target_histone: str, histone_mod_df: pd.DataFrame):
    """
    Predict one histone modification using all others as features.

    Parameters
    ----------
    target_histone : str
        Name of histone modification to predict
    
    histone_mod_df : pd.DataFrame
        DataFrame containing all histone modification data
    
    Returns
    -------
    float
        R² score for prediction
    """
    # Get feature histone set by removing target
    feature_columns = list(set(histone_mod_df.columns) - {target_histone})

    # Extract feature and target data
    feature_histone_mod_df = histone_mod_df[feature_columns]
    target_histone_df = pd.DataFrame(histone_mod_df[target_histone])

    # Initialize and run prediction model
    prediction_model = histone_mod_predictor.Histone_Regressor(
        feature=feature_histone_mod_df, target=target_histone_df
    )
    error_score = prediction_model.error_calculator()

    print(f"Target {target_histone}: {error_score}")

    return error_score


def run_parallel_predictions(
    histone_mod_df: pd.DataFrame, histone_list: list, n_jobs: int = -1
):
    """Run histone predictions in parallel using joblib

    Parameters
    ----------
    histone_mod_df : pd.DataFrame
        DataFrame containing all histone modification data
    histone_list : list
        List of histone modifications to predict
    n_jobs : int, optional
        Number of parallel jobs to run, -1 means using all processors

    Returns
    -------
    list
        List of error scores for each target histone
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(predict_target_histone)(target, histone_mod_df)
        for target in histone_list
    )
    return results


def run_sequential_predictions(
    histone_mod_df: pd.DataFrame, histone_list: list
):
    """Run histone predictions sequentially

    Parameters
    ----------
    histone_mod_df : pd.DataFrame
        DataFrame containing all histone modification data
    histone_list : list
        List of histone modifications to predict

    Returns
    -------
    list
        List of error scores for each target histone
    """
    results = []
    for target in histone_list:
        result = predict_target_histone(target, histone_mod_df)
        results.append(result)
    return results


if __name__ == "__main__":
    # Define file paths
    histone_file = f"{organism}_zero_mean.csv"
    histone_mod_file_path = os.path.join(cleaned_dir, organism, histone_file)

    # Start timing
    start_time = time.time()

    # Load data
    histone_mod_df = data_handler.csv_loader(histone_mod_file_path)
    histone_list = list(histone_mod_df.columns)

    # Choose between parallel and sequential processing
    print(f"Processing in {'parallel' if parallel_mode else 'sequential'} mode")
    if parallel_mode:
        # Run parallel predictions
        error_score_list = run_parallel_predictions(
            histone_mod_df=histone_mod_df, histone_list=histone_list
        )
    else:
        # Run sequential predictions
        error_score_list = run_sequential_predictions(
            histone_mod_df=histone_mod_df, histone_list=histone_list
        )

    # Create and sort results dictionary
    target_scores_dict = dict(zip(histone_list, error_score_list))
    sorted_target_r2 = dict(
        sorted(target_scores_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    print(f"Runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

    # Generate standardized filename
    output_r2_file_path = os.path.join(
        output_dir_full_path, f"{organism}_all_remaining_feature_r2.json"
    )
    json_file_saver(sorted_target_r2, output_r2_file_path)
    print(f"Results saved to {output_r2_file_path}")
