# Standard library imports
import os
import json
import argparse
import time

# Third-party imports
import pandas as pd
from joblib import Parallel, delayed

# Local application imports
import notebook__init__b
from ptms_utils.model import histone_mod_predictor
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths

"""
This file carries out the prediction of histone modifications using PCA transformed features.
which includes the following steps:
- Load the histone modification data
- Define PCA features (v1, v2, v3, v4 and v5 ) and target histones
- Run predictions in parallel or sequentially
- Save the results into a JSON file
"""

# Define command line arguments
parser = argparse.ArgumentParser(description="R² prediction using PCA transformed features")
parser.add_argument("-o", "--organism", type=str, default="yeast", help="Target organism (default: yeast)")
parser.add_argument("-m", "--method", type=str, default="mi", help="Method used (mi or cov) (default: mi)")
parser.add_argument("-p", "--parallel", action="store_true", help="Run in parallel mode (default: False)")
parser.add_argument("-j", "--jobs", type=int, default=-2, help="Number of parallel jobs if using parallel mode (default: -2)")
args = parser.parse_args()

# Global configuration variables
organism = args.organism
method = args.method
parallel_mode = args.parallel
n_jobs = args.jobs

dir_name = r"ptm_prediction"  # Directory for PTM prediction results
r2_dir_name = r"newly_added_pca"  # Subdirectory for PCA results
main_dir = ProjectPaths.get_project_root()  # Get project root directory
cleaned_dir = ProjectPaths.get_cleaned_data_dir()  # Get cleaned data directory

# Construct output directory path
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

def json_file_saver(save_dict: dict, absolute_file_name: str):
    """
    Save dictionary to JSON file with pretty formatting.
    
    Args:
        save_dict: Dictionary to be saved
        absolute_file_name: Full path for the JSON file
    """
    with open(absolute_file_name, "w") as json_file:
        json.dump(save_dict, json_file, indent=4)

def predict_target_histone(
    target_histone: str, histone_mod_df: pd.DataFrame, feature_list: list
):
    """
    Predict a target histone modification using PCA features.
    
    Args:
        target_histone: Name of histone modification to predict
        histone_mod_df: DataFrame with all histone modifications
        feature_list: List of PCA features to use for prediction
    
    Returns:
        float: R² score for the prediction
    """
    # Separate features and target
    feature_histone_mod_df = histone_mod_df[feature_list]
    target_histone_df = pd.DataFrame(histone_mod_df[target_histone])

    # Initialize and run regression model
    prediction_model = histone_mod_predictor.Histone_Regressor(
        feature=feature_histone_mod_df, target=target_histone_df
    )
    error_score = prediction_model.error_calculator()

    # Print results for monitoring
    print(f"Histone mods: {target_histone}")
    print(f"Target {target_histone}: {error_score}")

    return error_score

def run_predictions(
    histone_mod_df: pd.DataFrame,
    feature_list: list,
    target_list: list,
    parallel: bool = False,
    n_jobs: int = -2,
):
    """
    Run multiple histone predictions in parallel or sequentially.
    
    Args:
        histone_mod_df: DataFrame with all histone modifications
        feature_list: List of PCA features to use
        target_list: List of target histones to predict
        parallel: Whether to run in parallel (True) or sequential (False) mode
        n_jobs: Number of parallel jobs if using parallel mode
    
    Returns:
        list: R² scores for all target predictions
    """
    print(f"Running in {'parallel' if parallel else 'sequential'} mode")
    
    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(predict_target_histone)(target, histone_mod_df, feature_list)
            for target in target_list
        )
    else:
        results = []
        for target in target_list:
            results.append(predict_target_histone(target, histone_mod_df, feature_list))
    
    return results

if __name__ == "__main__":
    # Define input file paths
    histone_file = f"histone_{method}_with_eig_transf_features.csv"
    histone_mod_file_path = os.path.join(
        ProjectPaths.get_output_files_dir(), "pca", organism, histone_file
    )

    # Start runtime measurement
    start_time = time.time()
    
    print(f"Starting R² prediction for {organism} using {method} method")
    print(f"Mode: {'Parallel' if parallel_mode else 'Sequential'}")

    # Load histone modification data
    histone_mod_df = data_handler.csv_loader(histone_mod_file_path)
    
    # Define PCA features and target histones
    feature_list = ["v1", "v2", "v3", "v4", "v5"]  # PCA components
    target_list = [hist for hist in histone_mod_df.columns if hist not in feature_list]
    
    # Run predictions with selected mode (parallel or sequential)
    error_score_list = run_predictions(
        histone_mod_df=histone_mod_df,
        feature_list=feature_list,
        target_list=target_list,
        parallel=parallel_mode,
        n_jobs=n_jobs
    )

    # Create dictionary of results and sort by R² score
    target_scores_dict = dict(zip(target_list, error_score_list))
    sorted_target_r2 = dict(
        sorted(target_scores_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Calculate and print runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    print(f"Runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

    # Save results to JSON file
    parallel_tag = "parallel" if parallel_mode else "sequential"
    # For PCA script
    output_r2_file_path = os.path.join(
        output_dir_full_path,
        f"{organism}_{method}_pca_r2.json"
    )

    json_file_saver(save_dict=sorted_target_r2, absolute_file_name=output_r2_file_path)
    print(f"Results saved to: {output_r2_file_path}")
