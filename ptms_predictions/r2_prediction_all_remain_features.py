import os
import gc
import psutil
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
- Memory optimized for large datasets

## Usage
```bash
python r2_prediction_all_remain_features.py -o <organism> [-p] [-m <memory_limit>]
```
"""


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_memory_limit(memory_limit_mb):
    """Check if memory usage exceeds limit"""
    if memory_limit_mb is None:
        return True
    current_usage = get_memory_usage()
    if current_usage > memory_limit_mb:
        print(
            f"WARNING: Memory usage ({current_usage:.1f}MB) exceeds limit ({memory_limit_mb}MB)"
        )
        return False
    return True


def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage by downcasting numeric types"""
    original_memory = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB")

    return df


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
parser.add_argument(
    "-m",
    "--memory_limit",
    type=int,
    default=None,
    help="Memory limit in MB (default: no limit)",
)
parser.add_argument(
    "-j",
    "--n_jobs",
    type=int,
    default=None,
    help="Number of parallel jobs (default: auto-detect based on memory)",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=5,
    help="Batch size for parallel processing (default: 5)",
)

# Parse the arguments
args = parser.parse_args()

# Access the variable values
organism = args.organism
parallel_mode = args.parallel
memory_limit = args.memory_limit
n_jobs = args.n_jobs
batch_size = args.batch_size

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
    Memory optimized version.

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
    try:
        # Get feature histone set by removing target
        feature_columns = list(set(histone_mod_df.columns) - {target_histone})

        # Extract feature and target data with memory optimization
        feature_histone_mod_df = histone_mod_df[feature_columns].copy()
        target_histone_df = histone_mod_df[[target_histone]].copy()

        # Optimize memory usage
        feature_histone_mod_df = optimize_dataframe_memory(feature_histone_mod_df)
        target_histone_df = optimize_dataframe_memory(target_histone_df)

        # Initialize and run prediction model
        prediction_model = histone_mod_predictor.Histone_Regressor(
            feature=feature_histone_mod_df, target=target_histone_df
        )
        error_score = prediction_model.error_calculator()

        print(
            f"Target {target_histone}: {error_score:.4f} (Memory: {get_memory_usage():.1f}MB)"
        )

        # Explicit cleanup
        del feature_histone_mod_df, target_histone_df, prediction_model
        gc.collect()

        return error_score

    except Exception as e:
        print(f"Error processing {target_histone}: {str(e)}")
        gc.collect()
        return 0.0


def run_parallel_predictions_batched(
    histone_mod_df: pd.DataFrame,
    histone_list: list,
    n_jobs: int = -1,
    batch_size: int = 5,
):
    """Run histone predictions in parallel with batching for memory efficiency

    Parameters
    ----------
    histone_mod_df : pd.DataFrame
        DataFrame containing all histone modification data
    histone_list : list
        List of histone modifications to predict
    n_jobs : int, optional
        Number of parallel jobs to run
    batch_size : int, optional
        Number of targets to process in each batch

    Returns
    -------
    list
        List of error scores for each target histone
    """
    all_results = []

    # Process in batches to manage memory
    for i in range(0, len(histone_list), batch_size):
        batch_targets = histone_list[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(histone_list) + batch_size - 1)//batch_size}"
        )

        # Check memory before each batch
        if not check_memory_limit(memory_limit):
            print("Memory limit exceeded, switching to sequential processing...")
            return run_sequential_predictions(histone_mod_df, histone_list)

        batch_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(predict_target_histone)(target, histone_mod_df)
            for target in batch_targets
        )

        all_results.extend(batch_results)

        # Force garbage collection between batches
        gc.collect()
        print(f"Batch completed. Current memory: {get_memory_usage():.1f}MB")

    return all_results


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
    # Determine optimal number of jobs based on memory
    if n_jobs is None or n_jobs == -1:
        available_memory = psutil.virtual_memory().available / (1024**2)  # MB
        # Estimate memory per job (conservative estimate)
        estimated_memory_per_job = (
            histone_mod_df.memory_usage(deep=True).sum() / (1024**2) * 2
        )
        max_jobs_by_memory = max(
            1, int(available_memory / estimated_memory_per_job / 2)
        )
        n_jobs = min(os.cpu_count(), max_jobs_by_memory)
        print(
            f"Auto-detected optimal jobs: {n_jobs} (based on {available_memory:.0f}MB available memory)"
        )

    return run_parallel_predictions_batched(
        histone_mod_df, histone_list, n_jobs, batch_size
    )


def run_sequential_predictions(histone_mod_df: pd.DataFrame, histone_list: list):
    """Run histone predictions sequentially with memory monitoring

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
    for i, target in enumerate(histone_list):
        print(f"Processing {i+1}/{len(histone_list)}: {target}")

        # Check memory before each prediction
        if not check_memory_limit(memory_limit):
            print(f"Memory limit exceeded at target {target}")
            break

        result = predict_target_histone(target, histone_mod_df)
        results.append(result)

        # Force garbage collection
        if i % 5 == 0:  # Every 5 predictions
            gc.collect()

    return results


if __name__ == "__main__":
    # Define file paths
    histone_file = f"{organism}_zero_mean.csv"
    histone_mod_file_path = os.path.join(cleaned_dir, organism, histone_file)

    # Start timing
    start_time = time.time()
    print(f"Initial memory usage: {get_memory_usage():.1f}MB")

    # Load data with memory optimization
    print("Loading data...")
    histone_mod_df = data_handler.csv_loader(histone_mod_file_path)
    print(f"Loaded DataFrame shape: {histone_mod_df.shape}")
    print(f"Memory after loading: {get_memory_usage():.1f}MB")

    # Optimize DataFrame memory usage
    histone_mod_df = optimize_dataframe_memory(histone_mod_df)

    histone_list = list(histone_mod_df.columns)
    print(f"Total features to process: {len(histone_list)}")

    # Choose between parallel and sequential processing
    print(f"Processing in {'parallel' if parallel_mode else 'sequential'} mode")
    if parallel_mode:
        # Run parallel predictions with memory optimization
        error_score_list = run_parallel_predictions(
            histone_mod_df=histone_mod_df, histone_list=histone_list, n_jobs=n_jobs
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
    print(f"Final memory usage: {get_memory_usage():.1f}MB")

    # Generate standardized filename
    output_r2_file_path = os.path.join(
        output_dir_full_path, f"{organism}_all_remaining_feature_r2.json"
    )
    json_file_saver(sorted_target_r2, output_r2_file_path)
    print(f"Results saved to {output_r2_file_path}")

    # Final cleanup
    del histone_mod_df, target_scores_dict, sorted_target_r2
    gc.collect()
    print("Cleanup complete.")
