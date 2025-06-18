import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
import time
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil
import notebook__init__b
from utils.histone_mod_predictor import Histone_Regressor
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths
from utils import entropy_mutualinfo

# Define the argument parser
parser = argparse.ArgumentParser(description="Calculate mutual information using all remaining features approach")

parser.add_argument(
    "-o",
    "--organism",
    type=str,
    required=True,
    help="The organism to analyze",
)
parser.add_argument(
    "--cpu_limit",
    type=int,
    default=2,  # Default to 2 cores for low-spec systems
    help="Number of CPU cores to use (default: 2)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=5,  # Small batch size for low memory usage
    help="Number of histone modifications to process in one batch (default: 5)",
)
parser.add_argument(
    "--sequential",
    action="store_true",
    help="Use sequential processing instead of parallel (slower but uses less memory)",
)
parser.add_argument(
    "--memory_limit",
    type=int,
    default=4000,  # 4GB memory limit by default
    help="Memory limit in MB for the process",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from previous run if available",
)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Force clean start and remove any previous temporary results",
)

# Parse the arguments
args = parser.parse_args()

# Access the variable values
organism = args.organism
cpu_limit = args.cpu_limit
batch_size = args.batch_size
sequential_mode = args.sequential
memory_limit = args.memory_limit
resume_from_previous = args.resume
clean_start = args.clean

# Set up directories
dir_name = r"ptm_prediction"
mi_dir_name = r"all_remain_mi"
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, mi_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)

# Load the pre-defined mutual information binning parameters
try:
    mi_file = os.path.join(ProjectPaths.get_cleaned_data_dir(), r"mi_bin_info.json")
    mi_dict = data_handler.json_file_loader(mi_file)
    mi_dict = mi_dict[organism]
    global_max = mi_dict["global_max"]
    global_min = mi_dict["global_min"]
    bin_edges_num = mi_dict["bin_num"]
    print(f"Using pre-defined MI binning parameters:")
    print(f"  Global max: {global_max}")
    print(f"  Global min: {global_min}")
    print(f"  Bin count: {bin_edges_num}")
    
    # Pre-compute bin edges once for efficiency
    bin_edges = np.linspace(global_min, global_max, num=bin_edges_num+1)
except Exception as e:
    print(f"Error loading MI binning parameters: {e}")
    print("Cannot continue without binning parameters")
    sys.exit(1)

def json_file_saver(save_dict: dict, absolute_file_name: str):
    """Save dictionary to JSON file with pretty formatting."""
    with open(absolute_file_name, "w") as json_file:
        json.dump(save_dict, json_file, indent=4)

def check_memory_usage():
    """Monitor memory usage and return usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

def calculate_mutual_info(target_histone: str, histone_mod_df: pd.DataFrame):
    """
    Calculate mutual information between actual and predicted values for a target histone modification
    using all other modifications as features.
    
    Parameters
    ----------
    target_histone : str
        Name of histone modification to predict
    
    histone_mod_df : pd.DataFrame
        DataFrame containing all histone modification data
    
    Returns
    -------
    tuple
        (target_histone, mutual_info_value, self_info_value)
    """
    try:
        # Get feature histone set by removing target
        feature_columns = list(set(histone_mod_df.columns) - {target_histone})
        
        # Extract feature data
        feature_histone_mod_df = histone_mod_df[feature_columns].copy()
        
        # Extract target data
        target_values = histone_mod_df[target_histone].values
        
        # Initialize and run prediction model
        ml_model = Histone_Regressor(feature=feature_histone_mod_df, target=target_values)
        
        # Get actual and predicted values
        y_test, y_pred = ml_model.xgb_regressor_pred()

        # Calculate MI between y_test and y_pred (original)
        mi_value = entropy_mutualinfo.mi_betwn_uandy(arr1=y_test, arr2=y_pred, spaced_bin_edges=bin_edges)
        
        # Calculate MI between y_test and itself (self-information)
        self_info_value = entropy_mutualinfo.mi_betwn_uandy(arr1=y_test, arr2=y_test, spaced_bin_edges=bin_edges)
        
        # Clean up to free memory
        del feature_histone_mod_df, ml_model, y_test, y_pred
        gc.collect()
        
        return target_histone, mi_value, self_info_value
    except Exception as e:
        return target_histone, f"Error: {str(e)}", f"Error: {str(e)}"

def run_batch_predictions_sequential(histone_mod_df: pd.DataFrame, histone_batch: list):
    """Run mutual information calculations sequentially for a batch of targets"""
    results = []
    for target in tqdm(histone_batch, desc="Processing targets"):
        result = calculate_mutual_info(target, histone_mod_df)
        results.append(result)
        # Force garbage collection after each calculation
        gc.collect()
    return results

def run_batch_predictions_parallel(
    histone_mod_df: pd.DataFrame, histone_batch: list, n_jobs: int = 2
):
    """Run mutual information calculations in parallel for a batch of targets"""
    try:
        # Limit n_jobs to a safe value
        n_jobs = min(n_jobs, len(histone_batch), cpu_limit)
        
        results = Parallel(n_jobs=n_jobs, max_nbytes='50M')(
            delayed(calculate_mutual_info)(target, histone_mod_df)
            for target in histone_batch
        )
        return results
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        return run_batch_predictions_sequential(histone_mod_df, histone_batch)

def clean_temp_files(directory, prefix):
    """Remove all temporary result files for a clean start"""
    removed_count = 0
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and "batch" in filename and filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    return removed_count

if __name__ == "__main__":
    print(f"Starting MI calculation with {'sequential mode' if sequential_mode else f'{cpu_limit} CPU cores'}")
    print(f"Batch size: {batch_size}, Memory limit: {memory_limit} MB")
    print(f"System has {os.cpu_count()} logical processors available")
    print(f"Current memory usage: {check_memory_usage():.2f} MB")
    
    # Clean temporary files if requested
    if clean_start:
        batch_file_prefix = f"{organism}_batch"
        removed = clean_temp_files(output_dir_full_path, batch_file_prefix)
        print(f"Cleaned {removed} temporary result files for a fresh start")
    
    # Define file paths
    histone_file = f"{organism}_zero_mean.csv"
    histone_mod_file_path = os.path.join(cleaned_dir, organism, histone_file)

    # Start timing
    start_time = time.time()
    
    # Load data with optimizations
    print("Loading data...")
    try:
        # Try to optimize data types for memory efficiency
        histone_mod_df = data_handler.csv_loader(histone_mod_file_path)
        
        # Convert to float32 to save memory (half the memory usage of float64)
        for col in histone_mod_df.columns:
            if histone_mod_df[col].dtype == 'float64':
                histone_mod_df[col] = histone_mod_df[col].astype('float32')
        
        histone_list = list(histone_mod_df.columns)
        
        print(f"Loaded data with {len(histone_list)} histone modifications")
        print(f"DataFrame shape: {histone_mod_df.shape}")
        print(f"Memory usage: {histone_mod_df.memory_usage().sum() / 1024 / 1024:.2f} MB")
        print(f"Current process memory: {check_memory_usage():.2f} MB")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Create batches to reduce memory pressure
    batches = [histone_list[i:i+batch_size] for i in range(0, len(histone_list), batch_size)]
    print(f"Processing in {len(batches)} batches")
    
    # Dictionary to store results
    mi_values = {}
    self_info_values = {}  # New dictionary for self-information
    
    # Check if resuming from previous run
    latest_batch = 0
    if resume_from_previous:
        for i in range(len(batches), 0, -1):
            temp_path = os.path.join(
                output_dir_full_path, f"{organism}_batch_{i}_temp_results.json"
            )
            if os.path.exists(temp_path):
                print(f"Found previous results, resuming from batch {i}")
                try:
                    with open(temp_path, 'r') as f:
                        temp_dict = json.load(f)
                    
                    # Validate that this contains actual results
                    valid_results = True
                    for key, val in temp_dict.items():
                        if isinstance(val, str) and val.startswith("Error:"):
                            print(f"Warning: Previous result for {key} contains an error. Will recalculate.")
                            valid_results = False
                            break
                    
                    if valid_results:
                        mi_values = temp_dict
                        latest_batch = i
                        print(f"Successfully loaded {len(mi_values)} previous results")
                        break
                    else:
                        print("Previous results contain errors, starting from an earlier batch")
                except Exception as e:
                    print(f"Error loading previous results: {e}")
                    print("Starting from scratch")
    
    # Process each batch
    for batch_idx, batch in enumerate(batches[latest_batch:], start=latest_batch):
        print(f"\nProcessing batch {batch_idx+1}/{len(batches)} with {len(batch)} targets")
        print(f"Current memory usage: {check_memory_usage():.2f} MB")
        
        # Check if memory usage is too high before processing
        if check_memory_usage() > memory_limit:
            print(f"WARNING: Memory usage ({check_memory_usage():.2f} MB) exceeds limit ({memory_limit} MB)")
            print("Forcing garbage collection...")
            gc.collect()
            print(f"Memory usage after collection: {check_memory_usage():.2f} MB")
            
            # If still too high, switch to sequential mode
            if check_memory_usage() > memory_limit and not sequential_mode:
                print("Switching to sequential processing to reduce memory usage")
                sequential_mode = True
        
        # Remove already processed targets from batch
        batch_to_process = [target for target in batch if target not in mi_values]
        if len(batch_to_process) < len(batch):
            print(f"Skipping {len(batch) - len(batch_to_process)} already processed targets")
        
        if not batch_to_process:
            print("All targets in this batch were already processed, skipping")
            continue
            
        # Choose processing method based on mode
        if sequential_mode:
            batch_results = run_batch_predictions_sequential(
                histone_mod_df=histone_mod_df, 
                histone_batch=batch_to_process
            )
        else:
            batch_results = run_batch_predictions_parallel(
                histone_mod_df=histone_mod_df, 
                histone_batch=batch_to_process,
                n_jobs=cpu_limit
            )
        
        # Add batch results to the main results dictionary
        valid_results = 0
        for target, mi_value, self_info_value in batch_results:  # Note the additional return value
            mi_values[target] = mi_value
            self_info_values[target] = self_info_value
            if not isinstance(mi_value, str):  # It's a valid numerical value
                valid_results += 1
            print(f"Target {target}: MI = {mi_value}, Self-Info = {self_info_value}")
        
        print(f"Batch complete: {valid_results}/{len(batch_to_process)} valid results")
        
        # Force garbage collection after each batch
        gc.collect()
        print(f"Memory after batch: {check_memory_usage():.2f} MB")
        
        # Intermediate save to prevent data loss
        temp_output_path = os.path.join(
            output_dir_full_path, f"{organism}_batch_{batch_idx+1}_temp_results.json"
        )
        json_file_saver(save_dict=mi_values, absolute_file_name=temp_output_path)
    
    # Filter out error results for the final output
    valid_results = {k: v for k, v in mi_values.items() if not (isinstance(v, str) and v.startswith("Error:"))}
    error_results = {k: v for k, v in mi_values.items() if isinstance(v, str) and v.startswith("Error:")}
    
    if error_results:
        print(f"WARNING: {len(error_results)} targets had errors and were excluded from final results")
        for target, error in error_results.items():
            print(f"  {target}: {error}")
    
    # Sort results by mutual information value
    sorted_mi_values = dict(
        sorted(valid_results.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
    )

    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Runtime: {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds")

    # Save final results
    output_mi_file_path = os.path.join(
        output_dir_full_path, f"{organism}_all_remaining_feature_mi.json"
    )
    json_file_saver(save_dict=sorted_mi_values, absolute_file_name=output_mi_file_path)
    print(f"Results saved to {output_mi_file_path}")

    # Filter out error results for self-information
    valid_self_info = {k: v for k, v in self_info_values.items() if not (isinstance(v, str) and v.startswith("Error:"))}
    error_self_info = {k: v for k, v in self_info_values.items() if isinstance(v, str) and v.startswith("Error:")}

    # Sort results by self-information value
    sorted_self_info = dict(
        sorted(valid_self_info.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
    )

    # Save self-information results
    self_info_output_path = os.path.join(
        output_dir_full_path, f"{organism}_self_information_values.json"
    )
    json_file_saver(save_dict=sorted_self_info, absolute_file_name=self_info_output_path)
    print(f"Self-information results saved to {self_info_output_path}")