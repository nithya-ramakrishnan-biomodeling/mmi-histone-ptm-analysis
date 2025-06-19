#!/usr/bin/env python3
"""
Calculate mutual information between predicted and actual values
using MMI-selected features at specific cutoffs.
"""
import os
import sys 
import pandas as pd
import numpy as np
import json
import argparse
import time
from tqdm import tqdm
from joblib import Parallel, delayed
import notebook__init__b
from path_utils.path_handler import ProjectPaths    
from utils.histone_mod_predictor import Histone_Regressor
from datahandler import data_handler, dir_handler
from utils import entropy_mutualinfo

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate mutual information using MMI-selected features")
    parser.add_argument("-o", "--organism", type=str, required=True,
                        help="Target organism (e.g., 'human', 'yeast')")
    parser.add_argument("-c", "--cutoff", type=float, required=True,
                        help="MMI cutoff value")
    parser.add_argument("-op", "--operator", type=str, required=True,
                        help="Operator used for MMI cutoff (e.g., 'le', 'ge')")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Enable parallel processing")
    parser.add_argument("--cpu_limit", type=int, default=2,
                        help="Number of CPU cores to use (default: 2)")
    return parser.parse_args()

def save_mi_values_to_json(file_path, mi_values):
    """Save mutual information values to JSON file"""
    with open(file_path, "w") as f:
        json.dump(mi_values, f, indent=4)

def calculate_mi_for_target(target_name, feature_df, df, bin_edges):
    """Calculate mutual information for a single target"""
    try:
        # Get the target values
        target_values = df[target_name].values
        
        # Predict the values
        ml_model = Histone_Regressor(feature=feature_df, target=target_values)
        y_test, y_pred = ml_model.xgb_regressor_pred()
        
        # Calculate mutual information
        mi_value = entropy_mutualinfo.mi_betwn_uandy(arr1=y_test, arr2=y_pred, spaced_bin_edges=bin_edges)
        
        print(f"Target: {target_name}, MI: {mi_value}")
        return target_name, mi_value
    except Exception as e:
        print(f"Error processing {target_name}: {e}")
        return target_name, f"Error: {str(e)}"

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    organism = args.organism
    cutoff = args.cutoff
    operator = args.operator  # Added operator parameter
    parallel_mode = args.parallel
    cpu_limit = args.cpu_limit
    
    print(f"Calculating mutual information for {organism} with cutoff {cutoff}")
    
    # Set up directories
    data_file_name = f"{organism}_zero_mean.csv"
    data_file_path = os.path.join(ProjectPaths.get_cleaned_data_dir(), organism, data_file_name)
    
    # Create output directory
    output_dir = os.path.join(ProjectPaths.get_output_files_dir(), 
                              "mi_values_from_mmi_cutoff", organism)
    dir_handler.dir_maker(output_dir)
    
    # Use the correct directory and filename pattern
    feature_target_json_file_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), 
        "ptm_prediction", organism, "mmi_r2",
        f"{organism}_mmi_cutoff_{cutoff}_{operator}_feature_target.json")  # Include operator
    
    # Output file path should also include the operator
    mi_values_file = os.path.join(output_dir, f"mi_values_{str(cutoff)}_{operator}.json")
    
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
    
    # Check if the feature target file exists
    if not os.path.exists(feature_target_json_file_dir):
        print(f"Error: Feature target file not found at {feature_target_json_file_dir}")
        print("Please run the MMI feature selection first")
        sys.exit(1)
    
    # Load feature and target lists
    feature_target_dict = data_handler.json_file_loader(feature_target_json_file_dir)
    features = feature_target_dict.get("feature", [])
    targets = feature_target_dict.get("target", [])
    
    print(f"Loaded {len(features)} features and {len(targets)} targets")
    
    # Load the data file
    try:
        df = pd.read_csv(data_file_path, index_col=0)
        print(f"Loaded data with shape {df.shape}")
        
        # Extract feature data once (optimization)
        feature_df = df[features].values
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    # Process targets
    mi_values = {}
    
    if parallel_mode:
        print(f"Processing {len(targets)} targets in parallel using {cpu_limit} cores")
        try:
            # Run in parallel
            results = Parallel(n_jobs=min(cpu_limit, len(targets)))(
                delayed(calculate_mi_for_target)(target, feature_df, df, bin_edges)
                for target in targets
            )
            
            # Process results
            for target_name, mi_value in results:
                mi_values[target_name] = mi_value
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            print("Falling back to sequential processing")
            # Fall back to sequential processing
            for target_name in tqdm(targets, desc="Processing targets"):
                target_name, mi_value = calculate_mi_for_target(target_name, feature_df, df, bin_edges)
                mi_values[target_name] = mi_value
    else:
        print(f"Processing {len(targets)} targets sequentially")
        for target_name in tqdm(targets, desc="Processing targets"):
            target_name, mi_value = calculate_mi_for_target(target_name, feature_df, df, bin_edges)
            mi_values[target_name] = mi_value
    
    # Filter out error results
    valid_results = {k: v for k, v in mi_values.items() if not (isinstance(v, str) and v.startswith("Error:"))}
    error_results = {k: v for k, v in mi_values.items() if isinstance(v, str) and v.startswith("Error:")}
    
    if error_results:
        print(f"WARNING: {len(error_results)} targets had errors and were excluded from final results")
        error_file = os.path.join(output_dir, f"error_log_{str(cutoff)}.json")
        save_mi_values_to_json(error_file, error_results)
        print(f"Error details saved to {error_file}")
    
    # Sort by mutual information value
    sorted_mi_values = dict(
        sorted(valid_results.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
    )
    
    # Save results
    save_mi_values_to_json(mi_values_file, sorted_mi_values)
    print(f"Mutual information values saved to {mi_values_file}")
    
    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    print(f"Runtime: {int(minutes)} minutes and {seconds:.2f} seconds")

if __name__ == "__main__":
    main()