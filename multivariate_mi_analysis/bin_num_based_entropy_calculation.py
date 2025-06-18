"""
Entropy Calculation for All Histone PTMs Based on Varying Bin Numbers

This script automatically calculates entropy values for all histone PTMs across a range of bin numbers
for both human and yeast organisms, helping to determine the optimal bin count for mutual information calculations.
"""
import os
import sys 
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import time
import argparse

import notebook__init__b
from path_utils.path_handler import ProjectPaths    
from datahandler import data_handler
import entropy_mutualinfo

def process_organism(organism, use_global_minmax, bin_num_list):
    """Process a single organism, calculating entropy for all its PTMs.
    
    Parameters
    ----------
    organism : str
        Name of the organism to process (e.g., 'human', 'yeast')
    use_global_minmax : bool
        Whether to use global min/max values for all features
    bin_num_list : list
        List of bin numbers to test
        
    Returns
    -------
    dict
        Dictionary of entropy values for each feature and bin number
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING ORGANISM: {organism.upper()}")
    print(f"{'='*60}")
    
    # File paths
    input_file = os.path.join(ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv")
    input_mi_info_file = os.path.join(ProjectPaths.get_cleaned_data_dir(), "mi_bin_info.json")
    
    # Load the CSV file
    print(f"Loading data from {input_file}...")
    data = pd.read_csv(input_file, index_col=0)
    print(f"Data loaded with shape {data.shape}")
    
    # Loading the MI info file
    mi_dict_file = data_handler.json_file_loader(input_mi_info_file)
    mi_dict = mi_dict_file[organism]
    
    # Global min/max values
    global_min = mi_dict["global_min"]
    global_max = mi_dict["global_max"]
    
    # Print which min/max values are being used
    if use_global_minmax:
        print(f"Using global min ({global_min:.4f}) and max ({global_max:.4f}) for all features")
    else:
        print("Using individual min/max values for each feature")
    
    # Create dictionary to store entropy values for each feature and bin number
    entropy_values = {}
    
    # Track start time for performance reporting
    start_time = time.time()
    
    # Process each feature
    for feature in tqdm(data.columns, desc=f"Processing {organism} features"):
        feature_values = data[feature].values
        
        # Calculate individual min/max for the current feature
        individual_min = np.min(feature_values)
        individual_max = np.max(feature_values)
        
        # Select which min/max to use based on the boolean option
        min_value = global_min if use_global_minmax else individual_min
        max_value = global_max if use_global_minmax else individual_max
        
        # Initialize dictionary for this feature
        entropy_values[feature] = {}
        
        # Process each bin number
        for bin_num in bin_num_list:
            # Create bin edges using the selected min/max values
            bin_edges = np.linspace(min_value, max_value, bin_num + 1)
            
            # Calculate the entropy (MI with itself)
            entropy = entropy_mutualinfo.mi_betwn_uandy(feature_values, feature_values, bin_edges)
            
            # Store the entropy value
            entropy_values[feature][bin_num] = entropy
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Processed {len(data.columns)} features in {int(minutes)} minutes and {seconds:.2f} seconds")
    
    # Create output directory
    minmax_type = "global" if use_global_minmax else "individual"
    output_dir_name = "entropy_values"
    output_dir = os.path.join(ProjectPaths.get_output_files_dir(), output_dir_name, organism)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the entropy values to a JSON file
    output_file = os.path.join(output_dir, f"all_ptms_bin_based_entropy_values_{minmax_type}_minmax.json")
    
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(entropy_values, f, indent=4)
    
    # Generate a summary of entropy values at key bin counts
    summary_bins = [10, 50, 100, 200, 300, 400, 500]
    summary_bins = [b for b in summary_bins if b in bin_num_list]
    
    print("\nEntropy Summary for Selected Bin Numbers:")
    print("Feature".ljust(15), end="")
    for bin_num in summary_bins:
        print(f"{bin_num:>8}", end="")
    print()
    
    for feature in data.columns[:10]:  # Show first 10 features
        print(feature.ljust(15), end="")
        for bin_num in summary_bins:
            if bin_num in entropy_values[feature]:
                print(f"{entropy_values[feature][bin_num]:>8.3f}", end="")
            else:
                print(f"{'N/A':>8}", end="")
        print()
    
    if len(data.columns) > 10:
        print("... plus", len(data.columns) - 10, "more features")
    
    # Also save a CSV summary for easier analysis
    summary_df = pd.DataFrame(index=data.columns)
    for bin_num in summary_bins:
        summary_df[f"bin_{bin_num}"] = [entropy_values[feature][bin_num] for feature in data.columns]
    
    summary_file = os.path.join(output_dir, f"bin_entropy_summary_{minmax_type}_minmax.csv")
    summary_df.to_csv(summary_file)
    print(f"Summary CSV saved to {summary_file}")
    
    return entropy_values

def main():
    """Main function to calculate entropy values for all histone PTMs in both organisms."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate entropy values for histone PTMs")
    parser.add_argument("-o", "--organisms", nargs="+", default=["human", "yeast"],
                        help="Organisms to process (default: both human and yeast)")
    parser.add_argument("-g", "--global_minmax", action="store_true", 
                        help="Use global min/max values for all features")
    parser.add_argument("-b", "--start_bin", type=int, default=10,
                        help="Starting bin number (default: 10)")
    parser.add_argument("-e", "--end_bin", type=int, default=500,
                        help="Ending bin number (default: 500)")
    parser.add_argument("-s", "--step", type=int, default=10,
                        help="Step size for bin numbers (default: 10)")
    args = parser.parse_args()
    
    # Use command line arguments
    organisms = args.organisms
    use_global_minmax = args.global_minmax
    
    # Configure bin number list based on command line options
    start_bin = args.start_bin
    end_bin = args.end_bin
    step = args.step
    
    bin_num_list = list(range(start_bin, end_bin + 1, step))
    
    # Program start information
    print(f"\n{'='*80}")
    print("HISTONE PTM ENTROPY CALCULATION")
    print(f"{'='*80}")
    print(f"Organisms to process: {', '.join(organisms)}")
    print(f"Using global min/max: {use_global_minmax}")
    print(f"Testing {len(bin_num_list)} bin configurations from {start_bin} to {end_bin} with step {step}")
    
    # Track overall start time
    overall_start_time = time.time()
    
    # Store results for all organisms
    all_results = {}
    
    # Process each organism
    for i, organism in enumerate(organisms, 1):
        print(f"\n[{i}/{len(organisms)}] Processing organism: {organism}")
        all_results[organism] = process_organism(organism, use_global_minmax, bin_num_list)
    
    # Calculate overall elapsed time
    overall_elapsed_time = time.time() - overall_start_time
    minutes, seconds = divmod(overall_elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    
    # Final summary
    print(f"\n{'='*80}")
    print("ENTROPY CALCULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(organisms)} organisms: {', '.join(organisms)}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save consolidated results if multiple organisms were processed
    if len(organisms) > 1:
        minmax_type = "global" if use_global_minmax else "individual"
        output_dir = os.path.join(ProjectPaths.get_output_files_dir(), "entropy_values")
        os.makedirs(output_dir, exist_ok=True)
        consolidated_file = os.path.join(output_dir, f"consolidated_entropy_values_{minmax_type}_minmax.json")
        
        print(f"Saving consolidated results to {consolidated_file}")
        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()






