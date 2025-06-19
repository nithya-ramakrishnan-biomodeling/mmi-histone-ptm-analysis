#!/usr/bin/env python3
"""
Pseudo-Inverse Feature Selection and R² Calculation

This script performs feature selection using the pseudo-inverse method and calculates
R² scores for each histone modification target. It identifies the most influential
features for each target based on pseudo-inverse coefficients.

Usage:
    python r2_score_calcn_pseudo_inverse_based_feature.py -o <organism> -n <num_features>
"""
import numpy as np
import pandas as pd
import os
import json
import argparse
import time
from datahandler import dir_handler
import notebook__init__b
from path_utils.path_handler import ProjectPaths


def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing the data if successful, None otherwise
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def get_features(data, target_column):
    """
    Extract feature columns after removing the target column.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all columns
    target_column : str
        Name of the target column to remove
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing only feature columns
    """
    try:
        features = data.drop(columns=[target_column])
        return features
    except KeyError as e:
        print(f"Error: Column {target_column} not found - {e}")
        return None


def get_pseudo_inverse(y, A):
    """
    Calculate the pseudo-inverse solution to y = Ax.
    
    Parameters
    ----------
    y : numpy.ndarray
        Target values vector
    A : numpy.ndarray
        Feature matrix
        
    Returns
    -------
    numpy.ndarray or None
        Solution vector x
    """
    try:
        # Calculate the pseudo-inverse of A
        A_pseudo_inverse = np.linalg.pinv(A)
        
        # Calculate x using the pseudo-inverse
        x = np.dot(A_pseudo_inverse, y)
        
        return x
    except Exception as e:
        print(f"Error calculating pseudo-inverse: {e}")
        return None


def calculate_r_squared(y_actual, A, x):
    """
    Calculate the R² coefficient of determination.
    
    Parameters
    ----------
    y_actual : numpy.ndarray
        Actual target values
    A : numpy.ndarray
        Feature matrix
    x : numpy.ndarray
        Solution vector from pseudo-inverse
        
    Returns
    -------
    float or None
        R² value between 0 and 1
    """
    try:
        # Calculate predicted values
        y_pred = np.dot(A, x)
        
        # Calculate residuals
        residuals = y_actual - y_pred
        
        # Calculate sum of squared residuals
        ss_res = np.sum(residuals**2)
        
        # Calculate total sum of squares
        y_mean = np.mean(y_actual)
        ss_tot = np.sum((y_actual - y_mean)**2)
        
        # Calculate R-squared
        r_squared = 1 - (ss_res / ss_tot)
        
        return r_squared
    except Exception as e:
        print(f"Error calculating R-squared: {e}")
        return None


def get_top_features(x, feature_names, n):
    """
    Get the top n features based on coefficient magnitude.
    
    Parameters
    ----------
    x : numpy.ndarray
        Coefficient vector from pseudo-inverse
    feature_names : list
        List of feature names
    n : int
        Number of top features to select
        
    Returns
    -------
    tuple
        (indices, feature_names, coefficient_values)
    """
    try:
        # Get indices of top n features by absolute coefficient value
        indices = np.argsort(np.abs(x))[::-1][:n]
        
        # Get the names of the top features
        selected_feature_names = [feature_names[i] for i in indices]
        
        # Get the coefficient values for the top features
        selected_values = x[indices]
        
        return indices, selected_feature_names, selected_values
    except Exception as e:
        print(f"Error selecting top features: {e}")
        return None, None, None


def save_json_data(data_dict, output_file):
    """
    Save data dictionary to a JSON file.
    
    Parameters
    ----------
    data_dict : dict
        Data to be saved to JSON
    output_file : str
        Path to the output JSON file
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Check if the file already exists
        if os.path.exists(output_file):
            # Load existing data from the JSON file
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            # Append the new data to the existing data
            existing_data.update(data_dict)
            data_to_save = existing_data
        else:
            data_to_save = data_dict
        
        # Save the data to the JSON file
        with open(output_file, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error saving to JSON file: {e}")
        return False


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pseudo-inverse feature selection and R² calculation")
    parser.add_argument("-o", "--organism", type=str, default="human",
                        help="Organism to analyze (default: human)")
    parser.add_argument("-n", "--n_features", type=int, default=8,
                        help="Number of top features to select (default: 8)")
    parser.add_argument("-p", "--parallel", action="store_true",
                        help="Enable parallel processing (default: False)")
    args = parser.parse_args()
    
    organism = args.organism
    n_features = args.n_features
    parallel_mode = args.parallel
    
    print(f"Analyzing organism: {organism}")
    print(f"Selecting top {n_features} features for each target")
    print(f"Parallel mode: {'Enabled' if parallel_mode else 'Disabled'}")
    
    # Set up file paths
    input_file = os.path.join(ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv")
    
    output_dir_name = "pseudo_inverse"
    output_file_dir = os.path.join(ProjectPaths.get_output_files_dir(), output_dir_name, organism)
    output_file_name = os.path.join(output_file_dir, f"{organism}_pseudo_inverse_with_r2.json")
    r2_output_file = os.path.join(output_file_dir, f"{organism}_histone_r2_values.json")
    
    # Create output directory if it doesn't exist
    dir_handler.dir_maker(output_file_dir)
    
    # Load data
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, index_col=0)
    
    # Dictionaries to store results
    full_results = {}
    r2_results = {}
    
    # Start timing
    start_time = time.time()
    
    if parallel_mode:
        from joblib import Parallel, delayed
        
        def process_target(target, df):
            # The existing target processing logic
            print(f"Processing target: {target}")
            features = get_features(df, target)
            if features is None:
                return target, None, None
                
            target_values = df[target].values
            A = features.values
            feature_names = features.columns.tolist()
            
            x = get_pseudo_inverse(target_values, A)
            if x is None:
                return target, None, None
                
            r_squared = calculate_r_squared(target_values, A, x)
            if r_squared is None:
                return target, None, None
            
            print(f"  R² score: {r_squared:.4f}")
            
            _, selected_features, _ = get_top_features(x, feature_names, n_features)
            if selected_features is None:
                return target, None, None
                
            return target, selected_features, float(r_squared)
        
        # Execute in parallel
        results = Parallel(n_jobs=-1)(delayed(process_target)(target, df) for target in df.columns)
        
        # Process results
        for target, features, r_squared in results:
            if features is not None and r_squared is not None:
                full_results[target] = {
                    "features": features,
                    "r_squared": r_squared
                }
                r2_results[target] = r_squared
    else:
        # Process each target histone
        for i, target in enumerate(df.columns):
            print(f"Processing target {i+1}/{len(df.columns)}: {target}")
            
            # Get features (all columns except the target)
            features = get_features(df, target)
            
            if features is None:
                print(f"Skipping {target} due to error")
                continue
                
            target_values = df[target].values
            A = features.values
            feature_names = features.columns.tolist()
            
            # Calculate pseudo-inverse solution
            x = get_pseudo_inverse(target_values, A)
            
            if x is None:
                print(f"Skipping {target} due to error in pseudo-inverse calculation")
                continue
                
            # Calculate R-squared
            r_squared = calculate_r_squared(target_values, A, x)
            
            if r_squared is None:
                print(f"Skipping {target} due to error in R² calculation")
                continue
                
            print(f"  R² score: {r_squared:.4f}")
            
            # Get top n features
            _, selected_features, _ = get_top_features(x, feature_names, n_features)
            
            if selected_features is None:
                print(f"Skipping {target} due to error in feature selection")
                continue
                
            # Store results
            full_results[target] = {
                "features": selected_features,
                "r_squared": float(r_squared)
            }
            
            r2_results[target] = float(r_squared)
    
    # Save results to JSON files
    print(f"Saving full results to {output_file_name}")
    save_json_data(full_results, output_file_name)
    
    print(f"Saving R² values to {r2_output_file}")
    save_json_data(r2_results, r2_output_file)
    
    # Calculate and print elapsed time
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Processing complete in {int(minutes)} minutes and {seconds:.2f} seconds")
    print(f"Results saved to {output_file_dir}")


if __name__ == "__main__":
    main()














































































