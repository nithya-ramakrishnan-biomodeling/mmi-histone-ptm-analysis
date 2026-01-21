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
from sklearn.metrics import r2_score
import notebook__init__b
from ptms_utils.model.histone_mod_predictor import Histone_Regressor
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths

# Define the argument parser
parser = argparse.ArgumentParser(
    description="Calculate individual and cumulative R² scores for histone modifications"
)

parser.add_argument(
    "-o",
    "--organism",
    type=str,
    required=True,
    help="The organism to analyze",
)
parser.add_argument(
    "-t",
    "--target",
    type=str,
    required=True,
    help="The target histone modification to predict",
)
parser.add_argument(
    "--cutoff",
    type=float,
    default=-0.6,
    help="Cutoff value used in feature/target split (default: -0.6)",
)
parser.add_argument(
    "--cpu_limit",
    type=int,
    default=2,
    help="Number of CPU cores to use (default: 2)",
)
parser.add_argument(
    "--memory_limit",
    type=int,
    default=4000,
    help="Memory limit in MB for the process",
)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Force clean start and remove any previous results",
)

# Parse the arguments
args = parser.parse_args()

# Access the variable values
organism = args.organism
target_histone = args.target
cutoff = args.cutoff
cpu_limit = args.cpu_limit
memory_limit = args.memory_limit
clean_start = args.clean

# Set up directories
dir_name = r"histone_r2_prediction"
r2_dir_name = r"cumulative_r2"
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), dir_name, organism, r2_dir_name
)

# Create output directory if it doesn't exist
dir_handler.dir_maker(output_dir_full_path)


# Standard function to use in all files
def json_file_saver(save_dict: dict, absolute_file_name: str):
    """Save dictionary to JSON file with pretty formatting

    Parameters
    ----------
    data_dict : dict
        Dictionary to save
    file_path : str
        Path to output file
    """
    with open(absolute_file_name, "w") as json_file:
        json.dump(save_dict, json_file, indent=4)


def json_file_loader(file_path: str):
    """Load JSON file and return its contents."""
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def check_memory_usage():
    """Monitor memory usage and return usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


def calculate_correlations(
    histone_mod_df: pd.DataFrame, target_histone: str, feature_histones: list
):
    """
    Calculate correlations between the target histone and specified feature histones
    """
    # Calculate correlations for specified features only
    correlations = {}
    for feature in feature_histones:
        corr = histone_mod_df[feature].corr(histone_mod_df[target_histone])
        correlations[feature] = abs(corr)  # Use absolute correlation values

    # Sort by correlation (highest to lowest)
    sorted_correlations = dict(
        sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    )

    return sorted_correlations


def calculate_augmented_r2(
    histone_mod_df: pd.DataFrame,
    target_histone: str,
    base_features: list,
    extra_feature: str,
):
    """
    Calculate R² score for the base feature set PLUS one extra feature predicting the target histone
    """
    try:
        # Combine base features with the extra feature
        feature_list = base_features + [extra_feature]

        # Extract feature and target data
        X = histone_mod_df[feature_list].copy()
        y = histone_mod_df[target_histone].values

        # Initialize and run prediction model
        ml_model = Histone_Regressor(feature=X, target=y)

        # Get actual and predicted values
        y_test, y_pred = ml_model.xgb_regressor_pred()

        # Calculate R² score
        r2 = r2_score(y_test, y_pred)

        # Clean up to free memory
        del X, ml_model, y_test, y_pred
        gc.collect()

        return extra_feature, r2
    except Exception as e:
        return extra_feature, f"Error: {str(e)}"


def calculate_cumulative_r2(
    histone_mod_df: pd.DataFrame,
    target_histone: str,
    base_features: list,
    added_features: list,
):
    """
    Calculate R² score using base features plus a cumulative list of added features to predict the target histone
    """
    try:
        # Combine base features with cumulative added features
        feature_list = base_features + added_features

        # Extract feature and target data
        X = histone_mod_df[feature_list].copy()
        y = histone_mod_df[target_histone].values

        # Initialize and run prediction model
        ml_model = Histone_Regressor(feature=X, target=y)

        # Get actual and predicted values
        y_test, y_pred = ml_model.xgb_regressor_pred()

        # Calculate R² score
        r2 = r2_score(y_test, y_pred)

        # Clean up to free memory
        del X, ml_model, y_test, y_pred
        gc.collect()

        return len(added_features), r2
    except Exception as e:
        return len(added_features), f"Error: {str(e)}"


def run_augmented_r2_parallel(
    histone_mod_df: pd.DataFrame,
    target_histone: str,
    base_features: list,
    extra_features: list,
    n_jobs: int = 2,
):
    """Calculate augmented R² scores in parallel"""
    try:
        # Limit n_jobs to a safe value
        n_jobs = min(n_jobs, len(extra_features), cpu_limit)

        results = Parallel(n_jobs=n_jobs, max_nbytes="50M")(
            delayed(calculate_augmented_r2)(
                histone_mod_df, target_histone, base_features, feature
            )
            for feature in extra_features
        )
        return results
    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        results = []
        for feature in tqdm(extra_features, desc="Calculating augmented R²"):
            result = calculate_augmented_r2(
                histone_mod_df, target_histone, base_features, feature
            )
            results.append(result)
            gc.collect()
        return results


if __name__ == "__main__":
    print(f"Starting R² score calculation for target: {target_histone}")
    print(f"Using up to {cpu_limit} CPU cores, memory limit: {memory_limit} MB")
    print(f"System has {os.cpu_count()} logical processors available")
    print(f"Current memory usage: {check_memory_usage():.2f} MB")

    # Clean previous results if requested
    if clean_start:
        individual_output = os.path.join(
            output_dir_full_path, f"{organism}_{target_histone}_individual_r2.json"
        )
        cumulative_output = os.path.join(
            output_dir_full_path, f"{organism}_{target_histone}_cumulative_r2.json"
        )
        r2_values_output = os.path.join(
            output_dir_full_path, f"{organism}_{target_histone}_r2_values.json"
        )

        for file_path in [individual_output, cumulative_output, r2_values_output]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed previous results: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    # Define file paths
    histone_file = f"{organism}_zero_mean.csv"
    histone_mod_file_path = os.path.join(cleaned_dir, organism, histone_file)

    # Path to feature/target split JSON file
    feature_target_file = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "ptm_prediction",
        organism,
        "pca_r2",
        f"{organism}_pca_cutoff_{cutoff}_feature_target.json",
    )

    # Start timing
    start_time = time.time()

    # Load feature/target split JSON file
    print(f"Loading feature/target split from {feature_target_file}...")
    try:
        feature_target_data = json_file_loader(feature_target_file)
        base_feature_list = feature_target_data.get("feature", [])
        target_list = feature_target_data.get("target", [])

        print(
            f"Found {len(base_feature_list)} base feature histones and {len(target_list)} target histones"
        )

        # Verify our target histone is in the target list
        if target_histone not in target_list:
            print(
                f"Warning: Target histone '{target_histone}' is not in the target list from feature/target split"
            )
            print(
                "This means your target is likely a 'feature' in the original methodology"
            )
    except Exception as e:
        print(f"Error loading feature/target split: {e}")
        print("Continuing without feature filtering...")
        base_feature_list = []
        target_list = []

    # Load data with optimizations
    print("Loading histone modification data...")
    try:
        # Try to optimize data types for memory efficiency
        histone_mod_df = data_handler.csv_loader(histone_mod_file_path)

        # Convert to float32 to save memory (half the memory usage of float64)
        for col in histone_mod_df.columns:
            if histone_mod_df[col].dtype == "float64":
                histone_mod_df[col] = histone_mod_df[col].astype("float32")

        # Verify that the target histone exists in the data
        if target_histone not in histone_mod_df.columns:
            print(f"Error: Target histone '{target_histone}' not found in the data")
            print(f"Available histones: {', '.join(histone_mod_df.columns)}")
            exit(1)

        print(f"Loaded data with {histone_mod_df.shape[1]} histone modifications")
        print(f"DataFrame shape: {histone_mod_df.shape}")
        print(
            f"Memory usage: {histone_mod_df.memory_usage().sum() / 1024 / 1024:.2f} MB"
        )
        print(f"Current process memory: {check_memory_usage():.2f} MB")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Verify all base features exist in the dataset
    if base_feature_list:
        missing_features = [
            f for f in base_feature_list if f not in histone_mod_df.columns
        ]
        if missing_features:
            print(
                f"Warning: {len(missing_features)} base features not found in dataset: {', '.join(missing_features)}"
            )

        # Use only available base features
        base_features = [f for f in base_feature_list if f in histone_mod_df.columns]
        print(f"Using {len(base_features)} available base features")
    else:
        base_features = []
        print("No base features provided. Using empty base set.")

    # Get target list histones that are available in the dataset (excluding the target itself)
    available_target_histones = [
        h for h in target_list if h in histone_mod_df.columns and h != target_histone
    ]
    print(
        f"Found {len(available_target_histones)} available target histones (excluding the target itself)"
    )

    # Calculate correlations with the target (for all target histones)
    print(f"Calculating correlations with {target_histone}...")
    correlations = calculate_correlations(
        histone_mod_df, target_histone, available_target_histones
    )

    # Sort target histones by correlation with target
    sorted_target_histones = list(correlations.keys())

    print(f"Top 5 correlated histones:")
    for i, (histone, corr) in enumerate(list(correlations.items())[:5]):
        print(f"  {i+1}. {histone}: correlation = {corr:.4f}")

    # --- CALCULATE BASE MODEL R² SCORE ---
    print("\nCalculating R² score with just base features...")

    # Check if there are any base features
    if base_features:
        try:
            # Calculate R² with just base features
            X_base = histone_mod_df[base_features].copy()
            y_base = histone_mod_df[target_histone].values

            # Initialize and run prediction model
            ml_model_base = Histone_Regressor(feature=X_base, target=y_base)

            # Get actual and predicted values
            y_test_base, y_pred_base = ml_model_base.xgb_regressor_pred()

            # Calculate R² score
            base_r2 = r2_score(y_test_base, y_pred_base)

            print(f"Base feature set R² score: {base_r2:.4f}")

            # Clean up
            del X_base, y_base, ml_model_base, y_test_base, y_pred_base
            gc.collect()
        except Exception as e:
            print(f"Error calculating base feature R² score: {e}")
            base_r2 = 0
    else:
        print("No base features - skipping base R² calculation")
        base_r2 = 0

    # --- CALCULATE INDIVIDUAL (BASE + 1 FEATURE) R² SCORES ---
    print(
        f"\nCalculating individual R² scores (base features + 1 feature) for {len(sorted_target_histones)} features..."
    )
    individual_r2_results = run_augmented_r2_parallel(
        histone_mod_df,
        target_histone,
        base_features,
        sorted_target_histones,
        n_jobs=cpu_limit,
    )

    # Process individual results
    individual_r2_dict = {}
    valid_results = 0
    for feature, r2 in individual_r2_results:
        individual_r2_dict[feature] = r2
        if not isinstance(r2, str):  # It's a valid numerical value
            valid_results += 1

    print(
        f"Individual R² calculation complete: {valid_results}/{len(sorted_target_histones)} valid results"
    )

    # Sort individual R² by value
    sorted_individual_r2 = individual_r2_dict.copy()
    #  dict(
    #     sorted(
    #         {k: v for k, v in individual_r2_dict.items() if not isinstance(v, str)}.items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )
    # )

    # Save individual R² results
    individual_output_path = os.path.join(
        output_dir_full_path, f"{organism}_{target_histone}_individual_r2.json"
    )
    json_file_saver(sorted_individual_r2, individual_output_path)
    print(f"Individual R² scores saved to {individual_output_path}")

    # Calculate cumulative R² scores
    print(f"\nCalculating cumulative R² scores...")
    cumulative_r2_dict = {}
    simplified_r2_dict = {}

    # First, add an entry for the base feature set
    if base_features:
        cumulative_r2_dict["0"] = {
            "num_features": 0,  # 0 added features (just base)
            "r2_score": base_r2,
            "features_included": base_features.copy(),
        }
        simplified_r2_dict["0"] = base_r2

    # Calculate cumulative R² by adding one feature at a time from the sorted target histones
    added_features = []
    for i in tqdm(
        range(1, len(sorted_target_histones) + 1), desc="Calculating cumulative R²"
    ):
        # Add the next most correlated histone
        added_features.append(sorted_target_histones[i - 1])

        # Calculate R² with base features + added features
        _, r2 = calculate_cumulative_r2(
            histone_mod_df, target_histone, base_features, added_features
        )

        if isinstance(r2, str):  # Error occurred
            print(f"Error calculating cumulative R² with {i} added features: {r2}")
            continue

        # Store results
        cumulative_r2_dict[str(i)] = {
            "num_features": i,  # Number of added features
            "r2_score": r2,
            "features_included": base_features + added_features.copy(),
        }

        # Also save to simplified format for plotting
        simplified_r2_dict[str(i)] = r2

        # Print milestone results
        if i == 1 or i == len(sorted_target_histones) or i % 10 == 0:
            print(f"  {i} added features: R² = {r2:.4f}")

        # Force garbage collection
        gc.collect()

    # Save cumulative R² results (detailed version with feature lists)
    cumulative_output_path = os.path.join(
        output_dir_full_path, f"{organism}_{target_histone}_cumulative_r2.json"
    )
    json_file_saver(
        save_dict=cumulative_r2_dict, absolute_file_name=cumulative_output_path
    )
    print(f"Cumulative R² scores saved to {cumulative_output_path}")

    # Save simplified R² results (just numbers for easy plotting)
    r2_values_output_path = os.path.join(
        output_dir_full_path, f"{organism}_{target_histone}_r2_values.json"
    )
    json_file_saver(
        save_dict=simplified_r2_dict, absolute_file_name=r2_values_output_path
    )
    print(f"Simplified R² values saved to {r2_values_output_path}")

    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(
        f"Runtime: {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds"
    )
