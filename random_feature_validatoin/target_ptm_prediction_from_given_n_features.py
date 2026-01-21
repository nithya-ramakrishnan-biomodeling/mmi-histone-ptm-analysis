"""
Predict target histone PTMs from given n features
"""

import os
import argparse
import psutil
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import notebook__init__b
from datahandler import data_handler
from utils import histone_mod_predictor
from path_utils.path_handler import ProjectPaths


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict target histone PTM from given n features"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="The organism to analyze",
    )
    parser.add_argument(
        "-nf",
        "--num_features",
        type=int,
        default=1,
        help="Number of features used for prediction (default: 1)",
    )
    parser.add_argument(
        "-c",
        "--copy_num",
        type=int,
        default=3,
        help="Copy number of random feature set (default: 1)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Run predictions in parallel mode",
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def check_file_path(file_path: str):
    """Check if file path exists"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def predict_and_save_r2(target_histone, features_list, data_df, verbose=False):
    """
    Predict target histone PTM using given features and calculate R2 score

    Args:
        target_histone: Target histone PTM name
        features_list: List of feature column names from JSON
        data_df: DataFrame loaded from CSV with all data
        verbose: Print progress information

    Returns:
        Tuple of (target_histone, r2_score)
    """
    try:
        # Extract feature data from DataFrame using column names from JSON
        X = data_df[features_list].to_numpy()

        # Extract target data from DataFrame
        y = data_df[[target_histone]].to_numpy()

        if verbose:
            print(f"Predicting {target_histone} using features: {features_list}")

        # Run prediction
        r2_regressor = histone_mod_predictor.Histone_Regressor(feature=X, target=y)

        r2_score = r2_regressor.error_calculator().round(3)

        return target_histone, r2_score

    except KeyError as e:
        print(f"Error: Column not found for {target_histone}: {e}")
        return target_histone, 0.0


if __name__ == "__main__":
    args = parse_arguments()

    organism = args.organism
    num_features = args.num_features
    copy_num = args.copy_num
    parallel_mode = args.parallel
    random_seed = args.random_seed

    print("=" * 70)
    print("TARGET PTM PREDICTION FROM GIVEN N FEATURES")
    print("=" * 70)
    print(f"Organism: {organism}")
    print(f"Number of Features: {num_features}")
    print(f"Copy Number: {copy_num}")
    print(f"Mode: {'Parallel' if parallel_mode else 'Sequential'}")
    print("=" * 70)

    # Define file paths
    random_features_file = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "random_feature_selection_results",
        organism,
        f"{num_features}_features",
        f"random_{num_features}_features_selection_copy_{copy_num}.json",
    )

    data_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )

    # Check if files exist
    check_file_path(random_features_file)
    check_file_path(data_file_path)

    # Load JSON file with random features for each target
    random_features_data = data_handler.json_file_loader(random_features_file)

    # Load CSV data once for all predictions
    data_df = pd.read_csv(data_file_path)

    print(f"\nLoaded data shape: {data_df.shape}")
    print(f"Number of targets to predict: {len(random_features_data)}")

    # Run predictions
    target_r2 = {}

    n_workers = max(1, psutil.cpu_count(logical=False) - 1)

    if parallel_mode:
        print(f"\nRunning predictions in parallel mode ({n_workers} workers)...\n")
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(predict_and_save_r2)(
                target_histone=target,
                features_list=random_features_data[target],
                data_df=data_df,
                verbose=False,
            )
            for target in random_features_data.keys()
        )
        target_r2 = dict(results)
    else:
        print(f"\nRunning predictions in sequential mode...\n")
        for target in random_features_data.keys():
            target_name, score = predict_and_save_r2(
                target_histone=target,
                features_list=random_features_data[target],
                data_df=data_df,
                verbose=True,
            )
            target_r2[target_name] = score

    # Sort results by R2 score in descending order
    sorted_target_r2 = dict(sorted(target_r2.items(), key=lambda x: x[1], reverse=True))

    # Save results
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "ptm_prediction_from_given_features",
        organism,
        f"{num_features}_features",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f"{num_features}_features_copy_{copy_num}_r2.json"
    )

    data_handler.json_file_saver(sorted_target_r2, output_file)

    print(f"\n{'=' * 70}")
    print("RESULTS SAVED")
    print(f"{'=' * 70}")
    print(f"Output file: {output_file}")
    print(f"\nTop 5 R2 Scores:")
    for i, (target, score) in enumerate(list(sorted_target_r2.items())[:5], 1):
        print(f"{i}. {target:30s}: {score:.3f}")
    print(f"{'=' * 70}")
