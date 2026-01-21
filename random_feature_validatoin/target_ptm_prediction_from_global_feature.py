# target_ptm_prediction_from_global_random_selection.py
"""
This file predicts target histone PTM intensities using global random features.

Idea:
- Load the MMI Histone PTM data for a given organism
- Extract feature histone PTMs and target histone PTM
- Using global random features generated per MMI cutoff:
  - Get the global random features from the global feature set file
  - Predict all target histone PTMs using the same global random features
  - Calculate prediction performance using R2 score for each target
  - Save the prediction performance into a JSON file

Note: Key consideration:
- Input as command line argument
- Run in parallel mode using joblib if resources available
- Memory and CPU optimization for ML prediction
"""

import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import json
import notebook__init__b
from datahandler import data_handler, dir_handler
from utils import histone_mod_predictor
from path_utils.path_handler import ProjectPaths
import psutil


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict target histone PTM from global random feature selection"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="The organism to analyze (e.g., 'human', 'yeast')",
    )

    parser.add_argument(
        "-mmi",
        "--mmi_cutoff_value",
        type=str,
        required=True,
        help="MMI cutoff value to get the global random features for target histone prediction",
    )

    parser.add_argument(
        "-c",
        "--copy_num",
        type=int,
        required=True,
        help="Copy number of global random feature set to use for prediction",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Run predictions in parallel (default: False)",
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
    """Check if file path exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def load_global_random_features(organism: str, mmi_cutoff: str, copy_num: int) -> list:
    """
    Load global random features from the global feature set file.

    Parameters
    ----------
    organism : str
        Target organism
    mmi_cutoff : str
        MMI cutoff value
    copy_num : int
        Copy number of global random features

    Returns
    -------
    list
        List of feature names
    """
    global_feature_file = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "global_random_feature_selection_results",
        organism,
        f"mmi_cutoff_{mmi_cutoff}_global",
        f"global_random_features_copy_{copy_num}.txt",
    )

    check_file_path(global_feature_file)

    with open(global_feature_file, "r") as f:
        features = [line.strip() for line in f.readlines()]

    return features


if __name__ == "__main__":
    args = parse_arguments()

    organism = args.organism
    mmi_cutoff_value = args.mmi_cutoff_value
    copy_num = args.copy_num
    parallel_mode = args.parallel
    random_seed = args.random_seed

    print(f"\n{'='*80}")
    print("GLOBAL RANDOM FEATURE - R2 SCORE PREDICTION")
    print(f"{'='*80}")
    print(f"Organism: {organism}")
    print(f"MMI Cutoff: {mmi_cutoff_value}")
    print(f"Copy Number: {copy_num}")
    print(f"Parallel Mode: {parallel_mode}")
    print(f"{'='*80}\n")

    # Define file paths
    mmi_file_path_with_cutoff = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "ptm_prediction",
        organism,
        "mmi_r2",
        f"{organism}_mmi_cutoff_{mmi_cutoff_value}_le_feature_target.json",
    )

    # Load processed histone modification data
    input_data_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )

    # Check if files exist
    check_file_path(input_data_file_path)
    check_file_path(mmi_file_path_with_cutoff)

    # Load MMI data to get target histone names
    mmi_data = data_handler.json_file_loader(mmi_file_path_with_cutoff)
    target_histone_names = mmi_data["target"]
    num_features_in_cutoff = len(mmi_data["feature"])

    print(f"Loading global random features (copy {copy_num})...")
    # Load global random features
    global_random_features = load_global_random_features(
        organism, mmi_cutoff_value, copy_num
    )
    print(f"Number of global random features: {len(global_random_features)}")
    print(f"Features: {global_random_features}\n")

    # Load data ONCE before the loop
    print(f"Loading histone PTM data from {organism}...")
    input_data = pd.read_csv(input_data_file_path)
    print(f"Data shape: {input_data.shape}\n")

    target_r2 = {}

    # Function to predict and save R2 score for a single target histone
    def predict_and_save_r2(target_histone, verbose=False):
        """
        Predict R2 score for a target histone using global random features.

        Parameters
        ----------
        target_histone : str
            Target histone name
        verbose : bool
            Whether to print verbose output

        Returns
        -------
        tuple
            (target_histone, r2_score)
        """
        try:
            # Use global random features for prediction
            random_feature_data = input_data[global_random_features]
            target_data = input_data[[target_histone]]

            r2_regressor = histone_mod_predictor.Histone_Regressor(
                feature=random_feature_data.to_numpy(), target=target_data.to_numpy()
            )

            r2_score = r2_regressor.error_calculator().round(3)
            if verbose:
                print(f"  R2 score for {target_histone}: {r2_score}")
            return target_histone, r2_score
        except Exception as e:
            print(f"  ERROR predicting {target_histone}: {e}")
            return target_histone, None

    # Choose between parallel and sequential processing
    print(f"Processing in {'parallel' if parallel_mode else 'sequential'} mode")
    print(f"Number of targets: {len(target_histone_names)}\n")

    # Calculate optimal number of workers
    n_workers = max(1, psutil.cpu_count(logical=False) - 1)  # Leave 1 core free

    if parallel_mode:
        print(f"Running with {n_workers} workers...\n")
        # Use Parallel and delayed to run the loop in parallel
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(predict_and_save_r2)(target_, verbose=False)
            for target_ in target_histone_names
        )
        # Update the target_r2 dictionary with results
        target_r2 = dict(results)
    else:
        # Process sequentially
        for target_ in target_histone_names:
            target_name, score = predict_and_save_r2(target_, verbose=True)
            target_r2[target_name] = score

    # Filter out None values (failed predictions)
    target_r2 = {k: v for k, v in target_r2.items() if v is not None}

    # Print the final dictionary
    print(f"\n{'='*80}")
    print(f"Final R2 Scores ({len(target_r2)} targets):")
    print(f"{'='*80}")
    for target, score in sorted(target_r2.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        print(f"  {target}: {score}")
    if len(target_r2) > 10:
        print(f"  ... and {len(target_r2) - 10} more")
    print(f"{'='*80}\n")

    # Sort the results dictionary by R2 score in descending order
    sorted_target_r2 = dict(sorted(target_r2.items(), key=lambda x: x[1], reverse=True))

    # Create output directory
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), "global_ptm_prediction", organism
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save results with standardized naming convention
    output_r2_file_path = os.path.join(
        output_dir,
        f"global_random_feature_mmi_cutoff_{mmi_cutoff_value}_r2_copy{copy_num}.json",
    )

    data_handler.json_file_saver(sorted_target_r2, output_r2_file_path)
    print(f"Results saved to {output_r2_file_path}\n")

    # Also save summary statistics
    summary_stats = {
        "organism": organism,
        "mmi_cutoff": mmi_cutoff_value,
        "copy_number": copy_num,
        "num_targets": len(sorted_target_r2),
        "num_features": len(global_random_features),
        "features": global_random_features,
        "mean_r2": round(sum(sorted_target_r2.values()) / len(sorted_target_r2), 3),
        "max_r2": max(sorted_target_r2.values()),
        "min_r2": min(sorted_target_r2.values()),
        "results": sorted_target_r2,
    }

    summary_file_path = os.path.join(
        output_dir,
        f"global_random_feature_mmi_cutoff_{mmi_cutoff_value}_summary_copy{copy_num}.json",
    )

    with open(summary_file_path, "w") as f:
        json.dump(summary_stats, f, indent=4)

    print(f"Summary saved to {summary_file_path}\n")
