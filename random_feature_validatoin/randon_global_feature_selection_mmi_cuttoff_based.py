# global_random_feature_selection_main.py
"""
Script to generate global random features based on MMI cutoff values.
Instead of selecting features individually for each target histone,
this script generates one global set of random features per MMI cutoff value.
Uses random_n_feature_selector for feature selection.
"""
import os
import argparse
import json
import numpy as np
from pathlib import Path
import notebook__init__b
import pandas as pd
from path_utils.path_handler import ProjectPaths
from datahandler import dir_handler, data_handler
from random_utils.random_feature_selector import random_n_feature_selector


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Global Random Features for MMI Cutoff Values"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Target organism (e.g., 'human', 'yeast')",
    )
    parser.add_argument(
        "-mmi",
        "--mmi_cutoff_values",
        type=str,
        nargs="+",
        required=True,
        help="MMI cutoff values for feature selection (can specify multiple)",
    )
    parser.add_argument(
        "-c",
        "--num_copies",
        type=int,
        default=10,
        help="Number of global random feature copies to generate (default: 10)",
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
    """Check if the file path exists and raise error if it doesn't."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")


def get_all_features_from_data(organism: str) -> set:
    """
    Load all features from the zero-mean data file.

    Parameters
    ----------
    organism : str
        The organism name (e.g., 'human', 'yeast')

    Returns
    -------
    set
        Set of all feature names from the dataset
    """
    data_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )

    check_file_path(data_file_path)
    data = pd.read_csv(data_file_path, index_col=0)
    return set(data.columns)


def main():
    """Main function to generate global random features for each MMI cutoff."""
    args = parse_arguments()

    organism = args.organism
    mmi_cutoff_values = args.mmi_cutoff_values
    num_copies = args.num_copies
    random_seed = args.random_seed

    print(f"\n{'='*80}")
    print("GLOBAL RANDOM FEATURE GENERATION")
    print(f"{'='*80}")
    print(f"Organism: {organism}")
    print(f"MMI cutoff values: {mmi_cutoff_values}")
    print(f"Number of copies per cutoff: {num_copies}")
    print(f"Random seed: {random_seed}")

    # Get all available features from the dataset
    print(f"\nLoading all features from dataset...")
    all_features = get_all_features_from_data(organism)
    total_available_features = len(all_features)
    print(f"Total available features: {total_available_features}")

    # Process each MMI cutoff value
    results_summary = {}

    for mmi_cutoff in mmi_cutoff_values:
        print(f"\n{'-'*80}")
        print(f"Processing MMI cutoff: {mmi_cutoff}")
        print(f"{'-'*80}")

        # Define file path for this cutoff value
        input_file_path = os.path.join(
            ProjectPaths.get_output_files_dir(),
            "ptm_prediction",
            organism,
            "mmi_r2",
            f"{organism}_mmi_cutoff_{mmi_cutoff}_le_feature_target.json",
        )

        # Check if input file exists
        try:
            check_file_path(input_file_path)
            input_file = data_handler.json_file_loader(input_file_path)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

        # Extract feature information from the MMI cutoff file
        feature_list = input_file["feature"]
        target_list = input_file["target"]
        num_features = len(feature_list)

        print(f"Number of features for this cutoff: {num_features}")
        print(f"Number of targets: {len(target_list)}")

        # Create output directory for this cutoff
        output_dir = os.path.join(
            ProjectPaths.get_output_files_dir(),
            "global_random_feature_selection_results",
            organism,
            f"mmi_cutoff_{mmi_cutoff}_global",
        )
        os.makedirs(output_dir, exist_ok=True)

        # Generate multiple copies of global random features
        copy_results = []

        for copy_num in range(1, num_copies + 1):
            # Use different random seed for each copy
            copy_random_seed = random_seed + (int(float(mmi_cutoff) * 1000)) + copy_num

            # Use random_n_feature_selector to select n random features from all features
            selected_features = random_n_feature_selector(
                all_features, num_features, copy_random_seed
            )

            # Save selected features to text file
            output_file = os.path.join(
                output_dir, f"global_random_features_copy_{copy_num}.txt"
            )
            with open(output_file, "w") as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")

            # Also save as JSON for easier parsing
            json_output_file = os.path.join(
                output_dir, f"global_random_features_copy_{copy_num}.json"
            )
            with open(json_output_file, "w") as f:
                json.dump(
                    {
                        "copy_number": copy_num,
                        "mmi_cutoff": mmi_cutoff,
                        "organism": organism,
                        "num_features_selected": len(selected_features),
                        "random_seed": copy_random_seed,
                        "features": sorted(selected_features),
                    },
                    f,
                    indent=4,
                )

            copy_results.append(
                {
                    "copy_number": copy_num,
                    "num_features": len(selected_features),
                    "random_seed": copy_random_seed,
                    "output_file": output_file,
                    "json_output_file": json_output_file,
                }
            )

            print(
                f"  ✓ Copy {copy_num}/{num_copies}: Generated {num_features} random features (seed: {copy_random_seed})"
            )

        # Save summary for this cutoff
        summary_data = {
            "organism": organism,
            "mmi_cutoff": mmi_cutoff,
            "total_available_features": total_available_features,
            "features_selected_per_copy": num_features,
            "num_copies": num_copies,
            "output_directory": output_dir,
            "copies": copy_results,
        }

        summary_file = os.path.join(output_dir, "global_random_features_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)

        print(f"  ✓ Summary saved to: {summary_file}")

        results_summary[mmi_cutoff] = summary_data

    # Save consolidated summary for all cutoffs
    consolidated_summary_file = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "global_random_feature_selection_results",
        organism,
        f"{organism}_global_random_features_consolidated_summary.json",
    )
    os.makedirs(os.path.dirname(consolidated_summary_file), exist_ok=True)

    with open(consolidated_summary_file, "w") as f:
        json.dump(results_summary, f, indent=4)

    print(f"\n{'='*80}")
    print("GLOBAL RANDOM FEATURE GENERATION COMPLETED")
    print(f"{'='*80}")
    print(f"Organism: {organism}")
    print(f"MMI cutoff values processed: {len(mmi_cutoff_values)}")
    print(f"Copies per cutoff: {num_copies}")
    print(f"Consolidated summary: {consolidated_summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
