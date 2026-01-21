"""
Main script to run target PTM prediction using random feature selection with multiple copies
Using subprocess to execute predictions for each copy
"""

import os
import argparse
import subprocess
import sys
import notebook__init__b
from datahandler import data_handler
from path_utils.path_handler import ProjectPaths


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run target PTM prediction from random feature selection with multiple copies"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="The organism to analyze (e.g., 'human', 'yeast')",
    )

    parser.add_argument(
        "-c",
        "--mmi_cutoff_value",
        type=str,
        required=True,
        help="MMI cutoff value for feature selection",
    )

    parser.add_argument(
        "-nc",
        "--num_copies",
        type=int,
        required=True,
        help="Number of random feature copies to process (default: 3)",
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
    """Check if file path exists, raise error if not"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


if __name__ == "__main__":
    args = parse_arguments()

    organism = args.organism
    mmi_cutoff_value = args.mmi_cutoff_value
    num_copies = args.num_copies
    parallel_mode = args.parallel
    random_seed = args.random_seed

    print("=" * 70)
    print("TARGET PTM PREDICTION FROM RANDOM FEATURE SELECTION")
    print("=" * 70)
    print(f"Organism: {organism}")
    print(f"MMI Cutoff: {mmi_cutoff_value}")
    print(f"Number of Copies: {num_copies}")
    print(f"Mode: {'Parallel' if parallel_mode else 'Sequential'}")
    print("=" * 70)

    # Validate input files
    mmi_file_path = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "ptm_prediction",
        organism,
        "mmi_r2",
        f"{organism}_mmi_cutoff_{mmi_cutoff_value}_le_feature_target.json",
    )

    input_data_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )

    check_file_path(mmi_file_path)
    check_file_path(input_data_file_path)

    print(f"\nStarting predictions for {num_copies} copies...\n")

    # Loop through each copy
    for copy_num in range(1, num_copies + 1):
        print(f"\n{'─' * 70}")
        print(f"Processing Copy {copy_num}/{num_copies}")
        print(f"{'─' * 70}")

        prediction_file_path = os.path.join(
            ProjectPaths.get_project_root(),
            "random_feature_validatoin",
            "target_ptm_prediction_from_random_selection.py",
        )

        # Build command
        cmd = [
            sys.executable,
            prediction_file_path,
            "-o",
            organism,
            "-c",
            mmi_cutoff_value,
            "-g",
            str(copy_num),
            "-rs",
            str(random_seed),
        ]

        if parallel_mode:
            cmd.append("-p")

        print(f"Running: {' '.join(cmd)}")

        # Run subprocess
        subprocess.run(cmd, check=False)

    print(f"\n{'=' * 70}")
    print("All copies completed!")
    print(f"{'=' * 70}")
