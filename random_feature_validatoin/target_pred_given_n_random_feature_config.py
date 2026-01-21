"""
Configuration file to run target PTM predictions from given n features
Loops through different copy numbers using subprocess
"""

import subprocess
import sys
import os


# Configuration variables - MODIFY THESE
CONFIG = {
    "human": {
        "num_features": 1,
        "copy_numbers": [1, 2, 3],
    },
    "yeast": {
        "num_features": 1,
        "copy_numbers": [1, 2, 3],
    },
}

# Global settings
GLOBAL_CONFIG = {
    "parallel_mode": False,  # True for parallel, False for sequential
    "random_seed": 42,  # Random seed for reproducibility
}


def run_prediction_subprocess(
    organism, num_features, copy_num, parallel_mode, random_seed
):
    """Run prediction for a single copy using subprocess"""

    prediction_file = os.path.join(
        os.path.dirname(__file__), "target_ptm_prediction_from_given_n_features.py"
    )

    cmd = [
        sys.executable,
        prediction_file,
        "-o",
        organism,
        "-nf",
        str(num_features),
        "-c",
        str(copy_num),
        "-rs",
        str(random_seed),
    ]

    if parallel_mode:
        cmd.append("-p")

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print(f"✓ Copy {copy_num} completed successfully")
        if result.stdout:
            print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Copy {copy_num} failed with error:")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


if __name__ == "__main__":
    print("=" * 90)
    print("TARGET PTM PREDICTION FROM GIVEN N FEATURES - MULTIPLE COPY NUMBERS")
    print("=" * 90)
    print(f"Parallel Mode: {GLOBAL_CONFIG['parallel_mode']}")
    print(f"Random Seed: {GLOBAL_CONFIG['random_seed']}")
    print("=" * 90)

    total_runs = sum(len(config["copy_numbers"]) for config in CONFIG.values())
    current_run = 0
    successful_runs = 0
    failed_runs = []

    # Loop through each organism
    for organism in CONFIG.keys():
        organism_config = CONFIG[organism]
        num_features = organism_config["num_features"]
        copy_numbers = organism_config["copy_numbers"]

        print(f"\n{'*' * 90}")
        print(f"* ORGANISM: {organism.upper()}")
        print(f"{'*' * 90}")
        print(f"Number of Features: {num_features}")
        print(f"Copy Numbers: {copy_numbers}")
        print(f"{'*' * 90}\n")

        # Loop through each copy number
        for copy_num in copy_numbers:
            current_run += 1
            print(f"\n{'#' * 90}")
            print(
                f"# Run {current_run}/{total_runs} | Organism: {organism} | Copy: {copy_num}"
            )
            print(f"{'#' * 90}\n")

            success = run_prediction_subprocess(
                organism=organism,
                num_features=num_features,
                copy_num=copy_num,
                parallel_mode=GLOBAL_CONFIG["parallel_mode"],
                random_seed=GLOBAL_CONFIG["random_seed"],
            )

            if success:
                successful_runs += 1
            else:
                failed_runs.append(f"{organism}_copy_{copy_num}")

    # Summary
    print(f"\n{'=' * 90}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 90}")
    print(f"Total Runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print(f"\nFailed runs:")
        for failed in failed_runs:
            print(f"  - {failed}")

    print(f"{'=' * 90}")
