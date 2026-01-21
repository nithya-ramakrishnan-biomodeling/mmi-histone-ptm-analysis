# run_global_random_feature_prediction_config.py
"""
Configuration file to run global random feature-based R2 score predictions.
Loops through different organisms, MMI cutoff values, and copy numbers.
"""

import subprocess
import sys
import os


# Configuration variables - MODIFY THESE
CONFIG = {
    "human": {
        "num_copies": 3,  # Number of global random feature copies to process
        "mmi_cutoff_values": ["-0.6", "-0.7", "-0.9"],  # MMI cutoff values
    },
    "yeast": {
        "num_copies": 3,  # Number of global random feature copies to process
        "mmi_cutoff_values": ["-0.41", "-0.43", "-0.44"],  # MMI cutoff values
    },
}

# Global settings
GLOBAL_CONFIG = {
    "random_seed": 42,  # Random seed for reproducibility
    "parallel": False,  # Run predictions in parallel (True/False)
}


def run_global_random_feature_prediction(
    organism, mmi_cutoff_value, copy_num, parallel, random_seed
):
    """
    Run the global random feature R2 score prediction for a given organism and cutoff.

    Parameters
    ----------
    organism : str
        Target organism (e.g., 'human', 'yeast')
    mmi_cutoff_value : str
        MMI cutoff value
    copy_num : int
        Copy number of global random features
    parallel : bool
        Whether to run in parallel mode
    random_seed : int
        Random seed for reproducibility
    """

    prediction_main_file = os.path.join(
        os.path.dirname(__file__), "target_ptm_prediction_from_global_feature.py"
    )

    cmd = [
        sys.executable,
        prediction_main_file,
        "-o",
        organism,
        "-mmi",
        mmi_cutoff_value,
        "-c",
        str(copy_num),
        "-rs",
        str(random_seed),
    ]

    if parallel:
        cmd.append("-p")

    print(f"\nCommand: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main():
    """Main function to run global random feature predictions for all configurations."""

    random_seed = GLOBAL_CONFIG["random_seed"]
    parallel = GLOBAL_CONFIG["parallel"]

    print(f"\n{'#'*80}")
    print(f"GLOBAL RANDOM FEATURE - R2 SCORE PREDICTION CONFIGURATION")
    print(f"{'#'*80}")
    print(f"Global Random Seed: {random_seed}")
    print(f"Parallel Mode: {parallel}\n")

    total_jobs = 0

    # Process each organism
    for organism, config in CONFIG.items():
        num_copies = config["num_copies"]
        mmi_cutoff_values = config["mmi_cutoff_values"]

        print(f"\n{'='*80}")
        print(f"[ORGANISM: {organism.upper()}]")
        print(f"{'='*80}")
        print(f"  MMI Cutoff Values: {mmi_cutoff_values}")
        print(f"  Number of Copies per Cutoff: {num_copies}")

        # Process each MMI cutoff value
        for mmi_cutoff in mmi_cutoff_values:
            print(f"\n  {'-'*76}")
            print(f"  MMI Cutoff: {mmi_cutoff}")
            print(f"  {'-'*76}")

            # Process each copy
            for copy_num in range(1, num_copies + 1):
                print(f"    Copy {copy_num}/{num_copies}:")
                run_global_random_feature_prediction(
                    organism=organism,
                    mmi_cutoff_value=mmi_cutoff,
                    copy_num=copy_num,
                    parallel=parallel,
                    random_seed=random_seed,
                )
                total_jobs += 1

    print(f"\n{'#'*80}")
    print(f"GLOBAL RANDOM FEATURE PREDICTION COMPLETED")
    print(f"{'#'*80}")
    print(f"Total Jobs Processed: {total_jobs}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
