"""
Configuration file to run global random feature selection with multiple organisms and MMI cutoff values.
This config file sets up parameters for generating global random features based on MMI cutoffs.
Loops through different organisms and processes multiple MMI cutoff values in a single run.
"""

import subprocess
import sys
import os


# Configuration variables - MODIFY THESE
CONFIG = {
    "human": {
        "num_copies": 3,  # Number of global random feature copies to generate per cutoff
        "mmi_cutoff_values": ["-0.6", "-0.7", "-0.9"],  # MMI cutoff values to process
    },
    "yeast": {
        "num_copies": 3,  # Number of global random feature copies to generate per cutoff
        "mmi_cutoff_values": [
            "-0.41",
            "-0.43",
            "-0.44",
        ],  # MMI cutoff values to process
    },
}

# Global settings
GLOBAL_CONFIG = {
    "random_seed": 42,  # Random seed for reproducibility
}


def run_global_random_feature_selection(
    organism, mmi_cutoff_values, num_copies, random_seed
):
    """
    Run the global random feature selection script for a given organism and cutoff values.

    Parameters
    ----------
    organism : str
        Target organism (e.g., 'human', 'yeast')
    mmi_cutoff_values : list
        List of MMI cutoff values to process
    num_copies : int
        Number of random feature copies per cutoff
    random_seed : int
        Random seed for reproducibility
    """

    global_random_feature_main_file = os.path.join(
        os.path.dirname(__file__),
        "randon_global_feature_selection_mmi_cuttoff_based.py",
    )

    # Build command with all cutoff values
    cmd = (
        [sys.executable, global_random_feature_main_file, "-o", organism, "-mmi"]
        + mmi_cutoff_values
        + [
            "-c",
            str(num_copies),
            "-rs",
            str(random_seed),
        ]
    )

    print(f"\n{'='*80}")
    print(f"Running Global Random Feature Selection for {organism}")
    print(f"{'='*80}")
    print(f"MMI Cutoff Values: {mmi_cutoff_values}")
    print(f"Number of Copies: {num_copies}")
    print(f"Random Seed: {random_seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    subprocess.run(cmd, check=False)


def main():
    """Main function to run global random feature selection for all configured organisms."""

    random_seed = GLOBAL_CONFIG["random_seed"]

    print(f"\n{'#'*80}")
    print(f"GLOBAL RANDOM FEATURE SELECTION CONFIGURATION")
    print(f"{'#'*80}")
    print(f"Global Random Seed: {random_seed}\n")

    # Process each organism
    for organism, config in CONFIG.items():
        num_copies = config["num_copies"]
        mmi_cutoff_values = config["mmi_cutoff_values"]

        print(f"\n[ORGANISM: {organism.upper()}]")
        print(f"  Number of Copies: {num_copies}")
        print(f"  MMI Cutoff Values: {mmi_cutoff_values}")

        run_global_random_feature_selection(
            organism=organism,
            mmi_cutoff_values=mmi_cutoff_values,
            num_copies=num_copies,
            random_seed=random_seed,
        )

    print(f"\n{'#'*80}")
    print(f"GLOBAL RANDOM FEATURE SELECTION COMPLETED FOR ALL ORGANISMS")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
