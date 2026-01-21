"""
Configuration file to run random feature selection with multiple organisms and cutoff values
Loops through different organisms and MMI cutoff values
"""

import subprocess
import sys
import os


# Configuration variables - MODIFY THESE
CONFIG = {
    "human": {
        "num_copies": 3,
        "mmi_cutoff_values": ["-0.6", "-0.7", "-0.9"],
    },
    "yeast": {
        "num_copies": 3,
        "mmi_cutoff_values": ["-0.41", "-0.43", "-0.44"],
    },
}

# Global settings
GLOBAL_CONFIG = {
    "random_seed": 42,  # Random seed for reproducibility
}


def run_random_feature_selection(organism, mmi_cutoff_value, num_copies, random_seed):
    """Run the random feature selection script for a given organism and cutoff"""

    random_feature_main_file = os.path.join(
        os.path.dirname(__file__), "random_feature_selection_main.py"
    )

    cmd = [
        sys.executable,
        random_feature_main_file,
        "-o",
        organism,
        "-mmi",
        mmi_cutoff_value,
        "-c",
        str(num_copies),
        "-rs",
        str(random_seed),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    print("=" * 90)
    print("RANDOM FEATURE SELECTION WITH MULTIPLE ORGANISMS AND CUTOFF VALUES")
    print("=" * 90)
    print(f"Random Seed: {GLOBAL_CONFIG['random_seed']}")
    print("=" * 90)

    total_runs = sum(len(config["mmi_cutoff_values"]) for config in CONFIG.values())
    current_run = 0

    # Loop through each organism
    for organism in CONFIG.keys():
        organism_config = CONFIG[organism]
        num_copies = organism_config["num_copies"]
        cutoff_values = organism_config["mmi_cutoff_values"]

        print(f"\n{'*' * 90}")
        print(f"* ORGANISM: {organism.upper()}")
        print(f"{'*' * 90}")
        print(f"Number of Copies: {num_copies}")
        print(f"Cutoff Values: {cutoff_values}")
        print(f"{'*' * 90}\n")

        # Loop through each cutoff value
        for i, cutoff in enumerate(cutoff_values, 1):
            current_run += 1
            print(f"\n{'#' * 90}")
            print(
                f"# Run {current_run}/{total_runs} | Organism: {organism} | Cutoff: {cutoff}"
            )
            print(f"{'#' * 90}\n")

            run_random_feature_selection(
                organism=organism,
                mmi_cutoff_value=cutoff,
                num_copies=num_copies,
                random_seed=GLOBAL_CONFIG["random_seed"],
            )

    print(f"\n{'=' * 90}")
    print("ALL RANDOM FEATURE SELECTION RUNS COMPLETED!")
    print(f"{'=' * 90}")
