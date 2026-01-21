"""
Configuration file to run target PTM predictions with multiple cutoff values
- Different cutoff values per organism
- Different feature numbers per organism
- Auto-detect parallel mode based on system resources
"""

import subprocess
import sys
import psutil
import os


# Configuration variables - MODIFY THESE
CONFIG = {
    "yeast": {
        "num_copies": 3,
        "mmi_cutoff_values": ["-0.41", "-0.43", "-0.44"],
    },
    "human": {
        "num_copies": 3,
        "mmi_cutoff_values": ["-0.6", "-0.7", "-0.9"],
    },
}

# System resource settings
SYSTEM_CONFIG = {
    "random_seed": 42,  # Random seed for reproducibility
    "auto_detect_parallel": True,  # Auto-detect parallel mode based on CPU cores
    "min_cores_for_parallel": 10,  # Minimum CPU cores required for parallel mode
    "force_parallel": None,  # Set to True/False to override auto-detection (None = auto)
}


def get_system_info():
    """Get system resource information"""
    cpu_count = psutil.cpu_count(logical=False)
    memory_gb = psutil.virtual_memory().total / (1024**3)

    print(f"System Information:")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Memory: {memory_gb:.2f} GB")
    return cpu_count, memory_gb


def should_use_parallel(cpu_count):
    """Determine if parallel mode should be used"""

    # Override with force_parallel if set
    if SYSTEM_CONFIG["force_parallel"] is not None:
        return SYSTEM_CONFIG["force_parallel"]

    # Auto-detect based on CPU cores
    if SYSTEM_CONFIG["auto_detect_parallel"]:
        if cpu_count >= SYSTEM_CONFIG["min_cores_for_parallel"]:
            return True
        return False

    return False


def run_prediction(organism, mmi_cutoff_value, num_copies, parallel_mode, random_seed):
    """Run the main prediction script for a given cutoff value"""

    prediction_main_file = os.path.join(
        os.path.dirname(__file__), "target_ptm_prediction_random_main.py"
    )

    cmd = [
        sys.executable,
        prediction_main_file,
        "-o",
        organism,
        "-c",
        mmi_cutoff_value,
        "-nc",
        str(num_copies),
        "-rs",
        str(random_seed),
    ]

    if parallel_mode:
        cmd.append("-p")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    # Get system info
    cpu_count, memory_gb = get_system_info()

    # Determine parallel mode
    parallel_mode = should_use_parallel(cpu_count)

    print("=" * 80)
    print("PTM PREDICTION WITH MULTIPLE ORGANISMS AND CUTOFF VALUES")
    print("=" * 80)
    print(
        f"Parallel Mode: {parallel_mode} (Auto-detected: {SYSTEM_CONFIG['auto_detect_parallel']})"
    )
    print(f"Random Seed: {SYSTEM_CONFIG['random_seed']}")
    print("=" * 80)

    # Loop through each organism
    for organism in CONFIG.keys():
        organism_config = CONFIG[organism]
        num_copies = organism_config["num_copies"]
        cutoff_values = organism_config["mmi_cutoff_values"]

        print(f"\n{'*' * 80}")
        print(f"* ORGANISM: {organism.upper()}")
        print(f"{'*' * 80}")
        print(f"Number of Copies: {num_copies}")
        print(f"Cutoff Values: {cutoff_values}")
        print(f"{'*' * 80}\n")

        # Loop through each cutoff value
        for i, cutoff in enumerate(cutoff_values, 1):
            print(f"\n{'#' * 80}")
            print(f"# Organism: {organism} | Cutoff {i}/{len(cutoff_values)}: {cutoff}")
            print(f"{'#' * 80}\n")

            run_prediction(
                organism=organism,
                mmi_cutoff_value=cutoff,
                num_copies=num_copies,
                parallel_mode=parallel_mode,
                random_seed=SYSTEM_CONFIG["random_seed"],
            )

    print(f"\n{'=' * 80}")
    print("ALL PREDICTIONS COMPLETED!")
    print(f"{'=' * 80}")
