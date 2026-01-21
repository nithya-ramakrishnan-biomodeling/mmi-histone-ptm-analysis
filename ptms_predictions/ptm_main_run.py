import subprocess
import os
import argparse
from pathlib import Path

# Define command line arguments
parser = argparse.ArgumentParser(
    description="Run histone modification prediction pipeline"
)
parser.add_argument(
    "-p",
    "--parallel",
    action="store_true",
    help="Enable parallel processing for all scripts that support it",
)
args = parser.parse_args()

# Get the parallel processing flag
parallel_mode = args.parallel

# Get the current working directory
root_dir = os.getcwd()
# Get the directory of the current script
project_dir_name = Path(__file__).resolve().parent

# Define your organisms and cutoffs
organism_list = ["yeast", "human"]
operator_list = ["le", "ge"]
cutoff_dict_all = {
    "yeast": {
        "le": ["-0.44", "-0.43", "-0.41"],
    },
    "human": {
        "le": ["-0.6", "-0.7", "-0.9"],
    },
}

# Target histones for individual and cumulative analysis
target_histones = ["H3K9me3", "H3K79me3"]

print(f"Running in {'parallel' if parallel_mode else 'sequential'} mode")


# Function to execute commands
def run_command(command, description=None):
    """Execute a command and handle results"""
    if description:
        print(f"\n{'-'*80}\n{description}\n{'-'*80}")

    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Execution completed successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command:")
        print(f"Exit code: {e.returncode}")
        print(f"Error message: {e.stderr}")
        return False


# ===== First: Run 3MMI Feature Extractor =====
print("\n=== Running 3MMI Feature Extractor ===")
for organism in organism_list:
    for operator in operator_list:
        if operator in cutoff_dict_all[organism]:
            description = f"3MMI Feature extraction for organism: {organism}, operator: {operator}"

            command = [
                "python",
                os.path.join(
                    project_dir_name, "3mmi_feature_extractor_with_mmicutoff.py"
                ),  # Fixed typo
                "-o",
                organism,
                "-op",
                operator,
            ]

            run_command(command, description)


# ===== MMI Features Script =====
print("\n=== Running MMI Features Prediction ===")
for organism in organism_list:
    for operator in operator_list:
        if operator in cutoff_dict_all[organism]:
            for cutoff in cutoff_dict_all[organism][operator]:
                description = f"MMI Features prediction for organism: {organism}, cutoff: {cutoff}, operator: {operator}"

                command = [
                    "python",
                    os.path.join(project_dir_name, "r2_prediction_mmi_features.py"),
                    "-o",
                    organism,
                    "-c",
                    cutoff,
                    "-op",
                    operator,
                ]

                # Add parallel flag if parallel mode is enabled
                if parallel_mode:
                    command.append("-p")

                run_command(command, description)

# ===== MMI and Highest Correlation Script =====
print("\n=== Running MMI and Highest Correlation Prediction ===")
for organism in organism_list:
    for operator in operator_list:
        if operator in cutoff_dict_all[organism]:
            for cutoff in cutoff_dict_all[organism][operator]:
                description = f"MMI and Highest Correlation prediction for organism: {organism}, cutoff: {cutoff}, operator: {operator}"

                command = [
                    "python",
                    os.path.join(
                        project_dir_name, "r2_prediction_mmi_and_highest_corr.py"
                    ),  # Fixed typo
                    "-o",
                    organism,
                    "-c",
                    cutoff,
                    "-op",
                    operator,
                ]

                # Add parallel flag if parallel mode is enabled
                if parallel_mode:
                    command.append("-p")

                run_command(command, description)

# ===== All Remaining Features Script =====
print("\n=== Running All Remaining Features Prediction ===")
for organism in organism_list:
    description = f"All remaining features prediction for organism: {organism}"

    command = [
        "python",
        os.path.join(project_dir_name, "r2_prediction_all_remain_features.py"),
        "-o",
        organism,
    ]

    # Add parallel flag if parallel mode is enabled
    if parallel_mode:
        command.append("-p")

    run_command(command, description)

# ===== PCA Transformed Features Script =====
print("\n=== Running PCA Transformed Features Prediction ===")
for organism in organism_list:
    for method in ["mi", "cov"]:  # PCA analysis methods
        description = f"PCA transformed features prediction for organism: {organism}, method: {method}"

        command = [
            "python",
            os.path.join(project_dir_name, "r2_prediction_PCA_transformed_features.py"),
            "-o",
            organism,
            "-m",
            method,
        ]

        # Add parallel flags if parallel mode is enabled
        if parallel_mode:
            command.extend(["-p", "-j", "-2"])  # Use all cores except one

        run_command(command, description)

# ===== Pseudo Inverse Based Feature Script =====
print("\n=== Running Pseudo-Inverse Based Feature Prediction ===")
for organism in organism_list:
    description = f"Pseudo-inverse based feature prediction for organism: {organism}"

    command = [
        "python",
        os.path.join(project_dir_name, "r2_prediction_pseudo_inverse_based_feature.py"),
        "-o",
        organism,
    ]

    # Add parallel flag if parallel mode is enabled
    if parallel_mode:
        command.append("-p")

    run_command(command, description)

# ===== OMP Based XGB Script =====
print("\n=== Running OMP Based XGB Feature Prediction ===")
for organism in organism_list:
    description = f"OMP based XGB feature prediction for organism: {organism}"

    command = [
        "python",
        os.path.join(
            project_dir_name, "r2_prediction_omp_based_xgb_n_number_feature.py"
        ),
        "-o",
        organism,
        "-n",
        "8",  # Default number of features to select
    ]

    # Add parallel flag if parallel mode is enabled
    if parallel_mode:
        command.append("-p")

    run_command(command, description)

# ===== Individual Cumulative Analysis =====
print("\n=== Running Individual Cumulative Analysis ===")
for organism in organism_list:
    if organism == "human":
        for target in target_histones:
            description = f"Individual cumulative analysis for organism: {organism}, target: {target}"

            command = [
                "python",
                os.path.join(
                    project_dir_name, "r2_prediction_individual_cumulative.py"
                ),  # Fixed typo
                "-o",
                organism,
                "-t",
                target,
            ]

            # Add CPU limit for parallel processing if enabled
            if parallel_mode:
                command.extend(["--cpu_limit", "4"])

            run_command(command, description)

print("\nAll analyses completed!")
