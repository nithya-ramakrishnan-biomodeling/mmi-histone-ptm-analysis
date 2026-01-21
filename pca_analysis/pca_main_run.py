#!/usr/bin/env python3
"""
PCA Analysis Pipeline Runner

This script orchestrates the execution of all PCA analysis scripts with
appropriate parameters for each organism and method.

It runs:
1. Eigen decomposition analysis
2. PCA components generation with both covariance and MI methods
"""
import subprocess
import os
import argparse
from pathlib import Path


def run_command(command, description=None):
    """Execute a command and handle results"""
    if description:
        print(f"\n{'-'*80}\n{description}\n{'-'*80}")

    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[SUCCESS]: {description}")
        if result.stdout.strip():
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED]: {description}")
        print(f"Exit code: {e.returncode}")
        print(f"Error message: {e.stderr}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run PCA analysis pipeline")
    parser.add_argument(
        "-o",
        "--organisms",
        nargs="+",
        default=["yeast", "human"],
        help="List of organisms to analyze",
    )

    parser.add_argument(
        "-c",
        "--components",
        type=int,
        default=5,
        help="Number of PCA components to generate",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Enable parallel processing when possible",
    )
    args = parser.parse_args()

    # Get organisms from arguments or use defaults
    organism_list = args.organisms
    n_components = args.components

    # Get the current working directory and script directory
    root_dir = os.getcwd()
    project_dir = str(Path(__file__).resolve().parent)

    print(f"\n{'='*80}\nPCA Analysis Pipeline\n{'='*80}")
    print(f"Analyzing organisms: {', '.join(organism_list)}")
    print(f"Using {n_components} PCA components")

    # List of scripts to run
    analysis_tasks = [
        {
            "script": "eigen_decomposition.py",
            "description": "Eigen decomposition analysis",
            "params": {"-o": None},  # None will be replaced with organism
        },
        {
            "script": "pca_components.py",
            "description": "PCA components using covariance method",
            "params": {"-o": None, "-m": "cov", "-n": str(n_components)},
        },
        {
            "script": "pca_components.py",
            "description": "PCA components using mutual information method",
            "params": {"-o": None, "-m": "mi", "-n": str(n_components)},
        },
    ]

    # Track results
    successful = 0
    failed = 0

    # Run each analysis task for each organism
    for task in analysis_tasks:
        for organism in organism_list:
            # Build the command
            command = ["python", os.path.join(project_dir, task["script"])]

            # Add parameters
            for param, value in task["params"].items():
                command.append(param)
                if value is None:
                    # Replace None with organism
                    command.append(organism)
                else:
                    command.append(value)

            # Add parallel flag if requested
            if args.parallel and "-p" not in command:
                command.append("-p")

            # Run the command
            description = f"{task['description']} for {organism}"
            result = run_command(command, description)

            if result:
                successful += 1
            else:
                failed += 1

    # Print summary
    print(f"\n{'='*80}\nPCA Analysis Summary\n{'='*80}")
    print(f"Total tasks: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nSome tasks failed. Please check the error messages above.")
        return 1
    else:
        print("\nAll tasks completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
