#!/usr/bin/env python3
"""
Multivariate PTMs Analysis - Main Pipeline

This script orchestrates the full analysis pipeline by running all main modules in the correct sequence:
1. PCA Analysis
2. Multivariate MI Analysis
3. PTM Predictions
4. Pseudo-Inverse Analysis
5. MI Between Y-hat and Y

Usage:
    python main.py [-m <modules>] [-p] [-s] [-v]
"""
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Define module information
MODULES = {
    "pca": {
        "name": "PCA Analysis",
        "script": os.path.join("pca_analysis", "pca_main_run.py"),
        "description": "Principal Component Analysis of histone modifications",
    },
    "multivariate": {
        "name": "Multivariate MI Analysis",
        "script": os.path.join("multivariate_mi_analysis", "multivariate_main_run.py"),
        "description": "Multivariate mutual information analysis",
    },
    "ptm": {
        "name": "PTM Predictions",
        "script": os.path.join("ptms_predictions", "ptm_main_run.py"),
        "description": "Histone modification predictions using various methods",
    },
    "pseudo": {
        "name": "Pseudo-Inverse Analysis",
        "script": os.path.join("pseudo_inverse", "pseudo_invrs_main_run.py"),
        "description": "Pseudo-inverse based feature selection and predictions",
    },
    "mi_yhat": {
        "name": "MI Between Y-hat and Y",
        "script": os.path.join("mi_btw_yhat_and_y", "mi_yhat_y_main_run.py"),
        "description": "Mutual information between predicted and actual values",
    },
}

# Module order for execution
MODULE_ORDER = ["pca", "multivariate", "ptm", "pseudo", "mi_yhat"]


def run_command(command, description=None, verbose=False):
    """Execute a command and handle results"""
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}")

    print(f"Running command: {' '.join(command)}")
    try:
        if verbose:
            # Show output in real-time
            result = subprocess.run(command, check=True)
            success = True
        else:
            # Capture output and only show on error
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            success = True
            if result.stdout.strip():
                print("\nCommand output summary:")
                # Print just the first 5 lines and last 5 lines if output is long
                lines = result.stdout.strip().split("\n")
                if len(lines) > 12:
                    for line in lines[:5]:
                        print(f"  {line}")
                    print("  ...")
                    for line in lines[-5:]:
                        print(f"  {line}")
                else:
                    for line in lines:
                        print(f"  {line}")

        print(f"[SUCCESS]: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED]: {description} failed with exit code {e.returncode}")
        if not verbose and e.stdout:
            print("\nCommand output:")
            print(e.stdout)
        if e.stderr:
            print("\nError output:")
            print(e.stderr)
        return False


def run_module(module_key, parallel=False, verbose=False):
    """Run a specific analysis module"""
    module = MODULES[module_key]
    script_path = module["script"]

    if not os.path.exists(script_path):
        print(f"[FAILED]: Script not found at {script_path}")
        return False

    # Build command based on the module
    command = ["python", script_path]

    # Add parallel flag if supported and requested
    if parallel and module_key in ["pca", "multivariate", "ptm", "pseudo", "mi_yhat"]:
        command.append("-p")

    # Run the command
    start_time = time.time()
    success = run_command(command, f"Running {module['name']}", verbose=verbose)
    runtime = time.time() - start_time

    # Print timing information
    minutes, seconds = divmod(runtime, 60)
    hours, minutes = divmod(minutes, 60)
    timing_msg = f"Module runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s"

    if success:
        print(f"[SUCCESS] {module['name']} completed successfully! {timing_msg}")
    else:
        print(f"[FAILED] {module['name']} failed! {timing_msg}")

    return success


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the full multivariate PTMs analysis pipeline"
    )
    parser.add_argument(
        "-m",
        "--modules",
        nargs="+",
        choices=MODULE_ORDER + ["all"],
        default=["all"],
        help="Specific modules to run (default: all)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Enable parallel processing for all modules that support it",
    )
    parser.add_argument(
        "-s",
        "--stop-on-error",
        action="store_true",
        help="Stop execution if any module fails",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show full output from each module"
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Determine which modules to run
    if "all" in args.modules:
        modules_to_run = MODULE_ORDER
    else:
        modules_to_run = [m for m in MODULE_ORDER if m in args.modules]

    # Setup for execution tracking
    total_modules = len(modules_to_run)
    successful_modules = 0
    failed_modules = 0
    results = {}

    # Start timing
    pipeline_start = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*80}")
    print(f"MULTIVARIATE PTMs ANALYSIS PIPELINE")
    print(f"{'='*80}")
    print(f"Start time: {start_time_str}")
    print(f"Modules to run: {', '.join(modules_to_run)}")
    print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"Stop on error: {'Yes' if args.stop_on_error else 'No'}")

    # Run each module in order
    for i, module_key in enumerate(modules_to_run, 1):
        module = MODULES[module_key]
        print(f"\n{'='*80}")
        print(f"MODULE {i}/{total_modules}: {module['name'].upper()}")
        print(f"Description: {module['description']}")
        print(f"{'='*80}")

        # Run the module
        success = run_module(module_key, parallel=args.parallel, verbose=args.verbose)

        # Track results
        results[module_key] = success
        if success:
            successful_modules += 1
        else:
            failed_modules += 1
            if args.stop_on_error:
                print(f"\n[FAILED]: Stopping execution due to module failure")
                break

    # Calculate total runtime
    pipeline_runtime = time.time() - pipeline_start
    hours, remainder = divmod(pipeline_runtime, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Start time: {start_time_str}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"Total modules: {total_modules}")
    print(f"Successful: {successful_modules}")
    print(f"Failed: {failed_modules}")

    # Print detailed results
    print("\nDetailed Results:")
    for module_key in modules_to_run:
        status = "[SUCCESS]" if results.get(module_key, False) else "[FAILED]"
        print(f"  {status}: {MODULES[module_key]['name']}")

    # Return failure if any module failed
    return 0 if failed_modules == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
