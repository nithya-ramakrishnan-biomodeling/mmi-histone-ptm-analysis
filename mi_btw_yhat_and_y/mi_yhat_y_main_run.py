#!/usr/bin/env python3
"""
Mutual Information Between Predicted and Actual Values - Main Runner

This script automatically runs MI calculations for BOTH yeast and human organisms
with all predefined cutoffs and all-remaining features.

Usage:
    python mi_yhat_y_main_run.py [-p]
"""
import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

import notebook__init__b
from path_utils.path_handler import ProjectPaths

# Fixed configuration - all organisms and their cutoffs
CUTOFF_DICT_ALL = {
    "yeast": {
        "le": ["-0.44", "-0.43", "-0.41"],
    },
    "human": {
        "le": ["-0.6", "-0.7", "-0.9"],
    },
}


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


def run_mmi_calculations(organism, project_dir, parallel, cpu_limit):
    """Run all MMI calculations for the given organism"""
    successful = 0
    failed = 0

    organism_cutoffs = CUTOFF_DICT_ALL.get(organism, {})

    print(f"\n  Running MMI calculations for {organism}:")
    total_cutoffs = sum(len(cutoffs) for cutoffs in organism_cutoffs.values())
    print(f"  Total MMI tasks: {total_cutoffs}")

    for operator, cutoff_list in organism_cutoffs.items():
        print(
            f"    Processing operator '{operator}' with cutoffs: {', '.join(cutoff_list)}"
        )

        for i, cutoff in enumerate(cutoff_list, 1):
            # Build command for MMI features
            mmi_cmd = [
                "python",
                os.path.join(project_dir, "mi_calculation_mmi_feature_cutoffs.py"),
                "-o",
                organism,
                "-c",
                cutoff,
                "-op",
                operator,
            ]

            # Add parallel flag if specified
            if parallel:
                mmi_cmd.extend(["-p", "--cpu_limit", str(cpu_limit)])

            # Run command
            result = run_command(
                mmi_cmd,
                f"MMI: {organism} (cutoff: {cutoff}, operator: {operator}) [{i}/{len(cutoff_list)}]",
            )

            if result:
                successful += 1
            else:
                failed += 1

    return successful, failed


def run_remaining_features_calculation(organism, project_dir, parallel, cpu_limit):
    """Run all-remaining features calculation"""
    # Build command for all-remaining features
    remain_cmd = [
        "python",
        os.path.join(project_dir, "mi_calculation_all_remain_features.py"),
        "-o",
        organism,
        "--cpu_limit",
        str(cpu_limit),
    ]

    # Handle parallel processing flag
    if not parallel:
        remain_cmd.append("--sequential")

    # Run command
    result = run_command(remain_cmd, f"All-remaining features: {organism}")

    return (1, 0) if result else (0, 1)


def process_organism(organism, project_dir, parallel, cpu_limit):
    """Process all calculations for a single organism"""
    print(f"\n{'='*60}\nProcessing Organism: {organism.upper()}\n{'='*60}")

    organism_successful = 0
    organism_failed = 0

    # Run MMI calculations
    print(f"Phase 1: MMI Feature Calculations for {organism}")
    mmi_successful, mmi_failed = run_mmi_calculations(
        organism, project_dir, parallel, cpu_limit
    )
    organism_successful += mmi_successful
    organism_failed += mmi_failed

    # Run all-remaining features calculation
    print(f"\nPhase 2: All-Remaining Features Calculation for {organism}")
    remain_successful, remain_failed = run_remaining_features_calculation(
        organism, project_dir, parallel, cpu_limit
    )
    organism_successful += remain_successful
    organism_failed += remain_failed

    print(
        f"\n{organism.upper()} Summary: {organism_successful} successful, {organism_failed} failed"
    )

    return organism_successful, organism_failed


def main():
    """Main execution function"""
    # Parse command line arguments - only parallel flag is configurable
    parser = argparse.ArgumentParser(
        description="Calculate mutual information for BOTH yeast and human organisms with all predefined settings"
    )

    parser.add_argument(
        "-p", "--parallel", action="store_true", help="Enable parallel processing"
    )

    # Fixed parameters (not configurable)
    cpu_limit = 2
    organisms = ["yeast", "human"]  # Fixed list

    args = parser.parse_args()

    # Get directory of this script
    project_dir = str(Path(__file__).resolve().parent)

    # Track global results
    total_successful = 0
    total_failed = 0
    organism_results = {}

    # Start timing
    start_time = time.time()

    print(f"\n{'='*80}\nMutual Information Calculation Pipeline\n{'='*80}")
    print(f"Organisms: {', '.join(organisms)} (FIXED)")
    print(f"Methods: MMI (all cutoffs) + All-remaining features (FIXED)")
    print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"CPU limit: {cpu_limit} (FIXED)")

    # Calculate and display total expected tasks
    total_expected_tasks = 0
    for organism in organisms:
        organism_cutoffs = CUTOFF_DICT_ALL.get(organism, {})
        mmi_tasks = sum(len(cutoffs) for cutoffs in organism_cutoffs.values())
        remaining_tasks = 1
        organism_total = mmi_tasks + remaining_tasks
        total_expected_tasks += organism_total
        print(
            f"  {organism}: {organism_total} tasks ({mmi_tasks} MMI + {remaining_tasks} remaining)"
        )

    print(f"Total expected tasks: {total_expected_tasks}")

    # Process each organism
    for organism in organisms:
        organism_successful, organism_failed = process_organism(
            organism, project_dir, args.parallel, cpu_limit
        )

        # Store results for this organism
        organism_results[organism] = {
            "successful": organism_successful,
            "failed": organism_failed,
        }

        # Add to global totals
        total_successful += organism_successful
        total_failed += organism_failed

    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    # Print final comprehensive summary
    print(f"\n{'='*80}\nFINAL PIPELINE SUMMARY\n{'='*80}")
    print(f"Organisms processed: {', '.join(organisms)}")
    print(f"Expected total tasks: {total_expected_tasks}")
    print(f"Completed total tasks: {total_successful + total_failed}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(
        f"Overall success rate: {(total_successful/(total_successful + total_failed)*100):.1f}%"
        if (total_successful + total_failed) > 0
        else "N/A"
    )
    print(
        f"Total runtime: {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds"
    )

    # Print per-organism breakdown
    print(f"\nPer-Organism Breakdown:")
    for organism in organisms:
        results = organism_results[organism]
        organism_cutoffs = CUTOFF_DICT_ALL.get(organism, {})
        mmi_tasks = sum(len(cutoffs) for cutoffs in organism_cutoffs.values())

        print(f"  {organism.upper()}:")
        print(f"    Total tasks: {results['successful'] + results['failed']}")
        print(f"    Successful: {results['successful']}")
        print(f"    Failed: {results['failed']}")
        print(
            f"    MMI tasks: {mmi_tasks} (cutoffs: {', '.join([', '.join(cutoffs) for cutoffs in organism_cutoffs.values()])})"
        )
        print(f"    Remaining features: 1 task")

    if total_failed > 0:
        print(f"\n[WARNING]: {total_failed} tasks failed across all organisms.")
        print("Please check the error messages above for details.")
        return 1
    else:
        print(
            f"\n[SUCCESS]: All {total_successful} tasks completed successfully across all organisms!"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
