#!/usr/bin/env python3
# filepath: c:\Users\dhana\OneDrive\Documents\multivariate_ptms_analysis-\pseudo_inverse\pseudo_invrs_main_run.py
"""
Pseudo-Inverse Analysis Pipeline

This script runs the pseudo-inverse feature selection and R² calculation for
multiple organisms. It automates the execution of r2_score_calcn_pseudo_inverse_based_feature.py
with appropriate parameters.

Usage:
    python pseudo_invrs_main_run.py [-o organism_list] [-n features] [-p]

Parameters:
    -o, --organisms: List of organisms to analyze (default: yeast, human)
    -n, --n_features: Number of features to select (default: 8)
    -p, --parallel: Run in parallel mode when possible
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, description=None):
    """Execute a command and handle results
    
    Parameters
    ----------
    command : list
        Command to execute as a list of strings
    description : str, optional
        Description of the command for output
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
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
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run pseudo-inverse analysis pipeline")
    parser.add_argument("-o", "--organisms", nargs='+', default=["yeast", "human"],
                      help="List of organisms to analyze (default: yeast, human)")
    parser.add_argument("-n", "--n_features", type=int, default=8,
                      help="Number of features to select (default: 8)")
    parser.add_argument("-p", "--parallel", action="store_true",
                      help="Enable parallel processing when possible")
    args = parser.parse_args()
    
    # Get parameters from arguments
    organism_list = args.organisms
    n_features = args.n_features
    parallel_mode = args.parallel
    
    # Get script directory
    project_dir = os.getcwd()
    pseudo_inverse_script = os.path.join(
        project_dir,
        "pseudo_inverse", 
        "r2_score_calcn_pseudo_inverse_based_feature.py"
    )
    
    print(f"\n{'='*80}\nPseudo-Inverse Analysis Pipeline\n{'='*80}")
    print(f"Analyzing organisms: {', '.join(organism_list)}")
    print(f"Using top {n_features} features")
    print(f"Parallel mode: {'Enabled' if parallel_mode else 'Disabled'}")
    
    # Check if the script exists
    if not os.path.exists(pseudo_inverse_script):
        print(f"Error: Script not found at {pseudo_inverse_script}")
        return 1
    
    # Track results
    successful = 0
    failed = 0
    
    # Run the script for each organism
    for organism in organism_list:
        # Build the command
        command = [
            "python", 
            pseudo_inverse_script,
            "-o", organism,
            "-n", str(n_features)
        ]
        
        # Add parallel flag if requested
        if parallel_mode:
            command.append("-p")
            
        # Run the command
        description = f"Pseudo-inverse analysis for {organism} with {n_features} features"
        result = run_command(command, description)
        
        if result:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*80}\nAnalysis Summary\n{'='*80}")
    print(f"Total tasks: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n[WARNING]: {failed} tasks")
        print("\nSome tasks failed. Please check the error messages above.")
        return 1
    else:
        print(f"\n[SUCCESS]: All {successful} tasks")
        print("\nAll tasks completed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())