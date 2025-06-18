#!/usr/bin/env python3
"""
Multivariate MI Analysis - Main Runner

This script orchestrates the execution of all multivariate mutual information analysis
scripts for both human and yeast organisms.

It runs:
1. Basic multivariate MI calculations
2. MI redundancy removal 
3. Three-way MI value splitting
4. Bin-based entropy calculations for all PTMs
"""
import subprocess
import os
import argparse
from pathlib import Path
import time

def run_command(command, description=None):
    """Execute a command and handle results"""
    if description:
        print(f"\n{'-'*80}\n{description}\n{'-'*80}")
    
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"[SUCCESS]: {description}")
        if result.stdout.strip():
            print("Output summary:")
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                for line in lines[:5]:
                    print(f"  {line}")
                print("  ...")
                for line in lines[-5:]:
                    print(f"  {line}")
            else:
                for line in lines:
                    print(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED]: {description}")
        print(f"Exit code: {e.returncode}")
        print(f"Error message: {e.stderr}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run multivariate MI analysis pipeline")
    parser.add_argument("-o", "--organisms", nargs='+', default=["yeast", "human"],
                      help="List of organisms to analyze")
    parser.add_argument("-p", "--parallel", action="store_true",
                      help="Enable parallel processing when possible")
    parser.add_argument("-s", "--skip", nargs='+', default=[],
                      help="Scripts to skip (e.g., 'entropy' to skip bin-based entropy calculation)")
    args = parser.parse_args()
    
    # Get organisms from arguments or use defaults
    organism_list = args.organisms
    
    # Track execution times
    start_time = time.time()
    
    root_dir = os.getcwd()
    project_dir_name = Path(__file__).resolve().parent
    
    # Define scripts to run - added bin_num_based_entropy_calculation.py
    python_files = [
        "multivariate_run.py",
        "mi_red_rmvr.py",
        "three_mi_value_splitter.py",
        "bin_num_based_entropy_calculation.py",
    ]
    
    # Description for each script
    script_descriptions = {
        "multivariate_run.py": "Basic multivariate MI calculations",
        "mi_red_rmvr.py": "MI redundancy removal",
        "three_mi_value_splitter.py": "Three-way MI value splitting",
        "bin_num_based_entropy_calculation.py": "Bin-based entropy calculations for all PTMs",
    }
    
    # Maps for special command handling
    special_commands = {
        "bin_num_based_entropy_calculation.py": {
            # Since this script handles all organisms in one run
            "all_organisms": True,
        }
    }
    
    print(f"\n{'='*80}\nMULTIVARIATE MI ANALYSIS PIPELINE\n{'='*80}")
    print(f"Analyzing organisms: {', '.join(organism_list)}")
    print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    
    # Skip list handling
    skip_scripts = []
    for skip_item in args.skip:
        if skip_item == "entropy":
            skip_scripts.append("bin_num_based_entropy_calculation.py")
        else:
            for script in python_files:
                if skip_item in script:
                    skip_scripts.append(script)
    
    if skip_scripts:
        print(f"Skipping scripts: {', '.join(skip_scripts)}")
        python_files = [f for f in python_files if f not in skip_scripts]
    
    # Track results
    successful = 0
    failed = 0
    
    # Special case: Run entropy calculation once (handles all organisms internally)
    for python_file in python_files:
        # Check if this is a special all-organisms script
        if python_file in special_commands and special_commands[python_file].get("all_organisms", False):
            print(f"\n{'='*80}")
            print(f"Running {python_file} (processes all organisms internally)")
            print(f"{'='*80}")
            
            # Build the command - pass all organisms at once
            command = [
                "python",
                os.path.join(root_dir, project_dir_name, python_file),
                "-o"
            ] + organism_list
            
            # Add parallel flag if requested
            if args.parallel:
                command.append("-p")
            
            # Run the command
            script_desc = script_descriptions.get(python_file, python_file)
            result = run_command(command, f"{script_desc} for all organisms")
            
            if result:
                successful += 1
            else:
                failed += 1
                
        else:
            # Standard case: Run once per organism
            for organism in organism_list:
                print(f"\n{'='*80}")
                print(f"Running {python_file} for {organism}")
                print(f"{'='*80}")
                
                # Build the command
                command = [
                    "python",
                    os.path.join(root_dir, project_dir_name, python_file),
                    "-o",
                    organism,
                ]
                
                # Add parallel flag if requested
                if args.parallel:
                    command.append("-p")
                
                # Run the command
                script_desc = script_descriptions.get(python_file, python_file)
                result = run_command(command, f"{script_desc} for {organism}")
                
                if result:
                    successful += 1
                else:
                    failed += 1
    
    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    minutes, seconds = divmod(runtime_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    
    # Print summary
    print(f"\n{'='*80}\nMultivariate MI Analysis Summary\n{'='*80}")
    print(f"Total tasks: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Runtime: {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds")
    
    if failed > 0:
        print("\nSome tasks failed. Please check the error messages above.")
        return 1
    else:
        print("\nAll tasks completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())
