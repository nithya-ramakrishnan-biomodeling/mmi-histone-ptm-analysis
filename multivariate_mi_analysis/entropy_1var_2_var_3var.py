""" idea: 
calculate entropy for 1 variable, 2 variables, anḍ 3 variables
if mi argument is passed, calculate mutual information instead of entropy
if both arguments are passed, calculate both entropy and mutual information
store them in csv files with appropriate headers
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import time
import argparse
from itertools import combinations

import notebook__init__b
from path_utils.path_handler import ProjectPaths
from datahandler import data_handler
import entropy_mutualinfo


# pseudo code
"""" 
1. parse arguments for mi and entropy calculation
    it includes: first argument as histone modification name
2. load data for the given histone modification
3. calculate entropy for 1 variable, 2 variables, and 3 variables
4. if mi argument is passed, calculate mutual information instead of entropy
5. store results in csv files with appropriate headers
"""


def generate_bin_edges(
    global_min: float, global_max: float, bin_num: int
) -> np.ndarray:
    """Generate bin edges for histogram calculation.

    Parameters
    ----------
    global_min : float
        Minimum value for binning.
    global_max : float
        Maximum value for binning.
    bin_num : int
        Number of bins.

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    return np.linspace(global_min, global_max, bin_num + 1)


def calculate_1var_entropy(
    histone_df: pd.DataFrame, histone_mods: list, bin_edges: np.ndarray
) -> pd.DataFrame:
    """Calculate entropy for single variables.

    Parameters
    ----------
    histone_df : pd.DataFrame
        DataFrame containing histone modification data.
    histone_mods : list
        List of histone modification column names.
    bin_edges : np.ndarray
        Bin edges for histogram.

    Returns
    -------
    pd.DataFrame
        DataFrame with entropy values for each variable.
    """
    results = []

    for mod in tqdm(histone_mods, desc="Calculating 1-variable entropy"):
        if mod not in histone_df.columns:
            print(f"Warning: {mod} not found in data columns. Skipping.")
            continue

        arr = histone_df[mod].values
        # Calculate joint probability (1D case - marginal probability)
        joint_prob_1d = entropy_mutualinfo.joint_prob(arr, arr, bin_edges)
        marginal_prob = entropy_mutualinfo.marginal_prob_calculator(
            joint_prob_1d, axis=(1)
        )
        entropy_val = entropy_mutualinfo.entropy_calculator(marginal_prob)

        results.append({"variable": mod, "entropy": entropy_val})

    return pd.DataFrame(results)


def calculate_2var_entropy(
    histone_df: pd.DataFrame, histone_mods: list, bin_edges: np.ndarray
) -> pd.DataFrame:
    """Calculate joint entropy for pairs of variables.

    Parameters
    ----------
    histone_df : pd.DataFrame
        DataFrame containing histone modification data.
    histone_mods : list
        List of histone modification column names.
    bin_edges : np.ndarray
        Bin edges for histogram.

    Returns
    -------
    pd.DataFrame
        DataFrame with joint entropy values for each pair.
    """
    results = []

    # Generate all pairs
    pairs = list(combinations(histone_mods, 2))

    for mod1, mod2 in tqdm(pairs, desc="Calculating 2-variable entropy"):
        if mod1 not in histone_df.columns or mod2 not in histone_df.columns:
            print(f"Warning: {mod1} or {mod2} not found in data columns. Skipping.")
            continue

        arr1 = histone_df[mod1].values
        arr2 = histone_df[mod2].values

        # Calculate joint probability for 2 variables
        joint_prob_2d = entropy_mutualinfo.joint_prob(arr1, arr2, bin_edges)
        joint_entropy = entropy_mutualinfo.entropy_calculator(joint_prob_2d)

        results.append(
            {"variable_1": mod1, "variable_2": mod2, "joint_entropy": joint_entropy}
        )

    return pd.DataFrame(results)


def calculate_2var_mi(
    histone_df: pd.DataFrame, histone_mods: list, bin_edges: np.ndarray
) -> pd.DataFrame:
    """Calculate mutual information for pairs of variables.

    Parameters
    ----------
    histone_df : pd.DataFrame
        DataFrame containing histone modification data.
    histone_mods : list
        List of histone modification column names.
    bin_edges : np.ndarray
        Bin edges for histogram.

    Returns
    -------
    pd.DataFrame
        DataFrame with mutual information values for each pair.
    """
    results = []

    # Generate all pairs
    pairs = list(combinations(histone_mods, 2))

    for mod1, mod2 in tqdm(pairs, desc="Calculating 2-variable mutual information"):
        if mod1 not in histone_df.columns or mod2 not in histone_df.columns:
            print(f"Warning: {mod1} or {mod2} not found in data columns. Skipping.")
            continue

        arr1 = histone_df[mod1].values
        arr2 = histone_df[mod2].values

        # Calculate mutual information using the function from entropy_mutualinfo
        mi_val = entropy_mutualinfo.mi_betwn_uandy(arr1, arr2, bin_edges)

        results.append(
            {"variable_1": mod1, "variable_2": mod2, "mutual_information": mi_val}
        )

    return pd.DataFrame(results)


def calculate_3var_entropy(
    histone_df: pd.DataFrame, histone_mods: list, bin_edges: np.ndarray
) -> pd.DataFrame:
    """Calculate joint entropy for triplets of variables.

    Parameters
    ----------
    histone_df : pd.DataFrame
        DataFrame containing histone modification data.
    histone_mods : list
        List of histone modification column names.
    bin_edges : np.ndarray
        Bin edges for histogram.

    Returns
    -------
    pd.DataFrame
        DataFrame with joint entropy values for each triplet.
    """
    results = []

    # Generate all triplets
    triplets = list(combinations(histone_mods, 3))

    for mod1, mod2, mod3 in tqdm(triplets, desc="Calculating 3-variable entropy"):
        if (
            mod1 not in histone_df.columns
            or mod2 not in histone_df.columns
            or mod3 not in histone_df.columns
        ):
            print(
                f"Warning: {mod1}, {mod2}, or {mod3} not found in data columns. Skipping."
            )
            continue

        arr1 = histone_df[mod1].values
        arr2 = histone_df[mod2].values
        arr3 = histone_df[mod3].values

        # Calculate joint probability for 3 variables
        joint_prob_3d = entropy_mutualinfo.joint_prob(
            arr1, arr2, bin_edges, input_list_3=arr3
        )
        joint_entropy = entropy_mutualinfo.entropy_calculator(joint_prob_3d)

        results.append(
            {
                "variable_1": mod1,
                "variable_2": mod2,
                "variable_3": mod3,
                "joint_entropy": joint_entropy,
            }
        )

    return pd.DataFrame(results)


def calculate_3var_mi(
    histone_df: pd.DataFrame, histone_mods: list, bin_edges: np.ndarray
) -> pd.DataFrame:
    """Calculate 3-variable mutual information (interaction information) for triplets.

    This calculates I(U;Y) - I(U;Y|V) = H(U) + H(V) + H(Y) - H(U,V) - H(V,Y) - H(U,Y) + H(U,V,Y)

    Parameters
    ----------
    histone_df : pd.DataFrame
        DataFrame containing histone modification data.
    histone_mods : list
        List of histone modification column names.
    bin_edges : np.ndarray
        Bin edges for histogram.

    Returns
    -------
    pd.DataFrame
        DataFrame with 3-variable MI values for each triplet.
    """
    results = []

    # Generate all triplets
    triplets = list(combinations(histone_mods, 3))

    for mod1, mod2, mod3 in tqdm(
        triplets, desc="Calculating 3-variable mutual information"
    ):
        if (
            mod1 not in histone_df.columns
            or mod2 not in histone_df.columns
            or mod3 not in histone_df.columns
        ):
            print(
                f"Warning: {mod1}, {mod2}, or {mod3} not found in data columns. Skipping."
            )
            continue

        arr1 = histone_df[mod1].values
        arr2 = histone_df[mod2].values
        arr3 = histone_df[mod3].values

        # Calculate 3-variable MI using diff_uy_and_uy_givenv
        mi_3var = entropy_mutualinfo.diff_uy_and_uy_givenv(arr1, arr2, arr3, bin_edges)

        results.append(
            {
                "variable_1": mod1,
                "variable_2": mod2,
                "variable_3": mod3,
                "multivariate_mi": mi_3var,
            }
        )

    return pd.DataFrame(results)


def save_results(results_dict: dict, output_dir: str, organism: str):
    """Save results to CSV files.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing DataFrames with results.
    output_dir : str
        Directory to save output files.
    organism : str
        Name of the organism.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for key, df in results_dict.items():
        if df is not None and not df.empty:
            filename = f"{organism}_{key}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {key} results to: {filepath}")

    # Concatenate all results into a single CSV file
    concatenate_results(results_dict, output_dir, organism, timestamp)


def concatenate_results(
    results_dict: dict, output_dir: str, organism: str, timestamp: str
):
    """Concatenate all results into a single CSV file with a result_type column.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing DataFrames with results.
    output_dir : str
        Directory to save output file.
    organism : str
        Name of the organism.
    timestamp : str
        Timestamp for the filename.
    """
    combined_dfs = []

    for key, df in results_dict.items():
        if df is not None and not df.empty:
            # Add a column to identify the result type
            df_copy = df.copy()
            df_copy["result_type"] = key

            # Standardize column names for concatenation
            # Rename columns to have consistent naming
            if "variable" in df_copy.columns:
                df_copy = df_copy.rename(columns={"variable": "variable_1"})
            if "entropy" in df_copy.columns:
                df_copy = df_copy.rename(columns={"entropy": "value"})
            if "joint_entropy" in df_copy.columns:
                df_copy = df_copy.rename(columns={"joint_entropy": "value"})
            if "mutual_information" in df_copy.columns:
                df_copy = df_copy.rename(columns={"mutual_information": "value"})
            if "multivariate_mi" in df_copy.columns:
                df_copy = df_copy.rename(columns={"multivariate_mi": "value"})

            combined_dfs.append(df_copy)

    if combined_dfs:
        # Concatenate all DataFrames
        combined_df = pd.concat(combined_dfs, ignore_index=True)

        # Reorder columns for better readability
        cols = ["result_type", "variable_1"]
        if "variable_2" in combined_df.columns:
            cols.append("variable_2")
        if "variable_3" in combined_df.columns:
            cols.append("variable_3")
        cols.append("value")

        # Only include columns that exist
        cols = [c for c in cols if c in combined_df.columns]
        combined_df = combined_df[cols]

        # Save combined file
        combined_filename = f"{organism}_all_results_combined_{timestamp}.csv"
        combined_filepath = os.path.join(output_dir, combined_filename)
        combined_df.to_csv(combined_filepath, index=False)
        print(f"\nSaved combined results to: {combined_filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate entropy and mutual information for given histone modification data."
    )

    # organism name
    parser.add_argument(
        "--organism",
        type=str,
        required=True,
        help="Name of the organism (default: yeast)",
    )
    # name of the histone modifications as string with comma separated values
    parser.add_argument(
        "histone_mod", type=str, help="Name of the histone modification (e.g., H3K4me3)"
    )
    parser.add_argument(
        "--mi",
        action="store_true",
        help="Calculate mutual information instead of entropy",
    )
    parser.add_argument("--entropy", action="store_true", help="Calculate entropy")

    args = parser.parse_args()
    histone_mod = args.histone_mod
    calculate_mi = args.mi
    calculate_entropy = args.entropy
    organism = args.organism

    # If no flags are set, calculate both
    if not calculate_mi and not calculate_entropy:
        calculate_mi = True
        calculate_entropy = True

    print(f"Organism: {organism}")
    print(f"Histone Modifications: {histone_mod}")
    print(f"Calculate Entropy: {calculate_entropy}")
    print(f"Calculate Mutual Information: {calculate_mi}")

    # checking minimum number of histone modifcation for 2-var and 3-var calculations
    histone_mods = [mod.strip() for mod in histone_mod.split(",")]
    num_histone_mods = len(histone_mods)

    if num_histone_mods < 1:
        print("Error: At least one histone modification must be provided.")
        sys.exit(1)
    if num_histone_mods < 2 and calculate_mi:
        print(
            "Error: At least two histone modifications are required to calculate mutual information."
        )
        sys.exit(1)

    # load data
    data_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )
    mi_config_file_path = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), "mi_bin_info.json"
    )

    print(f"Loading data from: {data_file_path}")
    histone_df = data_handler.csv_loader(data_file_path)
    mi_config_dict = data_handler.json_file_loader(mi_config_file_path)

    if organism not in mi_config_dict:
        print(f"Error: Organism '{organism}' not found in mi_bin_info.json")
        sys.exit(1)

    organism_config = mi_config_dict[organism]
    global_min = organism_config["global_min"]
    global_max = organism_config["global_max"]
    bin_num = organism_config["bin_num"]

    print(f"Bin configuration - Min: {global_min}, Max: {global_max}, Bins: {bin_num}")

    bin_edges = generate_bin_edges(global_min, global_max, bin_num)

    results = {}

    # Calculate entropy for 1, 2, and 3 variables
    if calculate_entropy:
        print("\n--- Calculating Entropy ---")

        # 1-variable entropy
        entropy_1var_df = calculate_1var_entropy(histone_df, histone_mods, bin_edges)
        results["1var_entropy"] = entropy_1var_df
        print(f"\n1-Variable Entropy Results:\n{entropy_1var_df}")

        # 2-variable joint entropy
        if num_histone_mods >= 2:
            entropy_2var_df = calculate_2var_entropy(
                histone_df, histone_mods, bin_edges
            )
            results["2var_joint_entropy"] = entropy_2var_df
            print(f"\n2-Variable Joint Entropy Results:\n{entropy_2var_df}")

        # 3-variable joint entropy
        if num_histone_mods >= 3:
            entropy_3var_df = calculate_3var_entropy(
                histone_df, histone_mods, bin_edges
            )
            results["3var_joint_entropy"] = entropy_3var_df
            print(f"\n3-Variable Joint Entropy Results:\n{entropy_3var_df}")

    # Calculate mutual information for 2 and 3 variables
    if calculate_mi:
        print("\n--- Calculating Mutual Information ---")

        if num_histone_mods >= 2:
            mi_2var_df = calculate_2var_mi(histone_df, histone_mods, bin_edges)
            results["2var_mutual_information"] = mi_2var_df
            print(f"\n2-Variable Mutual Information Results:\n{mi_2var_df}")

        # 3-variable MI (interaction information)
        if num_histone_mods >= 3:
            mi_3var_df = calculate_3var_mi(histone_df, histone_mods, bin_edges)
            results["3var_multivariate_mi"] = mi_3var_df
            print(f"\n3-Variable Multivariate MI Results:\n{mi_3var_df}")

    # Save results to CSV files
    output_dir = os.path.join(
        ProjectPaths.get_output_dir(), "entropy_mi_results", organism
    )
    save_results(results, output_dir, organism)

    print("\n--- Calculation Complete ---")
    return results


if __name__ == "__main__":
    main()
