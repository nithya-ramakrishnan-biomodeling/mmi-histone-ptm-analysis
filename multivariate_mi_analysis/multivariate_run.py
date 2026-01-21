from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import os
from joblib import Parallel, delayed
import notebook__init__b
from mi_parallel_processing import mi_parallel_generator
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths


def create_bin_spacing(
    min_value: float, max_value: float, bin_number: int
) -> np.ndarray:
    """Creates equally spaced bins between minimum and maximum values.

    This function generates bin edges for histogram-based mutual information calculations.
    The bins are created with equal width across the entire range.

    Parameters
    ----------
    min_value : float
        The minimum value of the range
    max_value : float
        The maximum value of the range
    bin_number : int
        Number of bins to create

    Returns
    -------
    np.ndarray
        Array of bin edges with length bin_number
    """
    return np.linspace(min_value, max_value, bin_number)


def process_mutual_information(
    dataframe: pd.DataFrame,
    method: str,
    comb: int,
    spaced_bin_edges: np.ndarray,
    output_dir: str,
) -> None:
    """Calculate and save mutual information results for a specific method.

    This function handles the computation of various mutual information metrics
    and saves the results to a CSV file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data containing the variables for MI calculation
    method : str
        Type of mutual information calculation to perform
    comb : int
        Number of variables to consider (2 or 3)
    spaced_bin_edges : np.ndarray
        Pre-computed bin edges for discretization
    output_dir : str
        Directory where results will be saved

    Returns
    -------
    None
        Results are saved to a CSV file in the output directory
    """
    # Calculate mutual information using the specified method
    result_df = mi_parallel_generator(
        dataframe, spaced_bin_edges=spaced_bin_edges, mi_methods=method, comb=comb
    )

    # Save results to CSV file
    output_file_path = os.path.join(output_dir, f"{method}.csv")
    result_df.to_csv(output_file_path)

    print(f"Completed processing for method: {method}")


def run_parallel_mi_calculations(
    dataframe: pd.DataFrame,
    methods: list,
    combinations: list,
    bin_edges: np.ndarray,
    output_dir: str,
    n_jobs: int = -1,
) -> None:
    """Run multiple mutual information calculations in parallel using joblib.

    This function orchestrates parallel processing of different mutual information
    calculations using joblib's Parallel and delayed functions.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data for MI calculations
    methods : list
        List of MI calculation methods to run
    combinations : list
        List of combination numbers corresponding to each method
    bin_edges : np.ndarray
        Pre-computed bin edges for discretization
    output_dir : str
        Directory for saving results
    n_jobs : int, optional
        Number of parallel jobs to run, defaults to using all CPU cores
    """
    Parallel(n_jobs=n_jobs)(
        delayed(process_mutual_information)(
            dataframe, method, comb, bin_edges, output_dir
        )
        for method, comb in zip(methods, combinations)
    )


if __name__ == "__main__":
    # Set up argument parser for organism specification
    parser = argparse.ArgumentParser(
        description="Calculate various mutual information metrics in parallel"
    )
    parser.add_argument(
        "-o", "--organism", type=str, required=True, help="The organism to analyze"
    )
    args = parser.parse_args()

    # Configure directories
    out_folder_name = "multivariate analysis "
    main_dir = ProjectPaths.get_project_root()
    cleaned_dir = ProjectPaths.get_cleaned_data_dir()
    output_file_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), out_folder_name, args.organism
    )
    output_img_dir = os.path.join(
        ProjectPaths.get_output_images_dir(), out_folder_name, args.organism
    )

    # Create output directories if they don't exist
    dir_handler.dir_maker(output_file_dir)
    dir_handler.dir_maker(output_img_dir)

    # Load input data
    input_file_path = os.path.join(
        cleaned_dir, args.organism, f"{args.organism}_zero_mean.csv"
    )

    histone_df = data_handler.csv_loader(input_file_path)

    # Load binning information
    mi_bin_info_dict = data_handler.json_file_loader(
        os.path.join(cleaned_dir, "mi_bin_info.json")
    )[args.organism]

    # Create bin edges
    bin_edges = create_bin_spacing(
        min_value=mi_bin_info_dict["global_min"],
        max_value=mi_bin_info_dict["global_max"],
        bin_number=mi_bin_info_dict["bin_num"],
    )
    print("Generated bin edges:", bin_edges)

    # Define methods and their corresponding combination numbers
    methods = [
        "mi_btw_u_and_y",
        "diff_uy_and_uy_givenv_entro",
    ]
    combinations = [2, 3]

    for method, comb in zip(methods, combinations):

        process_mutual_information(
            dataframe=histone_df,
            method=method,
            comb=comb,
            spaced_bin_edges=bin_edges,
            output_dir=output_file_dir,
        )

    # # Run parallel calculations
    # run_parallel_mi_calculations(
    #     dataframe=histone_df,
    #     methods=methods,
    #     combinations=combinations,
    #     bin_edges=bin_edges,
    #     output_dir=output_file_dir,
    # )
