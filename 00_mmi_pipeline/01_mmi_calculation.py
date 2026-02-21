"""
 Task: Calculate MMI for all histone triplets using different bin numbers

 output: JSON files containing MMI values for each histone triplet per bin number
"""
import numpy as np
from itertools import combinations
from functools import partial
import click
from pathlib import Path

from multivariate_utils import (
    diff_uy_and_uy_givenv,
    load_data,
    json_file_saver,
    ProjectPaths
)
from multivariate_utils import smart_map


def histone_processing(triplet, df, bin_num, global_min, global_max):
    feature_1, feature_2, feature_3 = triplet
    ar1, ar2, ar3 = df[[feature_1, feature_2, feature_3]].values.T
    edges = np.linspace(global_min, global_max, bin_num + 1)
    mmi = diff_uy_and_uy_givenv(ar1, ar2, ar3, edges)
    # del ar1, ar2, ar3
    # gc.collect()
    return "_".join(triplet), mmi

# arguments for the script: 
# input_are:  file path, global max and min values for binning, number of bins, output directory optional
@click.command()
@click.option("--file-path", "-f", required=True, help="Path to the input data file (CSV format).")
@click.option("--n-bins", "-b", required=True, type=int, help="Number of bins for MMI calculation.")
@click.option("--global-min", "-min", required=False, type=float, help="Global minimum value for binning.")
@click.option("--global-max", "-max", required=False, type=float, help="Global maximum value for binning.")
@click.option("--output-path", "-o", default=None, help="Path to save the output JSON file.")
def main(file_path, global_min, global_max, n_bins, output_path):
    click.echo(f"Starting MMI calculation for file: {file_path}")
    click.echo(f"Global min: {global_min}, Global max: {global_max}, Number of bins: {n_bins}")
    click.echo(f"Output path: {output_path}")


    # ========================================================
    # set file path and global min/max based on organism
    # ========================================================
    file_path = Path(file_path)
    n_bins = n_bins
    if output_path is None:
        output_path = ProjectPaths.get_output_files_dir("mmi_results", "mmi_results.json")
    else:
        output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ========================================================
    # load data and prepare triplets
    # ========================================================
    histone_df = load_data(file_path)
    global_min = global_min if global_min is not None else histone_df.min().min()
    global_max = global_max if global_max is not None else histone_df.max().max()
    histone_features = histone_df.columns.tolist()
    histone_triplets = list(combinations(histone_features, 3)) # O(N^3) triplets

    # =========================================================
    # calculate MMI for each triplet in parallel
    # =========================================================
    func = partial(
        histone_processing,
        df=histone_df,
        bin_num=n_bins,
        global_min=global_min,
        global_max=global_max,
    )
    result = smart_map(func, histone_triplets[:], show_progress=True)
    triplet_mmi_dict = {triplet: mmi for triplet, mmi in result}
    sorted_triplet_mmi_dict = dict(
        sorted(triplet_mmi_dict.items(), key=lambda item: item[1])
    )

    # =========================================================
    # save output json file
    # =========================================================
    json_file_saver(sorted_triplet_mmi_dict, output_path)

if __name__ == "__main__":
    main()
