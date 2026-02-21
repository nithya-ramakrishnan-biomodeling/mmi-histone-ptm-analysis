from multivariate_utils import (
    total_correlation,
    load_data,
    MI_CONFIG,
    ORGANISMS,
    ProjectPaths,
    smart_map,
    estimate_optimal_workers,
    file_globl_max_min_config,
)

from functools import partial
import numpy as np
import pandas as pd


def tc_calculation_for_triplet(triplet, dataframe, edges):
    """Calculate the total correlation for a given triplet of histone modifications."""

    histone1, histone2, histone3 = triplet.split("_")
    tc_value = total_correlation(
        arr1=dataframe[histone1].values,
        arr2=dataframe[histone2].values,
        arr3=dataframe[histone3].values,
        spaced_bin_edges=edges,
    )
    return triplet, tc_value


if __name__ == "__main__":

    for organism in ORGANISMS[1:]:  # 'yeast' or 'human'
        print(f"Processing organism: {organism}")

        file_path, global_min, global_max = file_globl_max_min_config(organism)
        bin_num = MI_CONFIG[organism]["bin_num"]
        
        if organism == "yeast":
            bin_num = 70
        elif organism == "human":
            bin_num = 100
        df = load_data(file_path=file_path)
        histone_mods = df.columns.tolist()

        # path: output\files\mmi_value_analysis\mmi_values_bin_based\human\histone_triplet_mmi_bin130.json
        mmi_combinations_path = (
            ProjectPaths()
            .get_output_files_dir()
            .joinpath(
                "mmi_value_analysis",
                "mmi_values_bin_based",
                organism,
                f"histone_triplet_mmi_bin{bin_num}.json",
            )
        )

        mmi_triplets = load_data(mmi_combinations_path)
        mmi_triplet_list = mmi_triplets.keys()

        edges = np.linspace(global_min, global_max, bin_num + 1)
        tc_partial = partial(tc_calculation_for_triplet, dataframe=df, edges=edges)
        optimal_workers = estimate_optimal_workers()
        results = smart_map(
            tc_partial,
            items=mmi_triplet_list,
            max_workers=optimal_workers,
            show_progress=True,
        )
        print(results[:10])

        # dictionary to dataframe, columns: triplet, tc_value, mmi_value, dictionary comprehension
        tc_mmi_dict = {
            triplet: {"tc_value": tc_value, "mmi_value": mmi_triplets[triplet]}
            for triplet, tc_value in results
        }
        output_path = (
            ProjectPaths()
            .get_output_files_dir()
            .joinpath(
                "total_correlation_analysis",
                f"tc_values_bin_based",
                organism,
                f"histone_triplet_tc_bin{bin_num}.csv",
            )
        )

        output_path_dir = output_path.parent
        output_path_dir.mkdir(parents=True, exist_ok=True)

        tc_mmi_df = pd.DataFrame.from_dict(tc_mmi_dict, orient="index")
        tc_mmi_df.to_csv(output_path)
