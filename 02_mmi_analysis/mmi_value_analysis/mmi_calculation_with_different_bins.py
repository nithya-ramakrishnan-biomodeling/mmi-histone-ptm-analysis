"""
 Task: Calculate MMI for all histone triplets using different bin numbers

 output: JSON files containing MMI values for each histone triplet per bin number
"""

import os
import numpy as np
from itertools import combinations
from functools import partial

from multivariate_utils import (
    ProjectPaths,
    diff_uy_and_uy_givenv,
    load_data,
    json_file_saver,
    dir_maker,
)
from multivariate_utils import YEAST_FILE_PATH, HUMAN_FILE_PATH, MI_CONFIG
from multivariate_utils import smart_map
from  multivariate_utils import ORGANISMS, BIN_NUM_LIST, file_globl_max_min_config

output_dir = os.path.join(
    ProjectPaths().get_output_files_dir(), "mmi_value_analysis", "mmi_values_bin_based"
)


def histone_processing(triplet, df, bin_num, global_min, global_max):
    feature_1, feature_2, feature_3 = triplet
    ar1, ar2, ar3 = df[[feature_1, feature_2, feature_3]].values.T
    edges = np.linspace(global_min, global_max, bin_num + 1)
    mmi = diff_uy_and_uy_givenv(ar1, ar2, ar3, edges)
    # del ar1, ar2, ar3
    # gc.collect()
    return "_".join(triplet), mmi


if __name__ == "__main__":

    organism_list = ORGANISMS
    bin_n_list = BIN_NUM_LIST

    # organism = "yeast"  # "human" or "yeast"
    for organism in organism_list:

        # ========================================================
        # set file path and global min/max based on organism
        # ========================================================
        file_path, global_min, global_max = file_globl_max_min_config(organism)
        # ========================================================
        # load data and prepare triplets
        # ========================================================
        histone_df = load_data(file_path, index_col=0)
        histone_features = histone_df.columns.tolist()
        histone_triplets = list(combinations(histone_features, 3))

        bin_summary = {}
        for n_bin in bin_n_list[:]:
            bin_dict = {}
            bin_dict["bin_num"] = n_bin

            # =========================================================
            # calculate MMI for each triplet in parallel
            # =========================================================
            func = partial(
                histone_processing,
                df=histone_df,
                bin_num=n_bin,
                global_min=global_min,
                global_max=global_max,
            )
            result = smart_map(func, histone_triplets[:])
            # print(result[:10])
            # dictionary to store MMI output is list of tuples
            triplet_mmi_dict = {triplet: mmi for triplet, mmi in result}
            # sorted dict the triplet dict by MMI values in were large negative to positive
            sorted_triplet_mmi_dict = dict(
                sorted(triplet_mmi_dict.items(), key=lambda item: item[1])
            )
            # print(f"Sorted triplet MMI dict for bin {n_bin}: {list(sorted_triplet_mmi_dict.items())[:15]}")
            # print(sorted_triplet_mmi_dict)
            # break

            # =========================================================
            # save output json file
            # =========================================================
            output_file = os.path.join(
                output_dir, organism, f"histone_triplet_mmi_bin{n_bin}.json"
            )
            dir_maker(os.path.dirname(output_file))
            json_file_saver(sorted_triplet_mmi_dict, output_file)
