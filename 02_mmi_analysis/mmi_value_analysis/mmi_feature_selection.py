"""
 Task: MMI feature selection using different mmi thresholds and N-best features
 output: JSON files containing selected features, MMI triplet, and MMI threshold per bin number

"""

import os
from multivariate_utils.file_path_config import MI_CONFIG
import numpy as np
from multivariate_utils import (
    ProjectPaths,
    load_data,
    json_file_saver,
    dir_maker,
    n_mmi_feature_selector,
)
from multivariate_utils import YEAST_FILE_PATH, HUMAN_FILE_PATH, MI_CONFIG

bin_list = list(range(40, 151, 10))
organism_list = ["yeast", "human"]
mmi_file_dir = os.path.join(
    ProjectPaths().get_output_files_dir(), "mmi_value_analysis", "mmi_values_bin_based"
)

for organism in organism_list[:]:
    if organism == "yeast":
        file_path = YEAST_FILE_PATH
        global_min = MI_CONFIG["yeast"]["global_min"]
        global_max = MI_CONFIG["yeast"]["global_max"]
        top_n = 12

    if organism == "human":
        file_path = HUMAN_FILE_PATH
        global_min = MI_CONFIG["human"]["global_min"]
        global_max = MI_CONFIG["human"]["global_max"]
        top_n = 10

    histone_df = load_data(file_path)
    bin_feature_mmi_dict = {}
    for bin_num in bin_list:

        file_path = os.path.join(
            mmi_file_dir, organism, f"histone_triplet_mmi_bin{bin_num}.json"
        )
        mmi_data = load_data(file_path=file_path)
        edges = np.linspace(global_min, global_max, bin_num + 1)
        features, mmi_triplet, mmi_threshold = n_mmi_feature_selector(
            n=top_n, mmi_data=mmi_data, yeast_df=histone_df, edges=edges
        )

        print(
            f"Bin num: {bin_num}, Selected features: {features}, MMI triplet: {mmi_triplet}, MMI threshold: {mmi_threshold}"
        )
        # break

        bin_feature_mmi_dict[bin_num] = {
            "selected_features": list(features),
            "mmi_triplet": mmi_triplet,
            "mmi_threshold": mmi_threshold,
        }

    output_dir = os.path.join(
        ProjectPaths().get_output_files_dir(),
        "mmi_value_analysis",
        "mmi_feature_n_selection",
        organism,
    )
    dir_maker(output_dir)
    output_file = os.path.join(
        output_dir, f"{organism}_top{top_n}_features_mmi_selection.json"
    )
    json_file_saver(bin_feature_mmi_dict, output_file)
