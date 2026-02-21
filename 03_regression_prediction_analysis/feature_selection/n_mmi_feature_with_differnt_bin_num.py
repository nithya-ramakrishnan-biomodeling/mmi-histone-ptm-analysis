from multivariate_utils import (
    ORGANISMS,
    BIN_NUM_LIST,
    file_globl_max_min_config,
    load_data,
    json_file_saver,
    ProjectPaths,
    dir_maker,
    n_mmi_feature_selector,
)
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    for organism in ORGANISMS:
        print(f"Processing organism: {organism}")

        if organism == "yeast":
            top_n = 12
            file_path, global_min, global_max = file_globl_max_min_config(organism)
        if organism == "human":
            top_n = 10
            file_path, global_min, global_max = file_globl_max_min_config(organism)

        df = load_data(file_path)
        output_dir = Path.joinpath(
            ProjectPaths().get_output_files_dir(),
            "feature_selection",
            "mmi_features_with_different_bin_num",
        )
        dir_maker(output_dir.joinpath(organism))

        bin_num_feature_dict = {}
        for bin_num in BIN_NUM_LIST:

            edges = np.linspace(global_min, global_max, bin_num + 1)
            mmi_value_dir = Path.joinpath(
                ProjectPaths().get_output_files_dir(),
                "mmi_value_analysis",
                "mmi_values_bin_based",
            )
            mmi_value_path = mmi_value_dir.joinpath(
                organism, f"histone_triplet_mmi_bin{bin_num}.json"
            )
            mmi_value_dict = load_data(mmi_value_path)

            feature, triplet, triplet_mmi = n_mmi_feature_selector(
                n=top_n, mmi_data=mmi_value_dict, yeast_df=df, edges=edges
            )
            bin_num_feature_dict[bin_num] = {
                "feature": list(feature),
                "triplet": triplet,
                "mmi": triplet_mmi,
            }

        print(bin_num_feature_dict)
        # saving to json
        json_saver_path = output_dir.joinpath(
            organism, f"mmi_features_with_different_bin_num.json"
        )
        json_file_saver(bin_num_feature_dict, json_saver_path)
