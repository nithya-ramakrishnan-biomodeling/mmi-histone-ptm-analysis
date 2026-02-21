from multivariate_utils import (
    ORGANISMS,
    MI_CONFIG,
    load_data,
    json_file_saver,
    ProjectPaths,
    dir_maker,
    mmi_feature_selector,
)
from pathlib import Path


if __name__ == "__main__":
    for organism in ORGANISMS:
        print(f"Processing organism: {organism}")
        output_dir = Path.joinpath(
            ProjectPaths().get_output_files_dir(),
            "feature_selection",
            "mmi_features_thresholds",
        )
        dir_maker(output_dir.joinpath(organism))

        # organism = "yeast"
        # organism = "human"
        if organism == "yeast":
            threshold_list = [-0.45, -0.43, -0.41]

        if organism == "human":
            threshold_list = [-0.9, -0.7, -0.6]

        bin_num = MI_CONFIG[organism]["bin_num"]
        mmi_value_dir = Path.joinpath(
            ProjectPaths().get_output_files_dir(),
            "mmi_value_analysis",
            "mmi_values_bin_based",
        )
        mmi_value_path = mmi_value_dir.joinpath(
            organism, f"histone_triplet_mmi_bin{bin_num}.json"
        )
        mmi_value_dict = load_data(mmi_value_path)

        # for positive selection, use mode="ge" and threshold as positive value
        # for negative selection, use mode="le" and threshold as negative value
        # not equal to  is not supported currently
        threshold_feature_dict = {}
        for threshold in threshold_list:
            features, _ = mmi_feature_selector(
                threshold=threshold, mmi_data=mmi_value_dict, mode="le"
            )
            print(f"{threshold}: {set(features)}")

            threshold_feature_dict[threshold] = list(set(features))

        # saving to json
        json_saver_path = output_dir.joinpath(
            organism, f"mmi_features_thresholds_bin_{bin_num}.json"
        )
        json_file_saver(threshold_feature_dict, json_saver_path)
