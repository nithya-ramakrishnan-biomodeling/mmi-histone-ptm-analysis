from multivariate_utils import (
    ORGANISMS,
    YEAST_FILE_PATH,
    HUMAN_FILE_PATH,
    MI_CONFIG,
    load_data,
    json_file_saver,
    ProjectPaths,
    smart_map,
    dir_maker,
)

from functools import partial
from multivariate_utils.parallel_worker.convienence_methods import (
    estimate_optimal_workers,
)


def highest_correlated_feature(target_column, mmi_features, df):
    """getting highest correlated feature for each target histones"""
    correlations = df.corr()[target_column].drop(target_column)

    # drop mmi_feature columns
    for mmi_feature in mmi_features:
        if mmi_feature in correlations.index:
            correlations = correlations.drop(mmi_feature)

    highest_corr_feature = correlations.abs().idxmax()
    highest_corr_value = correlations[highest_corr_feature]
    return [highest_corr_feature, highest_corr_value]


if __name__ == "__main__":

    for organism in ORGANISMS[:]:
        print(f"Processing organism: {organism}")
        threshold_feature_dict = {}
        # "output\files\feature_selection\mmi_features_thresholds\human\mmi_features_thresholds_bin_130.json"
        # output\files\feature_selection\mmi_features_thresholds\yeast\mmi_features_thresholds_bin_70.json
        if organism == "yeast":

            bin_num = 70
            mmi_feature_file_path = ProjectPaths.get_output_files_dir().joinpath(
                "feature_selection",
                "mmi_features_thresholds",
                organism,
                f"mmi_features_thresholds_bin_{bin_num}.json",

        )
        # path : output\files\feature_selection\mmi_features_thresholds\human\mmi_features_thresholds_bin_100.json
        if organism == "human":

            bin_num = 100
            mmi_feature_file_path = ProjectPaths.get_output_files_dir().joinpath(
                "feature_selection",
                "mmi_features_thresholds",
                organism,
                f"mmi_features_thresholds_bin_{bin_num}.json", )
        # mmi_feature_file_path = ProjectPaths.get_output_files_dir().joinpath(
        #     "feature_selection",
        #     "mmi_features_thresholds",
        #     organism,
        #     f"mmi_features_thresholds_bin_{MI_CONFIG[organism]['bin_num']}.json",
        # )
        mmi_feature_dict = load_data(mmi_feature_file_path)

        if organism == "yeast":
            threshold_list = [-0.3, -0.29, -0.285, -0.283]
            # threshold_list = [-0.45, -0.43, -0.41]
            file_path = YEAST_FILE_PATH

        if organism == "human":
            # threshold_list = [-0.9, -0.7, -0.6] # for bin number 130
            threshold_list = [-0.5, -0.45, -0.4] # for bin number 100
            file_path = HUMAN_FILE_PATH

        df = load_data(file_path=file_path)
        histone_mods = df.columns
        for threshold in threshold_list:
            print(f"Loading MMI features for threshold: {threshold}")

            mmi_features = mmi_feature_dict[str(threshold)]
            results = smart_map(
                partial(highest_correlated_feature, df=df, mmi_features=mmi_features),
                histone_mods,
                show_progress=True,
                max_workers=estimate_optimal_workers(),
            )

            threshold_feature_dict[threshold] = dict(zip(histone_mods, results))

        # saving results
        print(threshold_feature_dict)

        # saving to json
        json_saver_path = ProjectPaths.get_output_files_dir().joinpath(
            "feature_selection", "mmi_and_highest_correlation_feature", organism
        )
        dir_maker(json_saver_path)
        json_saver_path = json_saver_path.joinpath(
            f"mmi_and_highest_correlation_features_bin_{bin_num}.json"
        )
        json_file_saver(threshold_feature_dict, json_saver_path)
