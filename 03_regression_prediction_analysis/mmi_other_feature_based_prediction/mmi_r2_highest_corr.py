from multivariate_utils import (
    ProjectPaths,
    Histone_Regressor,
    load_data,
    json_file_saver,
    smart_map,
    estimate_optimal_workers,
    YEAST_FILE_PATH,
    ORGANISMS,
    log_header,
)
import gc

# setting random seed for reproducibility
from multivariate_utils.file_path_config import HUMAN_FILE_PATH
import numpy as np

np.random.seed(42)


if __name__ == "__main__":

    for organism in ["human"]:
        log_header(f"Processing organism: {organism}")

        if organism == "yeast":
            file_path = YEAST_FILE_PATH
            n_jobs = -3

        elif organism == "human":
            file_path = HUMAN_FILE_PATH
            n_jobs = 3
        else:
            raise ValueError(
                f"Organism '{organism}' not recognized. Choose 'yeast' or 'human'."
            )

        df = load_data(file_path)
        all_features = df.columns.tolist()

        # =======================================================
        # mmi threshold based target histone modification prediction
        # =======================================================

        # file path for loading feature for each target histone modification
        # examples: output\files\feature_selection\mmi_features_thresholds\yeast\mmi_features_thresholds_bin_70.json

        # mmi _feature_file path 

        if organism == "yeast":
            bin_num = 70
        elif organism == "human":
            bin_num = 100
        feature_file = ProjectPaths.get_output_files_dir().joinpath(
            "feature_selection",
            "mmi_features_thresholds",
            organism,
            f"mmi_features_thresholds_bin_{bin_num}.json",
        )
        # highest correlated feature file path
        # output\files\feature_selection\mmi_and_highest_correlation_feature\yeast\mmi_and_highest_correlation_features_bin_70.json
        highest_corr_feature_file = ProjectPaths.get_output_files_dir().joinpath(
            "feature_selection",
            "mmi_and_highest_correlation_feature",
            organism,
            f"mmi_and_highest_correlation_features_bin_{bin_num}.json",
        )

        feature_selection_results = load_data(feature_file)
        highest_corr_feature_results = load_data(highest_corr_feature_file)
        threshold_list = list(feature_selection_results.keys())

        mmi_threshold_r2_dict = {}
        for threshold in threshold_list:
            features = feature_selection_results[threshold]
            targets = [target for target in all_features if target not in features]

            mmi_r2_dict = {}
            for target in targets:

                high_correl_feature = highest_corr_feature_results[threshold][target][0]
                print(f"Predicting target: {target} using {features} features, highest corr: {high_correl_feature}")

                feature_cp = features.copy()
                feature_cp.append(high_correl_feature)
                # use zero-copy numpy views (where possible)
                feature_arr = df.loc[:, feature_cp].to_numpy()
                target_arr = df[target].to_numpy()

                # instantiate regressor with constrained parallelism
                regressor = Histone_Regressor(
                    feature=feature_arr,
                    target=target_arr,
                    model_n_jobs=n_jobs,
                    cv_n_jobs=n_jobs,
                )
                file_path = ProjectPaths.get_output_files_dir().joinpath(
                    "regression_prediction_analysis",
                    f"mmi_r2_{bin_num}_bin",
                    organism,
                    f"mmi_threshold_{threshold}_regression_highest_corr_results.csv",
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                regressor.save_cv_results_to_csv(
                    file_path, append=True, feature_name=feature_cp, target_name=target
                )

                r2_score = regressor.get_cv_results()["test_score"]
                mmi_r2_dict[target] = r2_score

                # cleanup explicitly
                regressor.cleanup()
                del regressor, feature_arr, target_arr
                import gc

                gc.collect()

            # sorting the dictionary based on  value (r2 score)
            mmi_r2_dict = dict(
                sorted(
                    mmi_r2_dict.items(), key=lambda item: np.mean(item[1]), reverse=True
                )
            )

            mmi_threshold_r2_dict[threshold] = mmi_r2_dict
        # saving results
        output_dir = ProjectPaths.get_output_files_dir().joinpath(
            "regression_prediction_analysis", f"mmi_r2_{bin_num}_bin", organism
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir.joinpath(f"mmi_highest_corr_r2_{bin_num}_scores.json")
        json_file_saver(mmi_threshold_r2_dict, output_file)
