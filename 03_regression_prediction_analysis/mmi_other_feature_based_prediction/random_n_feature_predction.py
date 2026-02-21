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
from datahandler import timer
from random import sample, shuffle
from functools import partial

# setting random seed for reproducibility
from multivariate_utils.file_path_config import HUMAN_FILE_PATH
import numpy as np

np.random.seed(42)


def random_n_feature_selection(all_features, n_features, n_repeats=10):

    for _ in range(n_repeats):
        shuffle(all_features)
    selected_features = sample(all_features, n_features)
    return selected_features


def target_histone_predictor(target_arr, feature_arr):
    model = Histone_Regressor(feature=feature_arr, target=target_arr)
    r2_score = model.get_cv_results()["test_score"]
    return r2_score


if __name__ == "__main__":

    for organism in ORGANISMS:
        log_header(f"Processing organism: {organism}")

        if organism == "yeast":
            file_path = YEAST_FILE_PATH

        elif organism == "human":
            file_path = HUMAN_FILE_PATH
        else:
            raise ValueError(
                f"Organism '{organism}' not recognized. Choose 'yeast' or 'human'."
            )

        df = load_data(file_path)
        all_features = df.columns.tolist()

        # =======================================================
        # random n feature based target histone modification prediction
        # =======================================================
        log_header(
            f"Random n feature based target histone modification prediction for organism: {organism}"
        )
        n_features_to_select = 10
        n_workers = estimate_optimal_workers()
        feature_target_r2_dict = {}
        selected_features = random_n_feature_selection(
            all_features, n_features_to_select
        )
        target_features = [
            feat for feat in all_features if feat not in selected_features
        ][:]

        feature_target_r2_dict["selected_features"] = selected_features
        feature_arr = df[selected_features].values
        print(feature_arr.shape)
        partial_func = partial(target_histone_predictor, feature_arr=feature_arr)
        results = smart_map(
            partial_func,
            [df[tar_feat].values for tar_feat in target_features[:]],
            max_workers=n_workers,
            mode="sequential",
            show_progress=True,
        )

        target_predictor_r2_dict = dict(zip(target_features[:], results))
        sorted_target_predictor_r2_dict = dict(
            sorted(
                target_predictor_r2_dict.items(), key=lambda item: item[1], reverse=True
            )
        )

        feature_target_r2_dict["target_predictor_r2_scores"] = (
            sorted_target_predictor_r2_dict
        )
        # saving results
        output_dir = ProjectPaths.get_output_files_dir().joinpath(
            "regression_prediction_analysis",
            f"random_{n_features_to_select}_feature_prediction",
            organism,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sav_file = output_dir.joinpath(
            f"random_{n_features_to_select}_feature_target_predictor_r2_scores.json"
        )
        json_file_saver(feature_target_r2_dict, sav_file)

        # =======================================================
        # single random feature prediction for testing
        # =======================================================
        log_header(
            f"Single random feature based target histone modification prediction for organism: {organism}"
        )
        n_features_to_select = 1
        single_feature_target_dict = {}
        single_selected_features = random_n_feature_selection(
            all_features, n_features_to_select
        )
        single_target_features = [
            feat for feat in all_features if feat not in single_selected_features
        ]

        single_feature_target_dict["selected_features"] = single_selected_features
        single_feature_arr = df[single_selected_features].values
        print(single_feature_arr.shape)
        partial_func = partial(target_histone_predictor, feature_arr=single_feature_arr)
        single_results = smart_map(
            partial_func,
            [df[tar_feat].values for tar_feat in single_target_features[:]],
            max_workers=n_workers,
            mode="auto",
            show_progress=True,
        )
        single_target_predictor_r2_dict = dict(
            zip(single_target_features[:], single_results)
        )
        sorted_single_target_predictor_r2_dict = dict(
            sorted(
                single_target_predictor_r2_dict.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        single_feature_target_dict["target_predictor_r2_scores"] = (
            sorted_single_target_predictor_r2_dict
        )
        # saving results
        output_dir = ProjectPaths.get_output_files_dir().joinpath(
            "regression_prediction_analysis",
            f"random_{n_features_to_select}_feature_prediction",
            organism,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sav_file = output_dir.joinpath(
            f"random_{n_features_to_select}_feature_target_predictor_r2_scores.json"
        )
        json_file_saver(single_feature_target_dict, sav_file)
