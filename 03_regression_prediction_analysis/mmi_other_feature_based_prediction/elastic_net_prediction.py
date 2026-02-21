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

    for organism in ORGANISMS[:]:
        log_header(f"Processing organism: {organism}")

        if organism == "yeast":
            file_path = YEAST_FILE_PATH
            n_jobs = -2

        elif organism == "human":
            file_path = HUMAN_FILE_PATH
            n_jobs = 1
        else:
            raise ValueError(
                f"Organism '{organism}' not recognized. Choose 'yeast' or 'human'."
            )

        df = load_data(file_path)
        all_features = df.columns.tolist()

        # =======================================================
        # elastic net based target histone modification prediction
        # =======================================================

        # file path for loading feature for each target histone modification
        # examples: output\files\feature_selection\elastic_net\human\elastic_target_feature_results.json

        feature_file = ProjectPaths.get_output_files_dir().joinpath(
            "feature_selection",
            "elastic_net",
            organism,
            "elastic_target_feature_results.json",
        )

        feature_selection_results = load_data(feature_file)

        elastic_dict = {}
        for target in all_features:

            features = feature_selection_results[target]
            # use zero-copy numpy views (where possible)
            feature_arr = df.loc[:, features].to_numpy()
            target_arr = df[target].to_numpy()
            print(f"Predicting target: {target} using {features} features.")

            # instantiate regressor with constrained parallelism
            regressor = Histone_Regressor(
                feature=feature_arr,
                target=target_arr,
                model_n_jobs=n_jobs,
                cv_n_jobs=n_jobs,
            )
            file_path = ProjectPaths.get_output_files_dir().joinpath(
                "regression_prediction_analysis",
                "elastic_net_prediction",
                organism,
                f"elastic_net_regression_results.csv",
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            regressor.save_cv_results_to_csv(
                file_path, append=True, feature_name=features, target_name=target
            )

            r2_score = regressor.get_cv_results()["test_score"]
            elastic_dict[target] = r2_score

            # cleanup explicitly
            regressor.cleanup()
            del regressor, feature_arr, target_arr
            import gc

            gc.collect()

        # sorting the dictionary based on  value (r2 score)
        elastic_dict = dict(
            sorted(
                elastic_dict.items(), key=lambda item: np.mean(item[1]), reverse=True
            )
        )
        # saving results
        output_dir = ProjectPaths.get_output_files_dir().joinpath(
            "regression_prediction_analysis", "elastic_net_prediction", organism
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir.joinpath("elastic_net_r2_scores.json")
        json_file_saver(elastic_dict, output_file)
