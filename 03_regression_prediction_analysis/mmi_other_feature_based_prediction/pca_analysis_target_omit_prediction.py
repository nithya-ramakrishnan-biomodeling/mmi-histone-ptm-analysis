""" idea: 
load data
from total histone mods select a target histone mod
omit it from the predictor set
perform pca on the predictor set
use the principal components to predict the target histone mod
save the results 
"""

import numpy as np
from multivariate_utils import (
    pca_analyzer,
    file_globl_max_min_config,
    load_data,
    json_file_saver,
    ProjectPaths,
    log_header,
    MI_CONFIG,
    ORGANISMS,
    Histone_Regressor,
)

import pandas as pd

import random

np.random.seed(42)
random.seed(42)


if __name__ == "__main__":

    num_pcs = 5
    methods = ["cov", "mi"]

    for organism in ORGANISMS:
        log_header(f"Processing organism: {organism}")
        if organism == "yeast":
            file_path, global_min, global_max = file_globl_max_min_config(organism)
            bin_num = MI_CONFIG[organism]["bin_num"]
            bin_edges = np.linspace(global_min, global_max, bin_num + 1)
        if organism == "human":
            file_path, global_min, global_max = file_globl_max_min_config(organism)
            bin_num = MI_CONFIG[organism]["bin_num"]
            bin_edges = np.linspace(global_min, global_max, bin_num + 1)

        df = load_data(file_path)
        df.index = df.index.astype(str)
        all_histone_mods = df.columns.tolist()

        # debug + consistency checks
        print("data path:", file_path)
        print("data.shape:", df.shape, "num_na:", int(df.isna().sum().sum()))
        all_histone_mods = df.columns.tolist()
        # enforce stable ordering for predictors
        all_histone_mods = sorted(all_histone_mods)
        print("histone mods (first 10):", all_histone_mods[:10])

        target_pca_prediction_results = {}
        method_results = {}
        for method in methods:
            log_header(f"PCA omission prediction using method: {method}")

            target_dict = {}
            for target_mod in all_histone_mods:
                # log_header(f"{target_mod} using PCA components")

                # Define predictor mods by omitting the target mod
                predictor_mods = [mod for mod in all_histone_mods if mod != target_mod]

                # Perform PCA on the predictor mods
                pca = pca_analyzer.PCAanalyzer(
                    data=df[predictor_mods],
                    method=method,
                    n_components=num_pcs,
                    bin_edges=bin_edges,
                )

                # fit and transform data
                # transformed_df = pd.DataFrame(data=pca.fit_transform(),
                #                                 index=df.index,
                #                                 columns=[f"v{i+1}" for i in range(num_pcs)])

                transformed_arr = pca.fit_transform()

                # print(transformed_df.head())
                # Predict the target histone modification using the principal components
                target_df = df[[target_mod]]
                predictor = Histone_Regressor(feature=transformed_arr, target=target_df)

                # score = predictor.get_cv_results()
                # saving results
                feature_name = "_".join([f"v{i+1}" for i in range(num_pcs)])
                file_path = ProjectPaths.get_output_files_dir().joinpath(
                    "regression_prediction_analysis",
                    "pca_omit_target_prediction",
                    organism,
                    method,
                    f"pca_cv_results_{num_pcs}.csv",
                )

                sub_df = predictor.save_cv_results_to_csv(
                    file_path=file_path,
                    methods="r2",
                    feature_name=feature_name,
                    target_name=target_mod,
                    append=True,
                )

                test_r2 = predictor.get_cv_results()["test_score"]
                print(f"Target: {target_mod}, R2: {test_r2}")
                target_dict[target_mod] = test_r2
                del transformed_arr, target_df, predictor

            # sorting the target dict by r2 scores
            sorted_target_dict = dict(
                sorted(target_dict.items(), key=lambda item: item[1], reverse=True)
            )
            method_results[method] = sorted_target_dict

        # saving the results
        output_dir = ProjectPaths.get_output_files_dir().joinpath(
            "regression_prediction_analysis", "pca_omit_target_prediction", organism
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sav_file = output_dir.joinpath(f"pca_omit_target_prediction_r2_scores_{num_pcs}.json")
        json_file_saver(method_results, sav_file)
