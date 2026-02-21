from multivariate_utils import (
    ProjectPaths,
    load_data,
    mi_betwn_uandy,
    json_file_saver,
    MI_CONFIG,
    ORGANISMS,
    file_globl_max_min_config,
    Histone_Regressor,
    log_header,
)

import numpy as np

if __name__ == "__main__":

    for organism in ["human"]:
        log_header(f"Processing organism: {organism}")
        if organism == "yeast":
            file_path, global_min, global_max = file_globl_max_min_config(organism)
            bin_num = MI_CONFIG[organism]["bin_num"]
            # threshold_list = [-0.45, -0.43, -0.41]
            threshold_list = [-0.3, -0.29, -0.285, -0.283]
        if organism == "human":
            file_path, global_min, global_max = file_globl_max_min_config(organism)
            bin_num = MI_CONFIG[organism]["bin_num"]
            threshold_list = [-0.5, -0.45, -0.4]

        df = load_data(file_path)
        histone_mods = df.columns.tolist()

        # feature_file_path = ProjectPaths.get_output_files_dir().joinpath(
        #     f"feature_selection",
        #     "mmi_features_thresholds",
        #     organism,
        #     f"mmi_features_thresholds_bin_{bin_num}.json",
        # )
        # path : output\files\regression_prediction_analysis\mmi_r2_100_bin\human\mmi_r2_100_scores.json
        feature_file = ProjectPaths.get_output_files_dir().joinpath(
            "feature_selection",
            "mmi_features_thresholds",
            organism,
            f"mmi_features_thresholds_bin_{bin_num}.json",
        )

        mmi_features_thresholds = load_data(feature_file)

        threshold_mi_yhat_y = {}
        for threshold in threshold_list:
            log_header(f"Processing threshold: {threshold}")
            selected_features = mmi_features_thresholds.get(str(threshold), [])
            target_histone_mods = [
                md for md in histone_mods if md not in selected_features
            ]

            target_mi_yhat_y = {}
            for target in target_histone_mods:

                X = df[selected_features].values
                y = df[target].values

                regressor = Histone_Regressor(feature=X, target=y, model_n_jobs=-3)
                y_hat, y = regressor.xgb_regressor_pred()

                edges = np.linspace(global_min, global_max, bin_num + 1)
                mi_yhat_y = mi_betwn_uandy(y_hat, y, spaced_bin_edges=edges)

                target_mi_yhat_y[target] = mi_yhat_y
                print(f"Computed MI between {target} y_hat and y: {mi_yhat_y}")

            sorted_target_mi_yhat_y = dict(
                sorted(target_mi_yhat_y.items(), key=lambda item: item[1], reverse=True)
            )
            threshold_mi_yhat_y[threshold] = sorted_target_mi_yhat_y
            # print(y_hat.shape, y.shape)
            # print(y_hat[:5], y[:5])
            # break

        # saving the results
        sav_path = ProjectPaths.get_output_files_dir().joinpath(
            f"mi_btw_mmi_thresholds_yhat_and_y", organism
        )
        sav_path.mkdir(parents=True, exist_ok=True)
        sav_file = sav_path.joinpath(f"mi_btw_mmi_thresholds_yhat_and_y_bin_{bin_num}.json")
        json_file_saver(threshold_mi_yhat_y, sav_file)
