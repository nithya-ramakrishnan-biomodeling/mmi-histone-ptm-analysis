from multivariate_utils import (
    ProjectPaths,
    load_data,
    mi_betwn_uandy,
    json_file_saver,
    MI_CONFIG,
    ORGANISMS,
    YEAST_FILE_PATH,
    HUMAN_FILE_PATH,
    file_globl_max_min_config,
    Histone_Regressor,
    log_header,
)

import numpy as np

if __name__ == "__main__":

    for organism in ORGANISMS:
        log_header(f"Processing organism: {organism}")
        if organism == "yeast":
            file_path, global_min, global_max = file_globl_max_min_config(organism)
            bin_num = MI_CONFIG[organism]["bin_num"]
        if organism == "human":
            file_path, global_min, global_max = file_globl_max_min_config(organism)
            bin_num = MI_CONFIG[organism]["bin_num"]

        df = load_data(file_path)
        histone_mods = df.columns.tolist()
        target_mi_yhat_y = {}
        for target in histone_mods:

            features = [md for md in histone_mods if md != target]
            X = df[features].values
            y = df[target].values

            regressor = Histone_Regressor(feature=X, target=y)
            y_hat, y = regressor.xgb_regressor_pred()

            edges = np.linspace(global_min, global_max, bin_num + 1)
            mi_yhat_y = mi_betwn_uandy(y_hat, y, spaced_bin_edges=edges)

            target_mi_yhat_y[target] = mi_yhat_y
            print(f"Computed MI between {target} y_hat and y: {mi_yhat_y}")

            # print(y_hat.shape, y.shape)
            # print(y_hat[:5], y[:5])
            # break

        # saving the results
        # sort the dict by values:
        sorted_target_mi_yhat_y = dict(
            sorted(target_mi_yhat_y.items(), key=lambda item: item[1], reverse=True)
        )
        sav_path = ProjectPaths.get_output_files_dir().joinpath(
            f"mi_btw_all_remain_yhat_and_y", organism
        )
        sav_path.mkdir(parents=True, exist_ok=True)
        sav_file = sav_path.joinpath(f"mi_btw_all_remain_yhat_and_y.json")
        json_file_saver(sorted_target_mi_yhat_y, sav_file)
