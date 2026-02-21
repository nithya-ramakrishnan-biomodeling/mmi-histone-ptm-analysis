from multivariate_utils import (
    load_data,
    ProjectPaths,
    json_file_saver,
    dict_sort,
    MI_CONFIG,
    log_header,
    n_pos_zero_neg_triplet_selector,
    ORGANISMS,
)

if __name__ == "__main__":

    # example path: output\files\mmi_value_analysis\mmi_values_bin_based\human\histone_triplet_mmi_bin90.json
    total_features = 15
    k_feature = 5

    custom_bin_num = {
        # "yeast": 50,  # Set to an integer value to use a custom bin number, or None to use default from MI_CONFIG
        # "human": 50,  # Set to an integer value to use a custom bin number, or None to use default from MI_CONFIG
    }  # Set to an integer value to use a custom bin number, or None to use default from MI_CONFIG
    for organism in ORGANISMS:
        bin_num = custom_bin_num.get(organism, MI_CONFIG[organism]["bin_num"])

        triplet_mmi_file_path = ProjectPaths.get_output_files_dir().joinpath(
            "mmi_value_analysis",
            "mmi_values_bin_based",
            organism,
            f"histone_triplet_mmi_bin{bin_num}.json",
        )
        mmi_data_dict = load_data(triplet_mmi_file_path)
        mmi_data_dict = dict_sort(mmi_data_dict, by="value", reverse=True)

        log_header("Selecting negative, zero, positive MMI triplets for analysis")
        selected_triplet_dict = n_pos_zero_neg_triplet_selector(
            mmi_data_dict, n=total_features, k=k_feature
        )
        print(f"Selected triplets: {selected_triplet_dict}")

        # Save selected triplets
        save_path = ProjectPaths.get_output_files_dir().joinpath(
            "data_fit_analysis", "mmi_triplet_selection_for_analysis", organism
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path.joinpath(
            f"neg_zero_pos_triplet_selection_bin{bin_num}.json"
        )
        json_file_saver(selected_triplet_dict, save_path)
