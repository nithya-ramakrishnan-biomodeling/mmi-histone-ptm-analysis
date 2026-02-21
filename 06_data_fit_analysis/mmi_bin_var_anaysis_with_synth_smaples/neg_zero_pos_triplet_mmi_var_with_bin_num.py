from multivariate_utils import (
    ProjectPaths,
    load_data,
    log_header,
    ORGANISMS,
    MI_CONFIG,
    diff_uy_and_uy_givenv,
    json_file_saver,
)

import numpy as np


if __name__ == "__main__":

    # samples size is :
    sample_size = 250 * 1e4
    bin_list = list(range(40, 301, 10))

    # ========================================================
    # For each organism
    # ========================================================
    for organism in ORGANISMS:
        log_header(f"Processing organism: {organism}")

        if organism == "human":
            mmi_bin_num = MI_CONFIG[organism]["bin_num"]
            global_max = MI_CONFIG[organism]["global_max"]
            global_min = MI_CONFIG[organism]["global_min"]

        elif organism == "yeast":
            mmi_bin_num = MI_CONFIG[organism]["bin_num"]
            global_max = MI_CONFIG[organism]["global_max"]
            global_min = MI_CONFIG[organism]["global_min"]

        # example_path : output\files\data_fit_analysis\mmi_triplet_selection_for_analysis\human\neg_zero_pos_triplet_selection_bin50.json
        # ========================================================
        # For each sample size
        # ========================================================
        bin_num_mmi_dict = {}
        for bin_num in bin_list:
            triplet_selection_file_path = ProjectPaths.get_output_files_dir().joinpath(
                "data_fit_analysis",
                "mmi_triplet_selection_for_analysis",
                organism,
                f"neg_zero_pos_triplet_selection_bin{mmi_bin_num}.json",
            )
            triplet_selection_dict = load_data(triplet_selection_file_path)
            log_header(f"Loaded triplet selection dict:")
            # print(triplet_selection_dict)

            # ========================================================
            # For each triplet type (negative, zero, positive)
            # ========================================================
            triplet_type_triplet_sample_size_mmi_dict = {}
            for triplet_type, triplet_dict in triplet_selection_dict.items():
                # print(f"{triplet_type} triplets: {triplet_dict}")
                log_header(f"Processing {triplet_type} triplets")

                # ========================================================
                # For each triplet, load the corresponding GMM fit and scaler, generate synthetic data
                # ========================================================
                mmi_triplet_dict = {}
                for triplet, mmi_value in triplet_dict.items():
                    # print(f"Triplet: {triplet}, MMI: {mmi_value}")

                    triple1, triple2, triple3 = [
                        hist.lower() for hist in triplet.split("_")
                    ]
                    # gmm pickle_path: output\files\multi_modal_fit\bays_gmm_fit_pkl\human\h2a.h4r3me2_fit.pkl
                    gmm_fit_pkl_path_1 = ProjectPaths.get_output_files_dir().joinpath(
                        "multi_modal_fit",
                        "bays_gmm_fit_pkl",
                        organism,
                        f"{triple1}_fit.pkl",
                    )
                    gmm_fit_pkl_path_2 = ProjectPaths.get_output_files_dir().joinpath(
                        "multi_modal_fit",
                        "bays_gmm_fit_pkl",
                        organism,
                        f"{triple2}_fit.pkl",
                    )
                    gmm_fit_pkl_path_3 = ProjectPaths.get_output_files_dir().joinpath(
                        "multi_modal_fit",
                        "bays_gmm_fit_pkl",
                        organism,
                        f"{triple3}_fit.pkl",
                    )

                    gmm_fit_1 = load_data(gmm_fit_pkl_path_1)
                    gmm_fit_2 = load_data(gmm_fit_pkl_path_2)
                    gmm_fit_3 = load_data(gmm_fit_pkl_path_3)
                    # print(f"Loaded GMM fit: {gmm_fit_1}, {gmm_fit_2}, {gmm_fit_3}")

                    scaled_data_1 = gmm_fit_1.sample(sample_size)[0]
                    scaled_data_2 = gmm_fit_2.sample(sample_size)[0]
                    scaled_data_3 = gmm_fit_3.sample(sample_size)[0]
                    print(
                        f"Generated scaled synthetic data shape: {scaled_data_1.shape}, {scaled_data_2.shape}, {scaled_data_3.shape}"
                    )

                    # sccaler_path: output\files\multi_modal_fit\scaler_pkl\human\h2a.h4r3me2_scaler.pkl
                    scaler_pkl_path_1 = ProjectPaths.get_output_files_dir().joinpath(
                        "multi_modal_fit",
                        "scaler_pkl",
                        organism,
                        f"{triple1}_scaler.pkl",
                    )
                    scaler_pkl_path_2 = ProjectPaths.get_output_files_dir().joinpath(
                        "multi_modal_fit",
                        "scaler_pkl",
                        organism,
                        f"{triple2}_scaler.pkl",
                    )
                    scaler_pkl_path_3 = ProjectPaths.get_output_files_dir().joinpath(
                        "multi_modal_fit",
                        "scaler_pkl",
                        organism,
                        f"{triple3}_scaler.pkl",
                    )
                    scaler_1 = load_data(scaler_pkl_path_1)
                    scaler_2 = load_data(scaler_pkl_path_2)
                    scaler_3 = load_data(scaler_pkl_path_3)
                    print(f"Loaded scalers: {scaler_1}, {scaler_2}, {scaler_3}")

                    scaled_data_1 = scaler_1.inverse_transform(scaled_data_1)
                    scaled_data_2 = scaler_2.inverse_transform(scaled_data_2)
                    scaled_data_3 = scaler_3.inverse_transform(scaled_data_3)
                    print(
                        f"Inverse transformed synthetic data shape: {scaled_data_1.shape}, {scaled_data_2.shape}, {scaled_data_3.shape}"
                    )

                    edges = np.linspace(global_min, global_max, bin_num + 1)
                    mmi_from_synth_data = diff_uy_and_uy_givenv(
                        arr1=scaled_data_1,
                        arr2=scaled_data_2,
                        arr3=scaled_data_3,
                        spaced_bin_edges=edges,
                    )
                    print(
                        f"Computed MMI from synthetic data: {mmi_from_synth_data, sample_size}"
                    )
                    mmi_triplet_dict[triplet] = mmi_from_synth_data
                #     break
                # break

                triplet_type_triplet_sample_size_mmi_dict[triplet_type] = (
                    mmi_triplet_dict
                )
            bin_num_mmi_dict[str(bin_num)] = triplet_type_triplet_sample_size_mmi_dict

        # Save the sample_size_mmi_dict
        output_file_path = ProjectPaths.get_output_files_dir().joinpath(
            "data_fit_analysis",
            "mmi_bin_num_variation_analysis",
            organism,
            f"neg_zero_pos_triplet_sample_size_{sample_size}_mmi_var_with_bin_num.json",
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        json_file_saver(bin_num_mmi_dict, output_file_path)
