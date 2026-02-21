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


sample_size_list = (
    np.array(
        [
            6.6,
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
            5000,
            10000,
        ]
    )
    * 1e4
)

if __name__ == "__main__":

    # ========================================================
    # For each organism
    # ========================================================
    for organism in ORGANISMS:
        log_header(f"Processing organism: {organism}")

        if organism == "human":
            bin_num = MI_CONFIG[organism]["bin_num"]
            global_max = MI_CONFIG[organism]["global_max"]
            global_min = MI_CONFIG[organism]["global_min"]
            edges = np.linspace(global_min, global_max, bin_num + 1)

        elif organism == "yeast":
            bin_num = MI_CONFIG[organism]["bin_num"]
            global_max = MI_CONFIG[organism]["global_max"]
            global_min = MI_CONFIG[organism]["global_min"]
            edges = np.linspace(global_min, global_max, bin_num + 1)

        # example_path : output\files\data_fit_analysis\mmi_triplet_selection_for_analysis\human\neg_zero_pos_triplet_selection_bin50.json
        # ========================================================
        # For each sample size
        # ========================================================
        sample_size_mmi_dict = {}
        sample_size_list = sample_size_list.astype(int)
        for sample_size in sample_size_list:
            triplet_selection_file_path = ProjectPaths.get_output_files_dir().joinpath(
                "data_fit_analysis",
                "mmi_triplet_selection_for_analysis",
                organism,
                f"neg_zero_pos_triplet_selection_bin{bin_num}.json",
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
            sample_size_mmi_dict[str(sample_size)] = (
                triplet_type_triplet_sample_size_mmi_dict
            )

        # Save the sample_size_mmi_dict
        output_file_path = ProjectPaths.get_output_files_dir().joinpath(
            "data_fit_analysis",
            "mmi_sample_size_variation_analysis",
            organism,
            f"neg_zero_pos_triplet_mmi_var_with_sample_size_bin{bin_num}.json",
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        json_file_saver(sample_size_mmi_dict, output_file_path)
