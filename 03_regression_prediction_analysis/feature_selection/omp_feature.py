from multivariate_utils import (
    YEAST_FILE_PATH,
    HUMAN_FILE_PATH,
    ORGANISMS,
    load_data,
    json_file_saver,
    ProjectPaths,
    dir_maker,
    select_top_n_omp_features,
)

if __name__ == "__main__":

    organism_list = ORGANISMS
    feature_to_select = [8, 10, 12]

    for organism in organism_list:
        print(f"Processing organism: {organism}")
        if organism == "yeast":
            file_path = YEAST_FILE_PATH
        if organism == "human":
            file_path = HUMAN_FILE_PATH
        df = load_data(file_path)
        histone_mods = list(df.columns)

        histone_mod_dict = {}
        for histone in histone_mods:
            target_ar = df[histone].values
            feature_histones = [h for h in histone_mods if h != histone]
            feature_ar = df[feature_histones].values

            n_feature_dict = {}
            for n_features in feature_to_select:
                n_features_to_select = n_features
                indices, coefficients = select_top_n_omp_features(
                    features=feature_ar,
                    target=target_ar,
                    n_features=n_features_to_select,
                )
                n_feature_dict[n_features] = {
                    "selected_features": [feature_histones[i] for i in indices],
                    "coefficients": coefficients.tolist(),
                }

                # print(f"Selected features for target {histone}: {n_feature_dict}")
                # break

            histone_mod_dict[histone] = n_feature_dict

        # print(f"Completed processing for organism: {organism}")
        # print(histone_mod_dict)

        # Save the results to a JSON file
        output_dir = (
            ProjectPaths()
            .get_output_files_dir()
            .joinpath("feature_selection", "omp_features", organism)
        )
        dir_maker(output_dir)
        output_file = output_dir.joinpath(f"omp_selected_features.json")
        json_file_saver(histone_mod_dict, output_file)
