from multivariate_utils import (
    load_data,
    json_file_saver,
    ORGANISMS,
    ProjectPaths,
    HUMAN_FILE_PATH,
    YEAST_FILE_PATH,
)

if __name__ == "__main__":

    top_n = 5
    for organism in ORGANISMS:

        if organism == "yeast":
            df = load_data(YEAST_FILE_PATH)
        elif organism == "human":
            df = load_data(HUMAN_FILE_PATH)

        histone_mods = df.columns

        target_feature_dict = {}
        for histone_mod in histone_mods:

            # example file path:  output\files\shap_analysis_results\files\yeast\H2AK5ac_shap_importance.csv
            shap_file_path = ProjectPaths.get_output_files_dir().joinpath(
                "shap_analysis_results",
                "files",
                organism,
                f"{histone_mod}_shap_importance.csv",
            )
            feature_imp_df = load_data(shap_file_path)

            # select top 10 features, based on "rank" column
            top_n_features = feature_imp_df.nsmallest(top_n, "rank")["feature"].tolist()

            # print(f"Top {top_n} features for {organism} - {histone_mod}:")
            print(top_n_features)

            target_feature_dict[histone_mod] = top_n_features
            # save the top n features for each target feature in a json file

        output_path = ProjectPaths.get_output_files_dir().joinpath(
            "shap_analysis_results",
            "files",
            f"top_{top_n}_features",
            organism,
            f"shap_imp_top_{top_n}_features.json",
        )
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # create the directory if it doesn't exist
        json_file_saver(target_feature_dict, output_path)



        # break
