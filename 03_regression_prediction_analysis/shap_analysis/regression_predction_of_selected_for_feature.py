from multivariate_utils import (
    load_data,
    json_file_saver,
    dict_sort,
    ORGANISMS,
    ProjectPaths,
    HUMAN_FILE_PATH,
    YEAST_FILE_PATH,
    Histone_Regressor,
)

if __name__ == "__main__":


    for organism in ORGANISMS:

        # example file path:  output\files\shap_analysis_results\files\top_10_features\human\shap_imp_top_10_features.json
        feature_path = ProjectPaths.get_output_files_dir().joinpath(
            "shap_analysis_results",
            "files",
            "top_10_features",
            organism,
            "shap_imp_top_10_features.json",
        )

        feature_dict = load_data(feature_path)

        if organism == "yeast":
            df = load_data(YEAST_FILE_PATH)

        elif organism == "human":
            df = load_data(HUMAN_FILE_PATH)

        histone_mods = df.columns

        test_r2_dict = {}
        for histone_mod in histone_mods:
            top_n_features = feature_dict[histone_mod]

            print(f"Top features for {organism} - {histone_mod}:")
            print(top_n_features)
            
            feature_arr, target_arr = df[top_n_features].values, df[histone_mod].values
            # create a regressor object and train the model using only the top n features
            regressor = Histone_Regressor(
                feature=feature_arr,
                target=target_arr,
                cv_n_jobs=4,
            )
            cv_results = regressor.get_cv_results()
            test_r2 = cv_results["test_score"] # get the test r2 score from the cv results

            print(f"Test R2 score for {organism} - {histone_mod}: {test_r2}")
            test_r2_dict[histone_mod] = test_r2
        
        # sort the test r2 scores in descending order
        test_r2_dict = dict_sort(test_r2_dict, by="value", reverse=True)

        # save the test r2 scores for each target feature in a json file
        output_path = ProjectPaths.get_output_files_dir().joinpath(
            "shap_analysis_results",    
            "files",
            "top_10_features",
            organism,
            "test_r2_scores.json",
        )
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # create the directory if it doesn't exist
        json_file_saver(test_r2_dict, output_path)