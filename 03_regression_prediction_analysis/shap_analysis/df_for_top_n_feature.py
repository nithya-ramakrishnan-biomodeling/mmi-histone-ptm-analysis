from multivariate_utils import load_data, ORGANISMS, ProjectPaths
import pandas as pd


if __name__ == "__main__":

    for organism in ORGANISMS:
        

        # example_file_path:output\files\shap_analysis_results\files\top_5_features\yeast\shap_imp_top_5_features.json
        shap_top_n_feature_file_path = ProjectPaths.get_output_files_dir().joinpath(
            "shap_analysis_results",
            "files",
            "top_5_features",
            organism,
            "shap_imp_top_5_features.json",
        )
        shap_top_n_feature_list_dict = load_data(shap_top_n_feature_file_path)

        # making into dataframe for better visualization,
        # format: target as index and features as columns, values are the importance of the features for each target
        df = pd.DataFrame.from_dict(shap_top_n_feature_list_dict, orient="index")

        # sorting  column lexicographically, don't sor because the order of the features is important,
        #  it is based on the importance of the features, so we will keep the order as it is, 
        # if we sort it then we will lose the information about the importance of the features.
        # df = df.reindex(sorted(df.columns), axis=1)
        print(df)
        print(f"Top 5 features for {organism}:")

        # saving the dataframe as csv file
        output_csv_path = ProjectPaths.get_output_files_dir().joinpath(
            "shap_analysis_results",
            "files",
            "top_5_features",
            organism,
            "shap_imp_top_5_features.csv",
        )

        df.to_csv(output_csv_path, index=True)