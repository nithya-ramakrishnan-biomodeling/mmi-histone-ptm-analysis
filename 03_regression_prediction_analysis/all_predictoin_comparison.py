from multivariate_utils import (
    load_data,
    json_file_saver,
    ORGANISMS,
    ProjectPaths,
    HUMAN_FILE_PATH,
    YEAST_FILE_PATH,
)

import pandas as pd

if __name__ == "__main__":

    # MMI, PCA, OMP, Shap, elastic net - compare the test R2 scores for each method and each target feature in a json file
    for organism in ORGANISMS:

        # elastic_path: output\files\regression_prediction_analysis\elastic_net_prediction\human\elastic_net_r2_scores.json
        elastic_path = ProjectPaths.get_output_files_dir().joinpath("regression_prediction_analysis", 
                                                                    "elastic_net_prediction", organism, "elastic_net_r2_scores.json")
        elastic_r2_scores = load_data(elastic_path)

        # shap_path: output\files\shap_analysis_results\files\top_10_features\human\test_r2_scores.json
        shap_path = ProjectPaths.get_output_files_dir().joinpath("shap_analysis_results", "files",
                                                                  "top_10_features", organism, "test_r2_scores.json")
        shap_r2_scores = load_data(shap_path)

        # pca path cov: output\files\regression_prediction_analysis\pca_omit_target_prediction\pca_analysis_target_omit_prediction\human\PCA_10_prediction_cov_r2.json
        # pca_path = output\files\regression_prediction_analysis\pca_omit_target_prediction\yeast\pca_omit_target_prediction_r2_scores_5.json
        pca_path = ProjectPaths.get_output_files_dir().joinpath("regression_prediction_analysis", 
                                                                "pca_omit_target_prediction", 
                                                                organism, "pca_omit_target_prediction_r2_scores_5.json")
        


        pca_cov_r2_scores = load_data(pca_path)["cov"]
        pca_mi_r2_scores = load_data(pca_path)["mi"]

        # omp path : output\files\regression_prediction_analysis\omp_prediciton\yeast\yeast_omp_8_features_r2.json
        omp_path = ProjectPaths.get_output_files_dir().joinpath("regression_prediction_analysis", 
                                                                "omp_prediciton",organism, f"{organism}_omp_8_features_r2.json")
        omp_r2_scores = load_data(omp_path)

        # mmi path : output\files\regression_prediction_analysis\mmi_r2\yeast\yeast_mmi_cutoff_-0.41_le_r2.json, cut off -0.41 for yeast, -0.6 for human
        if organism == "yeast":
            mmi_cutoff = str(-0.285)
            # path_example: output\files\regression_prediction_analysis\mmi_r2_70_bin\yeast\mmi_r2_70_scores.json
            mmi_path = ProjectPaths.get_output_files_dir().joinpath("regression_prediction_analysis", "mmi_r2_70_bin", organism, "mmi_r2_70_scores.json")
            mmi_r2_scores = load_data(mmi_path)[mmi_cutoff]
        

        if organism == "human":
            mmi_cutoff = -0.6
            mmi_path = ProjectPaths.get_output_files_dir().joinpath("regression_prediction_analysis", "mmi_r2",
                                                                    organism, f"{organism}_mmi_cutoff_{mmi_cutoff}_le_r2.json")
            mmi_r2_scores = load_data(mmi_path)


        target_histone_mods = mmi_r2_scores.keys()

        comparison_dict = {}
        for target in target_histone_mods:
            # order:  MMi, PCA MI, PCA cov, OMP, Shap, Elastic net
            comparison_dict[target] = {
                "MMi": mmi_r2_scores[target],
                "PCA cov": pca_cov_r2_scores[target],
                "PCA MI": pca_mi_r2_scores[target],
                "OMP": omp_r2_scores[target],
                "Shap": shap_r2_scores[target],
                "Elastic net": elastic_r2_scores[target]
            }


        print(f"Comparison of R2 scores for {organism}:")
        # dataframe from comparison_dict
        comparison_df = pd.DataFrame.from_dict(comparison_dict, orient="index")
        print(comparison_df)

        # saving comparison df: 
        comparison_save_path = ProjectPaths.get_output_files_dir().joinpath("regression_prediction_analysis", 
                                                                            "all_prediction_comparison",
                                                                              f"{organism}_prediction_comparison.csv")
        
        comparison_save_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comparison_save_path)

        # break
        
