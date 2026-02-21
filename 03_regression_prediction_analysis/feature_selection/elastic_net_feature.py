from multivariate_utils import (
    ORGANISMS,
    YEAST_FILE_PATH,
    load_data,
    json_file_saver,
    ProjectPaths,
    dir_maker,
    ElastnetFeature,
    HUMAN_FILE_PATH,
)

import pandas as pd

if __name__ == "__main__":

    for organism in ORGANISMS:
        if organism == "yeast":
            file_path = YEAST_FILE_PATH
        if organism == "human":

            file_path = HUMAN_FILE_PATH

        df = load_data(file_path)
        histone_mods = df.columns.tolist()

        histone_elastic_dict = {}
        elastic_features = {}
        for histone in histone_mods:
            print(f"Processing organism: {organism}, histone: {histone}")

            ElastnetFeature_obj = ElastnetFeature(target_mod=histone, df=df)
            results = ElastnetFeature_obj.elastic_net_feature_selection_single_target()

            elastic_features[histone] = results["features_selected_by_coefficients"]
            histone_elastic_dict[histone] = results

        # Save results
        output_dir = (
            ProjectPaths()
            .get_output_files_dir()
            .joinpath("feature_selection", "elastic_net", organism)
        )
        dir_maker(output_dir)
        output_file = output_dir.joinpath(f"elastic_feature_results.csv")
        pd.DataFrame.from_dict(histone_elastic_dict, orient="index").to_csv(output_file)
        json_output_file = output_dir.joinpath(f"elastic_target_feature_results.json")
        json_file_saver(elastic_features, json_output_file)
