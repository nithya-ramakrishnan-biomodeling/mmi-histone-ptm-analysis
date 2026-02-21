from multivariate_utils import (
    YEAST_FILE_PATH,
    HUMAN_FILE_PATH,
    ORGANISMS,
    load_data,
    json_file_saver,
    smart_map,
    ProjectPaths,
    estimate_optimal_workers,
    dir_maker,
)
from functools import partial
from pathlib import Path


def highest_correlated_feature(target_column, df):
    """getting highest correlated feature for each target histones"""
    correlations = df.corr()[target_column].drop(target_column)
    # print(correlations)
    highest_corr_feature = correlations.abs().idxmax()
    highest_corr_value = correlations[highest_corr_feature]
    return [highest_corr_feature, highest_corr_value]


output_dir = Path.joinpath(
    ProjectPaths().get_output_files_dir(),
    "feature_selection",
    "highest_correlated_features",
)
dir_maker(output_dir)

if __name__ == "__main__":

    for organism in ORGANISMS:
        print(f"Processing organism: {organism}")
        file_path = YEAST_FILE_PATH if organism == "yeast" else HUMAN_FILE_PATH
        df = load_data(file_path=file_path)
        histone_mods = df.columns
        results = smart_map(
            partial(highest_correlated_feature, df=df),
            histone_mods,
            show_progress=True,
            max_workers=estimate_optimal_workers(),
        )

        # saving results
        results_dict = dict(zip(histone_mods, results))
        print(results_dict)

        # saving to json
        json_saver_path = output_dir.joinpath(
            f"{organism}_highest_correlated_features.json"
        )
        json_file_saver(results_dict, json_saver_path)
