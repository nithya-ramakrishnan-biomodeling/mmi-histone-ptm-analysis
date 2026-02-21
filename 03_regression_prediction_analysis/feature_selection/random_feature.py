from multivariate_utils import (
    YEAST_FILE_PATH,
    HUMAN_FILE_PATH,
    ORGANISMS,
    load_data,
    json_file_saver,
    ProjectPaths,
    dir_maker,
)
from random import sample, shuffle, randint

if __name__ == "__main__":

    for organism in ORGANISMS:
        print(f"Processing organism: {organism}")

        feature_target_dict = {}

        if organism == "yeast":
            file_path = YEAST_FILE_PATH
        if organism == "human":
            file_path = HUMAN_FILE_PATH

        df = load_data(file_path)
        features = list(df.columns)
        n_features_to_select = 10

        # number of repeats to shuffle the features, less than 100
        n_repeats = randint(10, 100)
        for _ in range(n_repeats):
            shuffle(features)

        n_selected_features = sample(features, n_features_to_select)
        print(f"Randomly selected features: {n_selected_features}")

        one_feature = sample(features, 1)
        feature_target_dict["one_random_feature"] = one_feature
        print(f"One random feature: {one_feature}")

        # targets
        targets = [col for col in df.columns if col not in n_selected_features]
        feature_target_dict["selected_features"] = n_selected_features
        feature_target_dict["targets"] = targets

        # Save the selected features and targets to a JSON file
        output_dir = (
            ProjectPaths()
            .get_output_files_dir()
            .joinpath("feature_selection", "random_features", organism)
        )
        dir_maker(output_dir)
        output_file = output_dir.joinpath("random_selected_features.json")
        json_file_saver(feature_target_dict, output_file)
