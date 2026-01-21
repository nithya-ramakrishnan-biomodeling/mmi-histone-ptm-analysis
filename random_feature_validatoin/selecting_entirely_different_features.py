import os
import argparse
import json
import numpy as np
from pathlib import Path
import notebook__init__b
from path_utils.path_handler import ProjectPaths
from datahandler import dir_handler, data_handler
from random_utils.random_feature_selector import random_n_feature_selector

""" I have to select entirely different random features for each organism"""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select Entirely Different Random Features for Each Organism"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Target organism (e.g., 'human', 'yeast')",
    )
    parser.add_argument(
        "-n",
        "--num_features",
        type=int,
        required=True,
        help="Number of random features to select",
    )
    parser.add_argument(
        "-c",
        "--num_copies",
        type=int,
        default=10,
        help="Number of random feature copies to generate (default: 10)",
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def check_file_path(file_path: str):
    """Check if the file path exists and raise error if it doesn't."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")


def get_all_features_from_data(organism: str) -> set:
    """
    Load all features from the zero-mean data file.

    Parameters
    ----------
    organism : str
        The target organism.

    Returns
    -------
    set
        A set of all feature names.
    """

    # path example:
    # data\processed\yeast\yeast_zero_mean.csv

    zero_mean_file = os.path.join(
        ProjectPaths.get_cleaned_data_dir(), organism, f"{organism}_zero_mean.csv"
    )
    check_file_path(zero_mean_file)

    data_df = data_handler.csv_loader(zero_mean_file)
    all_features = set(data_df.columns)

    return all_features


# save random features based on number of copies


def main():
    args = parse_arguments()

    organism = args.organism
    num_features = args.num_features
    num_copies = args.num_copies
    random_seed = args.random_seed

    all_features = get_all_features_from_data(organism)

    # output\files\random_feature_selection_results
    output_dir = os.path.join(
        ProjectPaths.get_output_files_dir(),
        "random_feature_selection_results",
        "entirely_different_features",
        organism,
    )
    dir_handler.dir_maker(output_dir)

    for copy_idx in range(num_copies):
        selected_features = random_n_feature_selector(
            all_features,
            n=num_features,
            random_seed=random_seed + copy_idx,  # Different seed for each copy
        )

        output_file = os.path.join(
            output_dir, f"random_features_copy_{copy_idx + 1}.json"
        )
        with open(output_file, "w") as f:
            json.dump(list(selected_features), f)

        print(f"Saved random features for copy {copy_idx + 1} to {output_file}")


if __name__ == "__main__":
    main()
