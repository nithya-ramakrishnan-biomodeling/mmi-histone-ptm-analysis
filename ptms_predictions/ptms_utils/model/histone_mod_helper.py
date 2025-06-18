import numpy as np
from typing import Dict, List


def calculate_histone_scores(
    corr_df: Dict,
    x_values: List,
    histone_mods: List[str],
    transform_func=lambda x: np.square(np.abs(x)),
) -> Dict[str, np.ndarray]:
    """
    Calculate histone modification scores for given x values and modifications.

    Parameters:
        corr_df (Dict): Dictionary of correlation data.
        x_values (List): List of x values.
        histone_mods (List[str]): List of histone modifications.
        transform_func (Callable): Function to apply to scores, defaulting to np.square(np.abs(x)).

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping x values to their transformed scores
    """
    results = {}

    for x in x_values:
        # Calculate scores based on histone_mods
        scores = np.array([corr_df[x][mod] for mod in histone_mods])

        # Apply transformation
        transformed_scores = transform_func(scores)
        print(f"{x}: {transformed_scores.round(2)}")

        results[x] = transformed_scores

    return results


def extract_unique_histones(data):
    """
    Extracts all unique histone values from both the top-level and inner keys of a dictionary.

    Args:
        data (dict): A dictionary where top-level keys are histone values and each value is
                     another dictionary with keys that may contain multiple histone values
                     separated by underscores.

    Returns:
        set: A set of all unique histone values.
    """
    # Initialize a set to collect unique histone values
    unique_histones = set()

    # Add top-level keys to the set
    unique_histones.update(data.keys())

    # Process each inner key and add individual histone values to the set
    for inner_dict in data.values():
        for inner_key in inner_dict.keys():
            # Split by "_" and add each part to the set
            unique_histones.update(inner_key.split("_"))

    return unique_histones
