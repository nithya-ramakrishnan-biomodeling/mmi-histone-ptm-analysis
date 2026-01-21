# random_feature_selector file:

## Idea:
"""Used to select random features from a dataset.

Parameters:
- num_features: int, number of features to select
- random_state: int, seed for random number generator, default is 42
Returns:
- selected_features: list of selected features"""

import numpy as np


# function: random features selector
def random_n_feature_selector(input_features: set, n: int, random_seed: int = 42):
    """Selects n random features from the input set of features.
    Parameters:
    - input_features: set, a set of available features to choose from
    - n: int, number of features to select
    - random_seed: int, seed for random number generator, default is 42
    Returns:
    - selected_features: list, a list of n randomly selected features
    """
    # Check if input_features is a set
    if not isinstance(input_features, set):
        raise TypeError(
            f"input_features must be a set, got {type(input_features).__name__}"
        )

    # Check if n is an integer
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}")

    # Check if n doesn't exceed available features
    if n > len(input_features):
        raise ValueError(
            f"n ({n}) cannot be greater than total features ({len(input_features)})"
        )

    # Set random seed for reproducibility
    # np.random.seed(random_seed)

    # Convert set to list (sets are unordered)
    features_list = list(input_features)

    # Randomly select n features without replacement
    selected_features = np.random.choice(features_list, size=n, replace=False).tolist()

    return selected_features


if __name__ == "__main__":
    # Example usage
    features = {"feature1", "feature2", "feature3", "feature4", "feature5"}
    num_to_select = 3
    selected = random_n_feature_selector(features, num_to_select, random_seed=123)
    print(f"Selected features: {selected}")
