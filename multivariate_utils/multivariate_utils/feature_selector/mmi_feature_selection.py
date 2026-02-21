import numpy as np
import pandas as pd
from argparse import ArgumentParser
from typing import Dict, Tuple, List
from ..datahandler.data_handler import load_data, dict_sort
from ..predictor.entropy_mutualinfo import mi_betwn_uandy


def n_pos_zero_neg_triplet_selector(
    mmi_data_dict: Dict[str, float], n: int, k: int
) -> Dict[str, List[str]]:
    """Select top n triplets with positive, zero, and negative MMI values with k each.

    Parameters
    ----------
    mmi_data_dict : Dict[str, float]
        Dictionary with triplet as key and MMI value as value.
    n : int
        Number of triplets to select.
    k : int
        Number of triplets to select for each category (positive, zero, negative).
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with keys "pos", "zero", "neg" and values as lists of selected triplets.
    """

    if mmi_data_dict is None or len(mmi_data_dict) == 0:
        raise ValueError("mmi_data_dict is empty or None.")

    if n % k != 0:
        raise ValueError("n must be divisible by k.")

    mmi_data_dict = dict_sort(mmi_data_dict, by="value", reverse=True)

    selected_triplet_dict = {"pos": {}, "zero": {}, "neg": {}}

    # pos and zero MMI selection
    for triplet, mmi_value in mmi_data_dict.items():

        # positive MMI
        if mmi_value > 0 and len(selected_triplet_dict["pos"]) < k:
            selected_triplet_dict["pos"][triplet] = mmi_value

        # zero MMI
        elif (
            mmi_value <= 0.09
            and mmi_value >= -0.09
            and len(selected_triplet_dict["zero"]) < k
        ):
            selected_triplet_dict["zero"][triplet] = mmi_value

        # # negative MMI
        # elif mmi_value < -0.09 and len(selected_triplet_dict["neg"]) < k:
        #     selected_triplet_dict["neg"].append((triplet, mmi_value))

        # check if we have selected k triplets for each category
        if (
            len(selected_triplet_dict["pos"]) >= k
            and len(selected_triplet_dict["zero"]) >= k
        ):
            break

        # neg MMI selection (from the end)
    for triplet, mmi_value in reversed(mmi_data_dict.items()):
        if mmi_value < -0.09 and len(selected_triplet_dict["neg"]) < k:
            selected_triplet_dict["neg"][triplet] = mmi_value

        if len(selected_triplet_dict["neg"]) >= k:
            break

    return selected_triplet_dict


def mmi_feature_selector(
    threshold: float,
    mmi_data: Dict[str, float],
    mode: str = "ge",  # "ge" for >=, "le" for <=
) -> Tuple[set, float]:
    """Select features based on MMI threshold and mode.

    Parameters
    ----------
    threshold : float
        MMI threshold for feature selection
    mmi_data : Dict[str, float]
        MMI values with feature triplet keys
    mode : str
        "ge" for >= threshold, "le" for <= threshold

    Returns
    -------
    Tuple[set, float]
        Selected feature set and threshold
    """
    feature_set = set()
    for key, mmi in mmi_data.items():
        if (mode == "ge" and mmi <= threshold) or (mode == "le" and mmi >= threshold):
            continue
        features = key.split("_")
        feature_set.update(features)
    return feature_set, threshold


def n_mmi_feature_selector(
    n: int, mmi_data: Dict[str, float], yeast_df: pd.DataFrame, edges: np.ndarray
) -> Tuple[set, str, float]:
    """Select n features from MMI triplet values

    Parameters
    ----------
    n : int
        Number of features to select
    mmi_data : Dict[str, float]
        MMI values with feature triplet keys
    yeast_df : pd.DataFrame
        Dataframe containing feature data
    edges : np.ndarray
        Bin edges for discretizing continuous variables

    Returns
    -------
    Tuple[set, str, float]
        Selected feature set, last MMI triplet, and corresponding MMI value.
        Note: MMI values shows the MMI threshold value at which the last feature was added.
    """

    feature_set = set()
    mmi = None
    last_key = None

    for key in mmi_data.keys():
        # check if we have enough features, then break
        if len(feature_set) >= n:
            print("Selected features:", feature_set, mmi)
            break

        features_raw = key.split("_")

        # Filter out features already in the set, keep only new ones
        new_features = [f for f in features_raw if f not in feature_set]

        if not new_features:
            continue  # All features already in set

        # getting the number of slots left
        slots_left = n - len(feature_set)
        if slots_left >= len(new_features):
            # Can add all new features
            feature_set.update(new_features)
            # print(f"added features: {new_features}")
        else:
            # Need to select `slots_left` features from `new_features`
            scored = []
            for f in new_features:
                if feature_set:
                    mi_scores = [
                        mi_betwn_uandy(
                            yeast_df[f].values, yeast_df[existing].values, edges
                        )
                        for existing in feature_set
                    ]
                    avg_mi = np.mean(mi_scores)
                else:
                    avg_mi = 0
                scored.append((f, avg_mi))
            scored.sort(key=lambda x: x[1], reverse=True)

            for f, _ in scored[:slots_left]:
                feature_set.add(f)
                # print(f"added feature (by best MI): {f}")

        mmi = mmi_data[key]
        last_key = key

    return feature_set, last_key, mmi


if __name__ == "__main__":

    parser = ArgumentParser(description="Select n features based on MMI values")
    parser.add_argument("n", type=int, help="Number of features to select")
    parser.add_argument(
        "feature_path",
        type=str,
        help="Path to the JSON file containing MMI values for feature triplets",
    )
    parser.add_argument(
        "feature_df_path",
        type=str,
        help="Path to the CSV file containing feature data",
    )
    parser.add_argument(
        "confg_path",
        type=str,
        help="Path to the configuration file for MI calculation",
    )
    args = parser.parse_args()

    n = args.n
    feature_path = args.feature_path
    feature_df_path = args.feature_df_path
    confg_path = args.confg_path
    yeast_df = load_data(feature_df_path)
    mmi_data = load_data(feature_path)
    config = load_data(confg_path)

    bin_num = config["bin_num"]
    global_min = config["global_min"]
    global_max = config["global_max"]
    edges = np.linspace(global_min, global_max, bin_num + 1)
    print(n_mmi_feature_selector(n, mmi_data, yeast_df, edges))
