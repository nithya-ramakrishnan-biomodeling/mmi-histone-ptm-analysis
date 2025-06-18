from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from itertools import product
from entropy_mutualinfo import mi_betwn_uandy
from entropy_mutualinfo import diff_uy_and_uy_givenv

# from visualization import plotter


def line_space_gen(min_value, max_value, bin_number: int):
    """Generates n number of bins from the min, max, and the number of bins.

    Parameters:
    ----------
    - iterable: input array.
    - bin_number: number of bins."""
    # finding min and max of the input list

    # iterable_min, iterable_max = min(iterable), max(iterable)
    bin_spacing = np.linspace(
        min_value, max_value, bin_number
    )  # Equally spacing the bins

    return bin_spacing


def mutual_info_assigner(
    data: np.ndarray,
    first_index: int,
    second_index: int,
    spaced_bin_edges,
    third_index: int = None,
    methods: str = None,
):
    """Function that assigns different MI calculation based on the values of method parameter

    Parameters
    ----------
    data : np.ndarray
        take input dataframe as numpy array
    first_index : int
        first col index of the input dataframe
    second_index : int
        second col index of the input dataframe
    third_index : int, optional
        third col index of the input dataframe, by default None (case for two variable MI)
    methods : str, optional
        the type of MI that are needed to calculate, by default "mi_btw_u_and_y"
    bin_num: int
        the number of bins that used for calculating the continuous random variables.

    """

    if third_index is not None:
        arr1, arr2, arr3 = (
            data[:, first_index],
            data[:, second_index],
            data[:, third_index],
        )

    if third_index is None:
        arr1, arr2 = data[:, first_index], data[:, second_index]

    if methods == "mi_btw_u_and_y" and third_index is None:
        return mi_betwn_uandy(arr1, arr2, spaced_bin_edges)

    if methods == "diff_uy_and_uy_givenv_entro":
        return diff_uy_and_uy_givenv(arr1, arr2, arr3, spaced_bin_edges)

    return None


def mi_parallel_generator(
    dataframe: pd.DataFrame, spaced_bin_edges, mi_methods: str, comb: int = "None"
) -> pd.DataFrame:
    """Used for to parallelize the Mi calculation for all the possible dataframe col combinations

    Parameters
    ----------
    dataframe : pd.DataFrame
        input pandas dataframe
    mi_methods : str
        The type of MI calculation that are needed.
        Please refer the entropy_mutual info.py for more details.
    comb : int, optional
        for the three variable cases it is 3 otherwise it is 2, by default "None"

    Returns
    -------
    pd.DataFrame
    Calculated MI values of all possible col combinations.
    """

    data = dataframe.to_numpy()
    feature_number = dataframe.shape[1]
    feature_number_array = list(range(feature_number))
    features = dataframe.keys()
    all_feature_index_comb = list(product(range(feature_number), repeat=comb))

    # case where 3 combinations.
    if comb == 3:

        all_feature_index_comb_corrctd = [
            (j, k, i)
            for i in feature_number_array
            for j in feature_number_array
            for k in feature_number_array
        ]
        print(all_feature_index_comb_corrctd)
        first_and_second_mods = [
            f"{i}_{j}" for i, j in product(features, repeat=comb - 1)
        ]

        parallel_output = Parallel(n_jobs=-1)(
            delayed(mutual_info_assigner)(
                data,
                first_index,
                second_index,
                spaced_bin_edges=spaced_bin_edges,
                third_index=third_index,
                methods=mi_methods,
            )
            for first_index, second_index, third_index in all_feature_index_comb_corrctd
        )

        reshape_value = (len(first_and_second_mods), len(features))
        index = first_and_second_mods

    if comb == 2:

        parallel_output = Parallel(n_jobs=4)(
            delayed(mutual_info_assigner)(
                data,
                first_index,
                second_index,
                spaced_bin_edges=spaced_bin_edges,
                methods=mi_methods,
            )
            for first_index, second_index in all_feature_index_comb
        )

        reshape_value = (len(features), len(features))
        index = features

    # creating pandas dataframe
    output_dataframe = pd.DataFrame(
        np.array(parallel_output).reshape(reshape_value), index=index, columns=features
    )

    return output_dataframe


if __name__ == "__main__":
    # file loading.
    input_file_path = r"/home/group_nithya01/Desktop/dhana_ibab_intern/file_downloads/cluster_work/data_work/data/histone_time/histone_mod_0.csv"
    input_data_frame = pd.read_csv(
        input_file_path, index_col=0
    )  # index_col =0 , is used to avoid the inclusion of default index as col in the pandas dataframe.

    # solving the formatting error due the col name formatting happened in the input csv file.
    new_col_name = [i.strip() for i in input_data_frame.keys()]
    input_data_frame.columns = new_col_name

    # selected histone modifications.
    input_data_frame_sliced = input_data_frame[
        ["H3K9ac", "H3K27ac", "H3K36me3", "H3K79me3", "H3K56ac"]
    ]

    bin_edges = line_space_gen(
        min_value=min(input_data_frame_sliced.min()),
        max_value=max(input_data_frame_sliced.max()),
        bin_number=30,
    )

    # running the mutual information calculation.
    dataframe_value = mi_parallel_generator(
        input_data_frame_sliced,
        spaced_bin_edges=bin_edges,
        mi_methods="diff_uy_and_uy_givenv_mi",
        comb=3,
    )  # .to_csv("difference_between_mi and_cond_mi.csv")
    # plotter(dataframe_value, "diff_uy_and_uy_givenv_mi")
