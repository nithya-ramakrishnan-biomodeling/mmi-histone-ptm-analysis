import numpy as np
from typing import List, Optional


def bin_gen(iterable, bin_edges):
    """Generates n number of bins and bin indices of the input array.

    Parameters:
    ----------
    - iterable: input array.
    - bin_number: number of bins."""

    # getting the indices for the bins
    binned_indices = np.digitize(iterable, bin_edges, right=True) - 1

    return binned_indices


def joint_prob(
    input_list_1: List[float],
    input_list_2: List[float],
    spaced_bin_edges: List[float],
    input_list_3: Optional[List[float]] = None,
) -> np.ndarray:
    """Calculate joint probability for 2D and 3D cases.

    Returns
    -------
    np.ndarray:  joint probability.

    Raises
    ------
    ValueError: if the length of input sequence is not equal.

    """
    # Input validation
    if len(input_list_1) != len(input_list_2):
        raise ValueError("input_list_1 and input_list_2 must have the same length")
    if input_list_3 is not None and len(input_list_3) != len(input_list_1):
        raise ValueError(
            "input_list_3 must have the same length as input_list_1 and input_list_2"
        )

    # Convert inputs to numpy arrays
    input_array_1 = np.array(input_list_1)
    input_array_2 = np.array(input_list_2)
    input_array_3 = np.array(input_list_3) if input_list_3 is not None else None

    # Prepare data and bins
    data = (
        [input_array_1, input_array_2]
        if input_array_3 is None
        else [input_array_1, input_array_2, input_array_3]
    )
    bins = [spaced_bin_edges] * (2 if input_array_3 is None else 3)

    # Calculate joint probability
    hist, _ = np.histogramdd(np.column_stack(data), bins=bins)
    joint_prob_calc = hist / len(input_array_1)

    return joint_prob_calc


def marginal_prob_calculator(jont_prb_value, axis):
    """Calculate the marginal probability from the joint probability.

    Parameters
    ----------
    jont_prb_value : nd.arraay
    Joint probability matrix.
    axis : tuple
    Axis which the probability summing occurs.
    """

    return np.sum(jont_prb_value, axis=axis, dtype=np.float64)


# Entropy functions:
def entropy_calculator(array: np.ndarray) -> float:
    """calculate the entropy of the given probability values.

    Parameters
    ----------
    array : np.ndarray
        probability arrray

    Returns
    -------
    float
        returns the calculated probability.
    """

    flatten_prob = array.flatten()

    # to avoid the log calculation  math domain error when it encounters the zero values.
    without_zero_prob = flatten_prob[flatten_prob > 0]

    return -np.sum(without_zero_prob * np.log2(without_zero_prob), dtype=np.float64)


# -------------------------------------Mutual--Information--Calculation-------------------------------------------------------------
# let assume:
# arr3 as V"
# "arr1 as U "
# "arr2 as Y "


# Mutual information  between variables U & Y
#  Function 2 follows this equation
#   #! : I(U ; Y ) = H(U ) + H(Y ) − H(U, Y )
def mi_betwn_uandy(arr1: np.array, arr2: np.array, spaced_bin_edges):

    entropy_of_two = joint_prob(arr1, arr2, spaced_bin_edges)
    entropy_u = entropy_calculator(marginal_prob_calculator(entropy_of_two, axis=(1)))
    entropy_y = entropy_calculator(marginal_prob_calculator(entropy_of_two, axis=(0)))
    mi_u_and_y = entropy_u + entropy_y - entropy_calculator(entropy_of_two)

    return mi_u_and_y


# Function 4 :
# Three variable mutual information calculation.
# #! I(U ; Y ) − IV (U ; Y/V ) = H(U ) + H(V ) + H(Y ) − (H(U, V ) + H(V, Y ) + H(U, Y )) + H(U, V, Y )
def diff_uy_and_uy_givenv(
    arr1: np.array, arr2: np.array, arr3: np.array, spaced_bin_edges
):

    entropy_from_three = joint_prob(arr1, arr2, spaced_bin_edges, input_list_3=arr3)
    entropy_from_two_uv = marginal_prob_calculator(entropy_from_three, axis=(1))
    entropy_from_two_vy = marginal_prob_calculator(entropy_from_three, axis=(0))
    entropy_from_two_uy = marginal_prob_calculator(entropy_from_three, axis=(2))

    entropy_u = marginal_prob_calculator(entropy_from_three, axis=(1, 2))
    entropy_y = marginal_prob_calculator(entropy_from_three, axis=(0, 2))
    entropy_v = marginal_prob_calculator(entropy_from_three, axis=(0, 1))

    equation = (
        (
            entropy_calculator(entropy_u)
            + entropy_calculator(entropy_v)
            + entropy_calculator(entropy_y)
        )
        - (
            entropy_calculator(entropy_from_two_uv)
            + entropy_calculator(entropy_from_two_vy)
            + entropy_calculator(entropy_from_two_uy)
        )
        + entropy_calculator(entropy_from_three)
    )
    return equation
