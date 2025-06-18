import numpy as np

def bin_gen(iterable, bin_edges):
    """Generates bins and bin indices of the input array.

    Parameters:
    ----------
    iterable : array-like
        Input array.
    bin_edges : array-like
        Edges of the bins.

    Returns:
    ----------
    binned_indices : array-like
        Indices of the bins for the input array.
    """
    # Finding the indices for the bins
    binned_indices = np.digitize(iterable, bin_edges) - 1

    return binned_indices

def joint_prob_two(input_list_1, input_list_2, spaced_bin_edges):
    """Calculate the joint probabilities between two arrays.

    Parameters:
    ----------
    input_list_1 : array-like
        First input array.
    input_list_2 : array-like
        Second input array.
    spaced_bin_edges : array-like
        Edges of the bins.

    Returns:
    ----------
    joint_prob : ndarray
        Joint probability matrix.
    """
    # Generating the bins & bin indices calculation
    input_bin_1 = bin_gen(input_list_1, spaced_bin_edges)
    input_bin_2 = bin_gen(input_list_2, spaced_bin_edges)

    bin_num = len(spaced_bin_edges)

    # Joint probability calculation
    joint_prob = np.zeros((bin_num, bin_num))
    for i in range(len(input_bin_1)):
        joint_prob_row_value = input_bin_1[i]
        joint_prob_col_value = input_bin_2[i]
        joint_prob[joint_prob_row_value, joint_prob_col_value] += 1

    joint_prob = np.divide(joint_prob, len(input_list_1))

    return joint_prob

def marginal_prob_calculator(joint_prob_value, axis):
    """Calculate the marginal probability from the joint probability.

    Parameters:
    ----------
    joint_prob_value : ndarray
        Joint probability matrix.
    axis : int
        Axis along which the probability summing occurs.

    Returns:
    ----------
    marginal_prob : ndarray
        Marginal probability array.
    """
    return np.sum(joint_prob_value, axis=axis)

def entropy_calculator(array: np.ndarray) -> float:
    """Calculate the entropy of the given probability values.

    Parameters:
    ----------
    array : np.ndarray
        Probability array.

    Returns:
    ----------
    entropy : float
        Calculated entropy.
    """
    flatten_prob = array.flatten()

    # To avoid the log calculation math domain error when it encounters zero values.
    without_zero_prob = flatten_prob[flatten_prob > 0]

    return -np.sum(without_zero_prob * np.log2(without_zero_prob))

def mi_betwn_uandy(arr1: np.array, arr2: np.array, spaced_bin_edges):
    """Calculate the mutual information between two variables U and Y.

    Parameters:
    ----------
    arr1 : np.array
        First input array.
    arr2 : np.array
        Second input array.
    spaced_bin_edges : array-like
        Edges of the bins.

    Returns:
    ----------
    mi_u_and_y : float
        Mutual information between the two arrays.
    """
    # Calculate joint probability
    entropy_of_two = joint_prob_two(arr1, arr2, spaced_bin_edges)
    # Calculate marginal probabilities
    entropy_u = marginal_prob_calculator(entropy_of_two, axis=(1))
    entropy_y = marginal_prob_calculator(entropy_of_two, axis=(0))
    # Calculate mutual information
    mi_u_and_y = (
        entropy_calculator(entropy_u)
        + entropy_calculator(entropy_y)
        - entropy_calculator(entropy_of_two)
    )

    return mi_u_and_y
