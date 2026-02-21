from multivariate_utils import (
    diff_uy_and_uy_givenv, 
    joint_prob, entropy_calculator, mi_betwn_uandy,
    YEAST_FILE_PATH,
    load_data, MI_CONFIG, 
)
import numpy as np


def joint_entropy (arr1: np.array, arr2: np.array, spaced_bin_edges, arr3: np.array=None):
    """Calculate the joint entropy of three variables.

    Parameters
    ----------
    arr1 : np.array
        First variable.
    arr2 : np.array
        Second variable.
    arr3 : np.array
        Third variable.
    spaced_bin_edges : np.array
        Edges for histogram binning.

    Returns
    -------
    float
        Joint entropy of the three variables.
    """
    joint_prob_three = joint_prob(arr1, arr2, spaced_bin_edges, input_list_3=arr3)
    return entropy_calculator(joint_prob_three)

if __name__ == "__main__":

    organism = "yeast"

    histone_df = load_data(YEAST_FILE_PATH)
    histone_mods = histone_df.columns[:3]
    # print(histone_df["H4R3me2s"].values)

    # test = ["H4R3me2s", "H4K16ac", "H3K4ac"]
    test= ["H3K4ac", "H3K9ac", "H3K56ac"]
    ar1, ar2, ar3 = histone_df[histone_mods].values.T
    ar1, ar2, ar3 = histone_df[test].values.T
    histone_mods = test
    # print(ar2)


    # parameters for histogram binning
    global_max, global_min = MI_CONFIG[organism]["global_max"], MI_CONFIG[organism]["global_min"]
    bin_num = MI_CONFIG[organism]["bin_num"]

    # individual entropies 
    spaced_bin_edges = np.linspace(global_min, global_max, bin_num + 1)

    h1 = mi_betwn_uandy(ar1, ar1, spaced_bin_edges).__round__(2)
    h2 = mi_betwn_uandy(ar2, ar2, spaced_bin_edges).__round__(2)
    h3 = mi_betwn_uandy(ar3, ar3, spaced_bin_edges).__round__(2)
    print(f"Entropy of {histone_mods[0]}: {h1}")
    print(f"Entropy of {histone_mods[1]}: {h2}")
    print(f"Entropy of {histone_mods[2]}: {h3}")

    #joint entropy3 
    joint_h = joint_entropy(ar1, ar2, spaced_bin_edges, arr3=ar3).__round__(2)
    print(f"Joint entropy of {histone_mods[0]}, {histone_mods[1]} and {histone_mods[2]}: {joint_h}")

    #  joint entropy2
    joint_h_12 = joint_entropy(ar1, ar2, spaced_bin_edges).__round__(2)
    joint_h_13 = joint_entropy(ar1, ar3, spaced_bin_edges).__round__(2)
    joint_h_23 = joint_entropy(ar2, ar3, spaced_bin_edges).__round__(2)
    print(f"Joint entropy of {histone_mods[0]} and {histone_mods[1]}: {joint_h_12}")
    print(f"Joint entropy of {histone_mods[0]} and {histone_mods[2]}: {joint_h_13}")
    print(f"Joint entropy of {histone_mods[1]} and {histone_mods[2]}: {joint_h_23}")

    # mmi 
    mmi_value = diff_uy_and_uy_givenv(ar1, ar2, ar3, spaced_bin_edges)
    print(f"MMI value for {histone_mods[0]}, {histone_mods[1]} and {histone_mods[2]}: {mmi_value}")

    print((h1 + h2 + h3) - (joint_h_12 + joint_h_13 + joint_h_23) + joint_h)


    # Mi between two variables: 
    mi12 = mi_betwn_uandy(ar1, ar2, spaced_bin_edges).__round__(2)
    print(f"Mutual information I({histone_mods[0]};{histone_mods[1]}): {mi12}")
    mi13 = mi_betwn_uandy(ar1, ar3, spaced_bin_edges).__round__(2)
    print(f"Mutual information I({histone_mods[0]};{histone_mods[2]}): {mi13}")
    mi23 = mi_betwn_uandy(ar2, ar3, spaced_bin_edges).__round__(2)
    print(f"Mutual information I({histone_mods[1]};{histone_mods[2]}): {mi23}")




    print("================================================== \n")
    # I(X;Y/Z) = I(X:Y) - I(X:Y:Z)
    # x, y
    ix_y = mi_betwn_uandy(ar1, ar2, spaced_bin_edges)
    i_x_y_given_z = ix_y - mmi_value
    print(f"I({histone_mods[0]};{histone_mods[1]}|{histone_mods[2]}): {i_x_y_given_z}")
    print(f" I({histone_mods[0]};{histone_mods[1]}): {ix_y}")

    # x, z
    ix_z = mi_betwn_uandy(ar1, ar3, spaced_bin_edges)
    i_x_z_given_y = ix_z - mmi_value
    print(f"I({histone_mods[0]};{histone_mods[2]}|{histone_mods[1]}): {i_x_z_given_y}")
    print(f" I({histone_mods[0]};{histone_mods[2]}): {ix_z}")

    # y, z
    iy_z = mi_betwn_uandy(ar2, ar3, spaced_bin_edges)
    i_y_z_given_x = iy_z - mmi_value
    print(f"I({histone_mods[1]};{histone_mods[2]}|{histone_mods[0]}): {i_y_z_given_x}")
    print(f" I({histone_mods[1]};{histone_mods[2]}): {iy_z}")




# latex code format 
# make grammatical corrections to the following text:
""" The conditional MI \text{I(X;Y|Z)} and MI \text{I(X;Y)} values are :
I(\text{H4R3me2s;H4K16ac|H3K4ac}): 0.62
 I(\text{H4R3me2s;H4K16ac}): 0.24, 
I(\text{H4R3me2s;H3K4ac|H4K16ac}): 0.76
 I(\text{H4R3me2s;H3K4ac}): 0.38, 
I(\text{H4K16ac;H3K4ac|H4R3me2s}): 0.48
 I(\text{H4K16ac;H3K4ac}): 0.10"""

