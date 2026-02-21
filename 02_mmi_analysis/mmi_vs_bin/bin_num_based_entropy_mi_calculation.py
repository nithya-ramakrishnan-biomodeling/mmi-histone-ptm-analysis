"""
Entropy Calculation for All Histone PTMs Based on Varying Bin Numbers

This script automatically calculates entropy values for all histone PTMs across a range of bin numbers
for both human and yeast organisms, helping to determine the optimal bin count for mutual information calculations.
"""

import os
from numpy import linspace
from functools import partial
from itertools import combinations
from multivariate_utils import (
    load_data,
    json_file_saver,
    dir_maker,
    mi_betwn_uandy,
    diff_uy_and_uy_givenv,
    ProjectPaths,
    smart_map,
    joint_prob,
    entropy_calculator,
    estimate_optimal_workers,
)

from multivariate_utils import ORGANISMS, BIN_NUM_LIST, file_globl_max_min_config


def entropy_calculation(histon_mod, df, edges):
    ar = df[histon_mod].values
    entropy = mi_betwn_uandy(ar, ar, edges)
    return histon_mod, entropy


def joint_entropy_calculation(duplet, edges, df):
    mod1, mod2 = duplet
    ar1 = df[mod1].values
    ar2 = df[mod2].values
    joint_entropy = joint_prob(ar1, ar2, edges)
    entropy_calculation = entropy_calculator(joint_entropy)
    return duplet, entropy_calculation


def two_mi_calculation(duplet, edges, df):
    mod1, mod2 = duplet
    ar1 = df[mod1].values
    ar2 = df[mod2].values
    mi_value = mi_betwn_uandy(ar1, ar2, edges)
    return duplet, mi_value


def triplet_joint_entropy_calculation(triplet, edges, df):
    mod1, mod2, mod3 = triplet.split("_")
    ar1 = df[mod1].values
    ar2 = df[mod2].values
    ar3 = df[mod3].values
    joint_entropy = joint_prob(ar1, ar2, edges, ar3)
    entropy = entropy_calculator(joint_entropy)
    return triplet, entropy


def three_var_mi_calculation(triplet, edges, df):
    mod1, mod2, mod3 = triplet.split("_")
    ar1 = df[mod1].values
    ar2 = df[mod2].values
    ar3 = df[mod3].values
    mi_value = diff_uy_and_uy_givenv(ar1, ar2, ar3, edges)
    return triplet, mi_value


if __name__ == "__main__":

    # ========================================================
    # entropy calculation function
    # ======================================================
    mode = "auto"
    max_workers = estimate_optimal_workers(mode=mode)
    for organism in ORGANISMS[:]:
        file_path, global_min, global_max = file_globl_max_min_config(organism)
        histone_df = load_data(file_path)
        histone_mods = histone_df.columns.tolist()

        # output\files\mmi_value_analysis\mmi_values_bin_based\yeast\histone_triplet_mmi_bin100.json
        # organism = "yeast"
        if organism == "yeast":
            neg_mmi_value_file = os.path.join(
                ProjectPaths().get_output_files_dir(),
                "mmi_value_analysis",
                "mmi_values_bin_based",
                organism,
                "histone_triplet_mmi_bin100.json",
            )

        if organism == "human":
            neg_mmi_value_file = os.path.join(
                ProjectPaths().get_output_files_dir(),
                "mmi_value_analysis",
                "mmi_values_bin_based",
                organism,
                "histone_triplet_mmi_bin130.json",
            )

        mmi_values_dict = load_data(neg_mmi_value_file)

        # ========================================================
        # selecting mmi histone triplets 3 each for negative, zero, and positive mmi values
        # =======================================================

        triplets = []
        number_of_triplets = 4
        # Sort triplets by their MMI value
        sorted_triplets = sorted(mmi_values_dict.items(), key=lambda x: x[1])
        neg_triplets = sorted_triplets[:number_of_triplets]
        zero_triplets = sorted(sorted_triplets, key=lambda x: abs(x[1]))[
            :number_of_triplets
        ]
        pos_triplets = sorted_triplets[-number_of_triplets:]
        selected_triplets = {
            "negative": list(neg_triplets),
            "zero": list(zero_triplets),
            "positive": list(pos_triplets),
        }
        # print("Selected triplets for MMI analysis:", selected_triplets)

        all_triplets = (
            selected_triplets["negative"]
            + selected_triplets["zero"]
            + selected_triplets["positive"]
        )

        all_calculation_dict = {}
        for bin_num in BIN_NUM_LIST:
            edges = linspace(global_min, global_max, bin_num + 1)

            partial_func = partial(entropy_calculation, df=histone_df, edges=edges)
            entropy_results = smart_map(partial_func, histone_mods)
            entropy_dict = dict(entropy_results)
            print(entropy_dict)
            all_calculation_dict.setdefault(bin_num, {})["entropy"] = entropy_dict
            # break

            # =======================================================
            # joint entropy dict for all bin numbers
            # =======================================================
            # joint duplets from triplets
            duplets = []
            for triplet, _ in all_triplets:
                triplet = triplet.split("_")
                duplet_combinations = list(combinations(triplet, 2))
                duplets.extend(duplet_combinations)

            duplet_keys = ["_".join(duplet) for duplet in duplets]
            # print(duplet_keys)

            partial_func_duplet = partial(
                joint_entropy_calculation, edges=edges, df=histone_df
            )
            duplet_results = smart_map(
                partial_func_duplet, duplets, mode="auto", max_workers=max_workers
            )
            print(duplet_results)
            joint_entropy_dict = {
                "_".join(duplet): result[1]
                for duplet, result in zip(duplets, duplet_results)
                if duplet == result[0]
            }
            print(joint_entropy_dict)
            all_calculation_dict.setdefault(bin_num, {})[
                "joint_entropy"
            ] = joint_entropy_dict
            # break

            # # ========================================================
            # two  mi calculation dict for all bin numbers
            # # ========================================================

            two_mi_partial_func = partial(
                two_mi_calculation, edges=edges, df=histone_df
            )
            two_mi_results = smart_map(
                two_mi_partial_func, duplets[:], mode="auto", max_workers=max_workers
            )
            two_mi_dict = {
                "_".join(duplet): result[1]
                for duplet, result in zip(duplets, two_mi_results)
                if duplet == result[0]
            }
            print(two_mi_dict)
            all_calculation_dict.setdefault(bin_num, {})["two_mi"] = two_mi_dict
            # break

            # # ========================================================
            # three variable joint entropy dict for all bin numbers
            # # ========================================================
            triplets = [i for i, _ in all_triplets]
            triplet_partial_func = partial(
                triplet_joint_entropy_calculation, edges=edges, df=histone_df
            )

            triplet_results = smart_map(
                triplet_partial_func, triplets, mode="auto", max_workers=max_workers
            )
            triplet_joint_entropy_dict = dict(triplet_results)
            print(triplet_joint_entropy_dict)
            all_calculation_dict.setdefault(bin_num, {})[
                "three_var_joint_entropy"
            ] = triplet_joint_entropy_dict
            # break

            # # ========================================================
            # three variable mi calculation dict for all bin numbers
            # # ========================================================
            three_var_mi_partial_func = partial(
                three_var_mi_calculation, edges=edges, df=histone_df
            )
            three_var_mi_results = smart_map(
                three_var_mi_partial_func,
                triplets,
                mode="auto",
                max_workers=max_workers,
            )
            three_var_mi_dict = dict(three_var_mi_results)
            print(three_var_mi_dict)
            all_calculation_dict.setdefault(bin_num, {})[
                "three_var_mi"
            ] = three_var_mi_dict
            # break

        # =======================================================
        # saving all calculation dict
        # =======================================================

        # including triplets names with mmi values
        all_calculation_dict["selected_triplets"] = all_triplets
        output_dir = os.path.join(
            ProjectPaths().get_output_files_dir(),
            "mmi_value_analysis",
            "entropy_mi_calculation_bin_based",
            organism,
        )
        dir_maker(output_dir)
        output_file = os.path.join(
            output_dir, f"all_entropy_mi_calculations_with_different_bin_num.json"
        )
        json_file_saver(all_calculation_dict, output_file)
