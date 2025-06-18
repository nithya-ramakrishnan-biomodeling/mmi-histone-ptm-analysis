import notebook__init__b
import os
import sys
import pandas as pd
import numpy as np
from datahandler import data_handler
from path_utils.path_handler import ProjectPaths
from pca_utils.model import pca_analyzer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--organism", type=str, default="human", help="The organism")
parser.add_argument("-m", "--method", type=str, default="cov", choices=["cov", "mi"], help="PCA method")
parser.add_argument("-n", "--n_components", type=int, default=5, help="Number of components")
args = parser.parse_args()

organism = args.organism
method = args.method
n_components = args.n_components


out_folder_name = r"pca"
file_name = f"{organism}_zero_mean.csv"
input_file_path = os.path.join(ProjectPaths.get_cleaned_data_dir(), organism, file_name)
histone_df = data_handler.csv_loader(input_file_path)
# print(histone_df.head())


# pca
if method == "cov":
    # covariance based PCAs
    pca = pca_analyzer.PCAanalyzer(
        data=histone_df, method=method, n_components=n_components
    )
elif method == "mi":

    # loading the mutual information binning details
    mi_info = data_handler.json_file_loader(
        os.path.join(ProjectPaths.get_cleaned_data_dir(), r"mi_bin_info.json")
    )
    gloabl_min = mi_info[organism]["global_min"]
    gloabl_max = mi_info[organism]["global_max"]
    bin_num = mi_info[organism]["bin_num"]
    bin_edges = np.linspace(gloabl_min, gloabl_max, bin_num)

    pca = pca_analyzer.PCAanalyzer(
        data=histone_df, method=method, n_components=n_components, bin_edges=bin_edges
    )
else:
    print(f"Unsupported method: {method}")
    sys.exit(1)


cov_pca_eig_vec = pca.eigen_vectors
print(f"The shape of the eigen vectors is: {cov_pca_eig_vec.shape}")

transformed_data = pca.fit_transform()
print(f" The shape of the transformed data is: {transformed_data.shape}")

# transformed dataframe
transformed_histone_df = pd.DataFrame(
    transformed_data, columns=["v1", "v2", "v3", "v4", "v5"]
)
print(transformed_histone_df)

# creating the new data with new additional modifications
new_histone_df = pd.concat([histone_df, transformed_histone_df], axis=1)
print(new_histone_df)

# saving the newly formed dataframe
output_file_path = os.path.join(
    ProjectPaths.get_output_files_dir(),
    out_folder_name,
    organism,
    f"histone_{method}_with_eig_transf_features.csv",
)
new_histone_df.to_csv(output_file_path)
