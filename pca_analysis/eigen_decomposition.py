# Importing required libraries
import os
import numpy as np
import pandas as pd
import argparse
import sys
import notebook__init__b
from datahandler import data_handler, dir_handler
from pca_utils.model import pca_analyzer
from path_utils.path_handler import ProjectPaths

# Define the argument parser
parser = argparse.ArgumentParser(description="Pass variables as command-line arguments")

# Add the arguments (with short and long names)
parser.add_argument(
    "-o",
    "--organism",
    type=str,
    required=True,
    help="The organism",
)

# Parse the arguments
args = parser.parse_args()

# Access the variable values
organism = args.organism
out_folder_name = r"pca"

# Get the directory details
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_file_dir = os.path.join(
    ProjectPaths.get_output_files_dir(), out_folder_name, organism
)
output_img_dir = os.path.join(
    ProjectPaths.get_output_images_dir(), out_folder_name, organism
)
# Create output directories if they don't exist
dir_handler.dir_maker(output_file_dir)
dir_handler.dir_maker(output_img_dir)
input_filepath = input_file_path = os.path.join(
    cleaned_dir, organism, f"{organism}_zero_mean.csv"
)

# Load the data
histone_df = data_handler.csv_loader(input_file_path)

# Perform covariance-based PCA
pca_cov = pca_analyzer.PCAanalyzer(histone_df, method="cov")
cov_df = pd.DataFrame(
    pca_cov.dist, columns=histone_df.columns, index=histone_df.columns
)
cov_df.to_csv(os.path.join(output_file_dir, "cov_matrix.csv"))
# Define bin allocation for MI-based PCA based on the organism
if organism == "human":
    global_max = 40
    global_min = -33
    bin_num = 130
    bin_edges = np.linspace(global_min, global_max, bin_num)

if organism == "yeast":
    global_max = 3.54
    gloabl_min = -6
    bin_num = 90
    bin_edges = np.linspace(gloabl_min, global_max, bin_num)

# Perform MI-based PCA
pca_mi = pca_analyzer.PCAanalyzer(histone_df, method="mi", bin_edges=bin_edges)
mi_df = pd.DataFrame(pca_mi.dist, columns=histone_df.columns, index=histone_df.columns)
mi_df.to_csv(os.path.join(output_file_dir, "mi_matrix.csv"))

# Calculate eigen vectors for covariance-based PCA
cov_eig_vec = pca_cov.eigen_vectors.T
cov_eig_vec_df = pd.DataFrame(cov_eig_vec, columns=histone_df.columns)
cov_eig_vec_df.head()

# Calculate eigen vectors for MI-based PCA
mi_eig_vec = pca_mi.eigen_vectors.T
mi_eig_vec_df = pd.DataFrame(mi_eig_vec, columns=histone_df.columns)
mi_eig_vec_df.head()

# Visualization of cumulative sum plots for both PCA methods
pca_cov.cumsum_plot(
    fig_sav=True,
    out_dir=output_img_dir,
    indvid_label="Variance",
    cumulative_label="variance",
)
pca_mi.cumsum_plot(
    fig_sav=True,
    out_dir=output_img_dir,
    indvid_label="Information",
    cumulative_label="information",
)

# Save the eigen vectors and eigen values to files
cov_eig_values = pca_cov.eigen_values
mi_eig_values = pca_mi.eigen_values

# Save covariance-based PCA results
np.save(os.path.join(output_file_dir, "cov_eig_value.npy"), cov_eig_values)
cov_eig_vec_df.to_csv(os.path.join(output_file_dir, "cov_eigen_vec.csv"))

# Save MI-based PCA results
np.save(os.path.join(output_file_dir, "mi_egen_value.npy"), mi_eig_values)
mi_eig_vec_df.to_csv(os.path.join(output_file_dir, "mi_eigen_vec.csv"))
