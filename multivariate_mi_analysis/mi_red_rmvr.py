# Note: this python file used for the removing the redundant three variable mi values and
# split the data frame with different MI range values

# Importing the required libraries
import os
import pandas as pd
import argparse
from itertools import combinations
import notebook__init__b
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths

out_folder_name = "multivariate analysis"
# organism = r"yeast"

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


# getting the directory details
main_dir = ProjectPaths.get_project_root()
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
output_file_dir = os.path.join(
    ProjectPaths.get_output_files_dir(), out_folder_name, organism
)
output_img_dir = os.path.join(
    ProjectPaths.get_output_images_dir(), out_folder_name, organism
)
dir_handler.dir_maker(output_file_dir)
dir_handler.dir_maker(output_img_dir)

# three variable mutual information dataframe path.
input_file_name = r"diff_uy_and_uy_givenv_entro.csv"
input_file_path = os.path.join(output_file_dir, input_file_name)
# print(f"Input File path: {input_file_path}")
three_mi_df = data_handler.csv_loader(input_file_path)

# creating unique three variable combinations.
unque_three_var_comb = list(combinations(three_mi_df.columns, r=3))

# mask the dataframe with unique combinations only
df_masked = pd.DataFrame(
    index=[i.strip() for i in three_mi_df.index], columns=three_mi_df.columns
)
for histone_mods in unque_three_var_comb:
    col, indx = histone_mods[0], histone_mods[1] + "_" + histone_mods[2]
    df_masked.loc[indx, col] = three_mi_df.loc[indx, col]


# dropping the rows in the pandas Dataframe where all the column values are Nan
df_masked = df_masked.dropna(axis=0, how="all")

# saving masked dataframe
df_masked.to_csv(os.path.join(output_file_dir, f"masked_{organism}_three_mi.csv"))
