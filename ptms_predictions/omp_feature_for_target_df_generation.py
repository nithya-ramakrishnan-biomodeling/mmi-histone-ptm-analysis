import json 
import os 
import pandas as pd 
import notebook__init__b
from datahandler import data_handler
from path_utils.path_handler import ProjectPaths

def json_file_loader(file_path):

    """
    Load a JSON file and return its contents as a dictionary.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The contents of the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

organism = "human"
cleaned_dir = ProjectPaths.get_cleaned_data_dir()
histone_mod_file_path = os.path.join(cleaned_dir, organism, f"{organism}_zero_mean.csv")
histone_mod_df = data_handler.csv_loader(histone_mod_file_path)

josn_file_path = r"C:\Users\dhana\OneDrive\Desktop\multivariate_ptms_analysis-\output\files\ptm_prediction\human\omp_xgb_sorted_8_feature_importance\eight_feature_summary.json"
histone_feature_dict = json_file_loader(josn_file_path)

histone_mod_order = list(histone_mod_df.columns)


# creating empty dataframe to store the feature presence of each histone
all_histonefeauter_df = pd.DataFrame(columns=histone_mod_order, index=histone_mod_order)

for histone in histone_mod_order:
    # getting feature names 
    feature_names = list(histone_feature_dict[histone].keys())

    # creating a dictionary to store the feature presence of each histone
    histone_feature_presence = dict.fromkeys(histone_mod_order, 0)

    for feature in feature_names:
        histone_feature_presence[feature] = 1

    all_histonefeauter_df.loc[histone] = histone_feature_presence

# saving the all_histone feature presence dictionary into csv 

output_dir_full_path = os.path.join(
    ProjectPaths.get_output_files_dir(), "ptm_prediction", organism, "omp_xgb_sorted_8_feature_importance"
)   
output_file_path = os.path.join(output_dir_full_path, "histone_feature_presence.csv")

histone_mod_presenecedf = all_histonefeauter_df.astype(int)
print(histone_mod_presenecedf)
histone_mod_presenecedf.to_csv(output_file_path, index=False)




