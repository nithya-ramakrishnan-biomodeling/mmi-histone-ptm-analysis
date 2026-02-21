from pathlib import Path
from multivariate_utils import ProjectPaths, load_data


# File paths for cleaned data
YEAST_FILE_PATH = Path.joinpath(
    ProjectPaths().get_cleaned_data_dir(), "yeast", "yeast_zero_mean.csv"
)
HUMAN_FILE_PATH = Path.joinpath(
    ProjectPaths().get_cleaned_data_dir(), "human", "human_zero_mean.csv"
)

# bin number and global min/max for MI calculation
MI_CONFIG_PATH = Path.joinpath(
    ProjectPaths().get_cleaned_data_dir(), "mi_bin_info.json"
)
MI_CONFIG = load_data(MI_CONFIG_PATH)


# ========================================================
# organism and bin number list
# ======================================================

ORGANISMS = ["yeast", "human"]
BIN_NUM_LIST = list(range(40, 151, 10))


def file_globl_max_min_config(organism):
    if organism == "yeast":
        global_min = MI_CONFIG["yeast"]["global_min"]
        global_max = MI_CONFIG["yeast"]["global_max"]
        file_path = YEAST_FILE_PATH
    elif organism == "human":
        global_min = MI_CONFIG["human"]["global_min"]
        global_max = MI_CONFIG["human"]["global_max"]
        file_path = HUMAN_FILE_PATH
    else:
        raise ValueError(
            f"Organism '{organism}' not recognized. Choose 'yeast' or 'human'."
        )
    return file_path, global_min, global_max

