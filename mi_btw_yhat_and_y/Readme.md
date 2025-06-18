## Dir: mi_btw_yhat_and_y
  > Note: This directory contains scripts for calculating mutual information between predicted (yhat) and actual (y) values of histone modifications.
----------
### Tree
- [mi_btw_yhat_and_y/](./)
  - [Readme.md](./Readme.md) - This documentation file
  - [mi_yhat_y_main_run.py](./mi_yhat_y_main_run.py) - Main script to orchestrate MI calculations
  - [mi_calculation_mmi_feature_cutoffs.py](./mi_calculation_mmi_feature_cutoffs.py) - Calculates MI for MMI-selected features
  - [mi_calculation_all_remain_features.py](./mi_calculation_all_remain_features.py) - Calculates MI for all-remaining features

## Overview

The mutual information (MI) analysis scripts quantify the information shared between predicted histone modification values and actual values. Unlike R² which only captures linear relationships, mutual information captures both linear and non-linear relationships between variables.

These scripts enable:
1. Calculating MI for predictions made using MMI-selected features
2. Calculating MI for predictions made using all-remaining features
3. Comparing prediction quality across different feature selection approaches

## Running the Scripts

### Main Script

To run the MI calculation pipeline:

```sh
python mi_yhat_y_main_run.py -o <organism> [-m <method>] [-c <cutoff>] [-op <operator>] [-p]
```

Parameters:
- `-o, --organism`: Target organism (e.g., 'human', 'yeast')
- `-m, --method`: Method to run (choices: 'all', 'mmi', default: 'all')
- `-c, --cutoff`: MMI cutoff value (required for MMI method)
- `-op, --operator`: Operator for MMI cutoff (choices: 'le', 'ge')
- `-p, --parallel`: Enable parallel processing
- `--cpu_limit`: Number of CPU cores to use (default: 2)

### Individual Scripts

#### MMI Features MI Calculation

```sh
python mi_calculation_mmi_feature_cutoffs.py -o <organism> -c <cutoff> -op <operator> [-p]
```

#### All Remaining Features MI Calculation

```sh
python mi_calculation_all_remain_features.py -o <organism> [--cpu_limit limit] [--sequential]
```

## Output Files

Results are saved in the following directory structure:

```
output/
└── mi_values_from_mmi_cutoff/
    └── <organism>/
        └── mi_values_<cutoff>_<operator>.json
```

The JSON files contain target histone names as keys and their MI values as values.

## Example Usage

```sh
# Run both MMI and all-remaining methods for yeast
python mi_yhat_y_main_run.py -o yeast -c -0.44 -op le -p

# Run only MMI method for human
python mi_yhat_y_main_run.py -o human -m mmi -c -0.5 -op ge -p --cpu_limit 4
```