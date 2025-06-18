## Dir: ptms_predictions 
  > Note: This directory contains scripts for predicting post-translational modifications (PTMs), implementing various machine learning approaches, multivariate mutual information (MMI) analysis, and other prediction techniques.
----------
### Tree 
- [ptms_predictions/](./)
  - [Readme.md](./Readme.md)  - Readme file for the ptms_predictions directory
  - [3mmi_feature_extractor_with_mmicutoff.py](./3mmi_feature_extractor_with_mmicutoff.py)  - Extracts features based on 3MMI analysis with cutoff values
  - [r2_prediction_all_remain_features.py](./r2_prediction_all_remain_features.py)  - Predicts each histone using all others as features
  - [r2_prediction_individual_cumulative.py](./r2_prediction_individual_cumulative.py)  - Individual and cumulative feature analysis for specific targets
  - [r2_prediction_mmi_and_highest_corr.py](./r2_prediction_mmi_and_highest_corr.py)  - Prediction using MMI and highest correlated features
  - [r2_prediction_mmi_features.py](./r2_prediction_mmi_features.py)  - Prediction using MMI-selected features
  - [r2_prediction_omp_based_xgb_n_number_feature.py](./r2_prediction_omp_based_xgb_n_number_feature.py)  - OMP feature selection with XGBoost regression
  - [r2_prediction_PCA_transformed_features.py](./r2_prediction_PCA_transformed_features.py)  - Prediction using PCA-transformed features
  - [r2_prediction_pseudo_inverse_based_feature.py](./r2_prediction_pseudo_inverse_based_feature.py)  - Feature selection using pseudo-inverse
  - [ptm_main_run.py](./ptm_main_run.py)  - Main script for running the entire prediction pipeline
  - [ptms_utils/](./ptms_utils/)
    - [data/](./ptms_utils/data/)  - Data handling utilities
    - [model/](./ptms_utils/model/)  - Machine learning model implementations
      - [histone_mod_predictor.py](./ptms_utils/model/histone_mod_predictor.py)  - Python script for predicting histone modifications
    - [visualization/](./ptms_utils/visualization/)  - Visualization tools for analysis results

## Running the Main Script

To run the main script for predicting post-translational modifications (PTMs), execute the following command:

```sh
# Run in sequential mode (default)
python ptm_main_run.py

# Run in parallel mode (faster, recommended for multi-core systems)
python ptm_main_run.py -p
```

## Script Details

### 3MMI Feature Extraction

```sh
python 3mmi_feature_extractor_with_mmicutoff.py -o [organism] -op [operator]
```

### MMI Feature Prediction

```sh
python r2_prediction_mmi_features.py -o [organism] -c [cutoff] -op [operator] [-p]
```

### MMI and Highest Correlation

```sh
python r2_prediction_mmi_and_highest_corr.py -o [organism] -c [cutoff] -op [operator] [-p]
```

### All Remaining Features

```sh
python r2_prediction_all_remain_features.py -o [organism] [-p]
```

### PCA Transformed Features

```sh
python r2_prediction_PCA_transformed_features.py -o [organism] -m [method] [-p] [-j jobs]
```

### Pseudo-Inverse Based Feature Selection

```sh
python r2_prediction_pseudo_inverse_based_feature.py -o [organism] [-p]
```

### OMP XGBoost Feature Selection

```sh
python r2_prediction_omp_based_xgb_n_number_feature.py -o [organism] -n [features] [-p]
```

### Individual Cumulative Analysis

```sh
python r2_prediction_individual_cumulative.py -o [organism] -t [target] [--cpu_limit limit]
```

## Common Parameters

- **organism**: Target organism (yeast/human)
- **cutoff**: Threshold value for MMI analysis
- **operator**: Comparison operator (le/ge)
- **-p**: Enable parallel processing
- **method**: Analysis method (mi/cov for PCA)
- **features**: Number of features to select (default: 8)
- **target**: Target histone modification (e.g., H3K9me3, H3K79me3)
- **--cpu_limit**: Limit number of CPU cores used in parallel processing

## Output Files

Results are saved in the output directory with the following standardized structure:
```
output/ptm_prediction/[organism]/[analysis_type]/[results.json]
```

Each script produces JSON files with R² scores and supporting information following consistent naming conventions:
- `[organism]_mmi_cutoff_[value]_[operator]_r2.json`
- `[organism]_mmi_highest_corr_cutoff_[value]_r2.json`
- `[organism]_all_remaining_feature_r2.json`
- `[organism]_[method]_pca_r2.json`
- `[organism]_omp_[n_features]_features_r2.json`
- `[organism]_[histone]_individual_r2.json`