## Dir: 00_mmi_pipeline
  > Note: This directory contains scripts for the MMI (Multivariate Mutual Information) pipeline, including data preprocessing, MMI calculation, feature selection, and regression prediction.

----------
### Tree
- [00_mmi_pipeline/](./)
  - [Readme.md](./Readme.md) - Readme file for the 00_mmi_pipeline directory
  - [00_data_preprocessing.py](./00_data_preprocessing.py) - Python script for data preprocessing
  - [01_mmi_calculation.py](./01_mmi_calculation.py) - Python script for MMI calculation
  - [02_feature_selection.py](./02_feature_selection.py) - Python script for feature selection based on MMI
  - [03_regression_prediction.py](./03_regression_prediction.py) - Python script for regression prediction
  - [config_loader.py](./config_loader.py) - Python script for loading pipeline configuration
  - [mmi_main.py](./mmi_main.py) - Main script for running the MMI pipeline
  - [pipeline_config.yaml](./pipeline_config.yaml) - YAML configuration file for the pipeline

## Overview

The MMI Pipeline module implements a complete workflow for analyzing histone modification data using Multivariate Mutual Information (MMI). The pipeline consists of four sequential steps:

1. **Data Preprocessing** - Cleans and normalizes raw histone modification data
2. **MMI Calculation** - Computes MMI values for all histone feature triplets
3. **Feature Selection** - Selects relevant features based on MMI thresholds
4. **Regression Prediction** - Predicts histone modifications using selected features

## Main Components

### Data Preprocessing
The `00_data_preprocessing.py` script handles raw input data, performs missing value imputation, and normalizes the data for downstream analysis.

### MMI Calculation
The `01_mmi_calculation.py` script computes MMI values for all possible triplets of histone features using parallel processing. Results are saved as JSON files per bin number.

### Feature Selection
The `02_feature_selection.py` script applies MMI threshold-based feature selection to identify the most informative histone modifications.

### Regression Prediction
The `03_regression_prediction.py` script uses the selected features to train regression models and predict target histone modifications using cross-validation.

## Running the Pipeline

### Run Full Pipeline
```sh
# Run with default parameters
python mmi_main.py

# Run with custom configuration
python mmi_main.py --config pipeline_config.yaml
```

## Script Details

### Data Preprocessing
```sh
python 00_data_preprocessing.py \
    --file-path <path/to/input.csv> \
    --output-path <path/to/output.csv>
```
Parameters:
- **file-path** (`-f`): Path to the raw input data file (CSV format)
- **output-path** (`-o`): Path to save the preprocessed data (optional)

---

### MMI Calculation
```sh
python 01_mmi_calculation.py \
    --file-path <path/to/input.csv> \
    --n-bins <number_of_bins> \
    --global-min <min_value> \
    --global-max <max_value> \
    --output-path <path/to/output.json>
```
Parameters:
- **file-path** (`-f`): Path to the input data file (CSV format)
- **n-bins** (`-b`): Number of bins for MMI calculation
- **global-min** (`-min`): Global minimum value for binning (optional)
- **global-max** (`-max`): Global maximum value for binning (optional)
- **output-path** (`-o`): Path to save the output JSON file (optional)

---

### Feature Selection
```sh
python 02_feature_selection.py \
    --mmi-file <path/to/mmi_results.json> \
    --threshold <mmi_threshold> \
    --output-path <path/to/output.json>
```
Parameters:
- **mmi-file** (`-m`): Path to the MMI results JSON file
- **threshold** (`-t`): MMI threshold for feature selection
- **output-path** (`-o`): Path to save selected features (optional)

---

### Regression Prediction
```sh
python 03_regression_prediction.py \
    --feature-file <path/to/feature_data.csv> \
    --features-file <path/to/features.json> \
    --threshold <mmi_threshold> \
    --output-path <path/to/output.json> \
    --n-jobs <number_of_jobs>
```
Parameters:
- **feature-file** (`-f`): Path to the input feature data file (CSV format)
- **features-file** (`-m`): Path to the JSON file containing selected features
- **threshold** (`-t`): MMI threshold used for selecting features
- **output-path** (`-o`): Path to save the output files (optional)
- **n-jobs** (`-n`): Number of parallel jobs for regression (default: 1)

## Output Files

The pipeline generates the following output files:

1. **Preprocessing**:
   - `{organism}_preprocessed.csv` - Cleaned and normalized data

2. **MMI Calculation**:
   - `histone_triplet_mmi_bin{n_bins}.json` - MMI values for all histone triplets per bin number

3. **Feature Selection**:
   - `mmi_features_threshold_{threshold}.json` - Selected features per MMI threshold

4. **Regression Prediction**:
   - `prediction_results.json` - R² scores for each target histone modification
   - `cv_results_{target}.csv` - Cross-validation results per target

## Configuration

The pipeline can be configured using `pipeline_config.yaml`:

```yaml
# Example pipeline_config.yaml
data:
  --input_path: data/processed/yeast/yeast_zero_mean.csv

mmi:
  --global_min: -3.0
  --global_max: 3.0
  --n_bins: 70

feature_selection:
  --threshold: -0.4 #mmi threshold for feature selection

regression:
  --n_jobs: 4
```

## Dependencies

- `multivariate_utils` (for data handling, MMI calculation, regression, logging, etc.)
- `click`
- `numpy`
- `pandas`
- `pathlib`
- `PyYAML`