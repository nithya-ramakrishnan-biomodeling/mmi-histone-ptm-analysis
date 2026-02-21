## Project: Multivariate PTM Analysis
> Note: This project implements a comprehensive multivariate analysis framework for Post-Translational Modifications (PTMs) of histone proteins, using mutual information, PCA, regression, and other statistical methods to uncover co-modification patterns.

----------
### Tree
- [multivariate_ptm_analysis/](./)
  - [Readme.md](./Readme.md) - Main project readme
  - [00_mmi_pipeline/](./00_mmi_pipeline/) - Full MMI pipeline (preprocessing → MMI → feature selection → regression)
  - [01_pca_analysis/](./01_pca_analysis/) - PCA analysis using covariance and mutual information matrices
  - [02_mmi_analysis/](./02_mmi_analysis/) - Multivariate Mutual Information analysis
  - [03_regression_prediction_analysis/](./03_regression_prediction_analysis/) - Regression-based histone modification prediction
  - [04_pseudo_inverse_analysis/](./04_pseudo_inverse_analysis/) - Pseudo-inverse based analysis
  - [05_total_correlation_analysis/](./05_total_correlation_analysis/) - Total correlation analysis
  - [06_data_fit_analysis/](./06_data_fit_analysis/) - Data distribution fitting analysis
  - [07_review_analysis/](./07_review_analysis/) - Review and summary analysis
  - [multivariate_utils/](./multivariate_utils/) - Shared utility functions and classes
  - [data/](./data/) - Input and processed data files

----------

## Overview

This project provides a complete analytical framework for studying histone Post-Translational Modifications (PTMs) across organisms (e.g., yeast, human). It uses information-theoretic and statistical methods to identify co-modification patterns, select informative features, and predict histone modification states.

### Key Methods
- **Multivariate Mutual Information (MMI)** - Identifies synergistic/redundant relationships among histone modifications
- **Principal Component Analysis (PCA)** - Reduces dimensionality using covariance and MI-based approaches
- **Regression Prediction** - Predicts histone modification levels from selected features
- **Pseudo-Inverse Analysis** - Solves underdetermined systems for modification relationships
- **Total Correlation** - Measures multivariate statistical dependence
- **Data Fit Analysis** - Evaluates distributional assumptions of the data

----------

## Setup

### 1. Create and Activate Virtual Environment
```sh
# Create virtual environment in the root directory
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 2. Install `multivariate_utils` in Editable Mode
```sh
# From the root directory of the project
pip install -e ./multivariate_utils
```
> **Note:** Installing in editable mode (`-e`) allows changes to `multivariate_utils` to be reflected immediately without reinstalling.

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

----------

## Modules

### [`00_mmi_pipeline/`](./00_mmi_pipeline/)
End-to-end pipeline for MMI-based analysis:
1. Data preprocessing and normalization
2. MMI calculation for all histone triplets
3. Feature selection based on MMI thresholds
4. Regression prediction using selected features

### [`01_pca_analysis/`](./01_pca_analysis/)
PCA of histone modification data using:
- Traditional covariance-based PCA
- Mutual Information-based PCA

### [`02_mmi_analysis/`](./02_mmi_analysis/)
Standalone MMI analysis scripts for computing and visualizing multivariate mutual information across histone features.

### [`03_regression_prediction_analysis/`](./03_regression_prediction_analysis/)
Regression models for predicting histone modifications using cross-validation and R² scoring.

### [`04_pseudo_inverse_analysis/`](./04_pseudo_inverse_analysis/)
Pseudo-inverse based approach to analyze relationships between histone modifications.

### [`05_total_correlation_analysis/`](./05_total_correlation_analysis/)
Total correlation analysis to measure overall statistical dependence among histone modifications.

### [`06_data_fit_analysis/`](./06_data_fit_analysis/)
Analyzes how well the histone modification data fits assumed statistical distributions.

### [`multivariate_utils/`](./multivariate_utils/)
Shared utilities used across all modules:
- Data loading and saving
- Parallel processing helpers
- Logging utilities
- Project path management
- Statistical computation tools

----------

## Dependencies

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `click`
- `PyYAML`
- `matplotlib`
- `multivariate_utils` (internal)

----------

## Data

Input data should be placed in the `data/` directory:
```
data/
├── raw/                  - Raw histone modification data
└── processed/            - Preprocessed data ready for analysis
    ├── yeast/            - Yeast-specific processed data
    ├── human/            - Human-specific processed data
    └── mi_bin_info.json  - Binning information for MMI calculation
```

----------

## Notes
- Results are saved in the respective module output directories.
- See individual module `README.md` files for detailed usage.