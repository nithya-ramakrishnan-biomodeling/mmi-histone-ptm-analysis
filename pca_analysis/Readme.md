## Dir: pca_analysis 
  > Note: This directory contains scripts for principal component analysis (PCA), including eigen decomposition, main run scripts, and utility functions.
----------
### Tree 
- [pca_analysis/](./)
  - [Readme.md](./Readme.md)  - Readme file for the pca_analysis directory
  - [eigen_decomposition.py](./eigen_decomposition.py)  - Python script for eigen decomposition
  - [pca_components.py](./pca_components.py)  - Python script for generating PCA-transformed features
  - [checking_correctness_of_pca_calc.py](./checking_correctness_of_pca_calc.py)  - Python script for PCA validation
  - [pca_main_run.py](./pca_main_run.py)  - Main script for running PCA pipeline
  - [pca_utils/](./pca_utils/)
    - [data/](./pca_utils/data/)  - Data handling utilities
    - [model/](./pca_utils/model/)  - Model implementations
      - [histone_mod_predictor.py](./pca_utils/model/histone_mod_predictor.py)  - Python script for predicting histone modifications
      - [mi_calculator.py](./pca_utils/model/mi_calculator.py)  - Python script for mutual information calculation
      - [pca_analyzer.py](./pca_utils/model/pca_analyzer.py)  - Python script for PCA analysis
    - [visualization/](./pca_utils/visualization/)  - Visualization tools
      - [pca_plotter.py](./pca_utils/visualization/pca_plotter.py)  - Python script for PCA visualization

## Overview

The PCA Analysis module implements two different approaches to Principal Component Analysis of histone modification data:

1. **Traditional Covariance-based PCA** - Captures linear relationships between histone modifications
2. **Mutual Information (MI) based PCA** - Captures both linear and non-linear relationships

These analyses help identify the most important patterns of variation in epigenetic data and can reveal underlying biological mechanisms that govern histone modification co-occurrence.

## Main Components

### Eigen Decomposition

The `eigen_decomposition.py` script performs eigenvalue decomposition using both covariance and mutual information matrices, extracting the principal components that explain the most variance in the data.

### PCA Component Generation

The `pca_components.py` script creates transformed PCA features from the original histone modification data, which can be used for downstream analysis or visualization.

### Validation

The `checking_correctness_of_pca_calc.py` script implements a leave-one-out approach to verify the mathematical correctness and robustness of the PCA implementation.

## Running the Main Script

To run the complete PCA pipeline, execute the main script with optional parameters:

```sh
# Run with default parameters (all organisms, 5 components)
python pca_main_run.py

# Run for specific organisms
python pca_main_run.py -o yeast human

# Run with custom number of components
python pca_main_run.py -c 8

# Run with specific organisms and custom components
python pca_main_run.py -o yeast -c 3
```

## Script Details

### Eigen Decomposition

```sh
python eigen_decomposition.py -o [organism]
```

Generates eigenvalues and eigenvectors for both covariance and mutual information matrices.

### PCA Components

```sh
python pca_components.py -o [organism] -m [method] -n [components]
```

Parameters:
- **organism**: Target organism (default: "human")
- **method**: PCA method, either "cov" or "mi" (default: "cov")
- **components**: Number of principal components to generate (default: 5)

### PCA Correctness Check

```sh
python checking_correctness_of_pca_calc.py
```

Performs leave-one-out analysis to validate PCA calculations.

## Output Files

The scripts generate several output files:

1. **Matrices**:
   - `{organism}_cov_mat.csv` - Covariance matrix
   - `{organism}_mi_mat.csv` - Mutual information matrix

2. **Eigen Analysis**:
   - `{organism}_cov_eig_vectors.csv` - Eigenvectors from covariance PCA
   - `{organism}_mi_eig_vectors.csv` - Eigenvectors from MI-based PCA
   - `{organism}_cov_eig_value.npy` - Eigenvalues from covariance PCA
   - `{organism}_mi_eig_value.npy` - Eigenvalues from MI-based PCA

3. **Transformed Data**:
   - `{organism}_histone_{method}_with_eig_transf_features.csv` - Original data augmented with PCA features

4. **Visualization**:
   - `{organism}_{method}_cum_sum.jpg` - Cumulative variance/information plot
   - `{organism}_{method}_cum_sum.eps` - Publication-quality version of the same plot
   - `{organism}_{method}_cum_sum.png` - Transparent PNG version for presentations

