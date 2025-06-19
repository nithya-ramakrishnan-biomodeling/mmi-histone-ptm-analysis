# Multivariate PTMs Analysis

This repository contains a comprehensive pipeline for analyzing post-translational modifications (PTMs) using multivariate analysis methods, including mutual information analysis, PCA, machine learning predictions, and pseudo-inverse analysis.

---

## Directory Structure

- [`data/`](./data/): Raw and processed data for human and yeast
- [`datahandler/`](./datahandler/): Data loading, processing, and file operations
- [`mi_btw_yhat_and_y/`](./mi_btw_yhat_and_y/): MI between predicted and actual values
- [`multivariate_mi_analysis/`](./multivariate_mi_analysis/): Multivariate MI analysis and entropy calculations
- [`output/`](./output/): Output files, images, and analysis results
- [`path_utils/`](./path_utils/): Project path management and configuration
- [`pca_analysis/`](./pca_analysis/): Principal component analysis scripts
- [`pseudo_inverse/`](./pseudo_inverse/): Pseudo-inverse based feature selection and analysis
- [`ptms_predictions/`](./ptms_predictions/): PTM prediction using XGBoost and OMP
- [`main.py`](./main.py): Main pipeline orchestrator

---

## Pipeline Overview

This project implements a five-stage analysis pipeline for histone post-translational modifications:

1. **PCA Analysis**  
   Principal component analysis with eigen decomposition.

2. **Multivariate MI Analysis**  
   Mutual information calculations, redundancy removal, and entropy optimization.

3. **PTM Predictions**  
   Machine learning predictions using XGBoost and feature selection.

4. **Pseudo-Inverse Analysis**  
   Matrix-based feature importance and prediction analysis.

5. **MI Between Y-hat and Y**  
   Quality assessment of predictions using mutual information.

---

## Getting Started

### Prerequisites

- Python 3.6+
- Recommended: Use a virtual environment

### Setup

#### Create and Activate a Virtual Environment

- On Windows:
    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```
- On macOS and Linux:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

#### Install Dependencies

```sh
pip install -r requirements.txt
```

---

## Running the Pipeline

### Run the Complete Pipeline

```sh
python main.py
```

#### Additional Options

```sh
# Run with parallel processing
python main.py -p

# Run specific modules only (e.g., PCA and multivariate analysis)
python main.py -m pca multivariate

# Stop execution if any module fails
python main.py -s

# Show verbose output
python main.py -v
```

### Run Individual Modules

```sh
# 1. PCA Analysis
python pca_analysis/pca_main_run.py -o human -p

# 2. Multivariate MI Analysis
python multivariate_mi_analysis/multivariate_main_run.py -o yeast -p

# 3. PTM Predictions
python ptms_predictions/ptm_main_run.py -p

# 4. Pseudo-Inverse Analysis
python pseudo_inverse/pseudo_invrs_main_run.py -p

# 5. MI Between Y-hat and Y
python mi_btw_yhat_and_y/mi_yhat_y_main_run.py -p
```

### Special Analysis: Bin-based Entropy Calculation

```sh
python multivariate_mi_analysis/bin_num_based_entropy_calculation.py -o human yeast -g
```

---

## Output Structure

Results are organized in the [`output/`](./output/) directory:

```
output/
├── files/
│   ├── pca_analysis/
│   ├── multivariate_analysis/
│   ├── ptms_predictions/
│   ├── pseudo_inverse/
│   └── mi_btw_yhat_and_y/
└── images/
    ├── pca_analysis/
    ├── multivariate_analysis/
    ├── ptms_predictions/
    ├── pseudo_inverse/
    └── mi_btw_yhat_and_y/
```

---

## Configuration

- **Organisms:** Human and yeast
- **MI Methods:** Pairwise and three-way mutual information
- **Prediction Models:** XGBoost with hyperparameter optimization
- **Feature Selection:** Orthogonal matching pursuit and pseudo-inverse methods
- **Cutoff Values:** Organism-specific thresholds for MI calculations

---

## Troubleshooting

- **Memory Errors:** Use the `--memory_limit` parameter to constrain memory usage
- **Unicode Errors on Windows:** Scripts use ASCII-compatible output formatting
- **Missing Dependencies:** Ensure all packages in `requirements.txt` are installed

### Performance Optimization

- Use `-p` flag for parallel processing
- Adjust `--cpu_limit` parameter based on your system
- For large datasets, consider running modules individually

---

## Support

For questions or issues, check the individual script help messages using the `-h`