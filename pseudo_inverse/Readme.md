## Dir: pseudo_inverse
  > Note: This directory performs pseudo-inverse matrix analysis on histone modifications to identify key features and make R² score calculations.

### Contents
```
pseudo_inverse/
├── Readme.md                  # Directory documentation
└── r2_score_calcn_pseudo_inverse_based_feature.py     # Feature selection and R² calculation
```

## Overview

The pseudo-inverse method analyzes histone modifications by solving y = Ax where:
- y: Target histone modification level
- A: Matrix of other modifications 
- x: Solution showing feature importance

Key steps:
1. Calculate feature influence coefficients
2. Select top features
3. Compute R² score

## Usage

Run the analysis:
```sh
python r2_score_calcn_pseudo_inverse_based_feature.py
```

## Outputs

Creates two JSON files:

1. `{organism}_pseudo_inverse_with_r2.json`
   - Features and R² scores per target
   - Format: `{"target": {"features": [...], "r_squared": value}}`

2. `{organism}_histone_r2_values.json` 
   - R² scores per target
   - Format: `{"target": r2_value}`

## Config

- n: Features to select (default: 8)
- organism: Species to analyze (default: "human")
