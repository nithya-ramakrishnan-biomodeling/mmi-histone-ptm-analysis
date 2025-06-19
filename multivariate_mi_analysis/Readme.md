# Multivariate Mutual Information Analysis

> This directory contains a comprehensive suite of scripts for multivariate mutual information analysis, including pairwise and three-way MI calculations, entropy optimization, redundancy removal, and advanced visualization tools for histone PTM data.

---

## Directory Structure

```
multivariate_mi_analysis/
├── bin_num_based_entropy_calculation.py    # Entropy optimization across bin ranges
├── data_mod.py                             # Data modification and filtering utilities
├── entropy_mutualinfo.py                  # Core entropy and MI calculation functions
├── mi_parallel_processing.py               # Parallel MI computation engine
├── mi_red_rmvr.py                         # MI redundancy removal and masking
├── multivariate_main_run.py               # Main pipeline orchestrator
├── multivariate_run.py                    # Core multivariate analysis runner
├── notebook__init__b.py                   # Project environment initialization
├── three_mi_value_splitter.py             # Three-way MI value filtering and visualization
├── visualization.py                       # Basic heatmap and plotting utilities
├── Readme.md                              # This documentation file
└── three_var_mi_utils/                    # Advanced three-variable MI utilities
    ├── data/
    │   └── csv_handler.py                 # CSV data handling and slicing
    ├── notebooks/
    │   └── data_processor.py              # Advanced data processing for three-way MI
    └── visualization/
        ├── heatmap_colors.py              # Scientific color palette management
        ├── heatmap_plotter.py             # Publication-quality heatmap creation
        ├── visulaizer.py                  # Interactive Plotly-based visualizations
        └── visulaizer_mod.py              # Matplotlib-based static visualizations
```

---

## Core Features

### 🔬 **Mutual Information Calculations**
- **Pairwise MI**: Standard mutual information between any two histone modifications
- **Three-way MI**: Conditional mutual information analysis with third variable conditioning
- **Parallel Processing**: High-performance computation using joblib parallelization
- **Bin Optimization**: Automatic entropy calculation across multiple bin configurations

### 📊 **Data Processing & Analysis**
- **Redundancy Removal**: Eliminates redundant three-variable MI combinations
- **Threshold Filtering**: Organism-specific filtering based on MI value ranges
- **Covariance Integration**: Combines MI analysis with correlation and covariance metrics
- **Missing Data Handling**: Robust NaN value management and data validation

### 🎨 **Advanced Visualization**
- **Scientific Color Palettes**: Colorblind-accessible, publication-ready color schemes
- **Interactive Heatmaps**: Plotly-based interactive visualizations with hover information
- **Static Publication Plots**: High-resolution matplotlib/seaborn heatmaps
- **Dynamic Sizing**: Automatic figure size adjustment based on data dimensions

---

## Pipeline Modules

### 1. **Entropy Optimization** (`bin_num_based_entropy_calculation.py`)
Automatically determines optimal bin numbers for discretization by calculating entropy across multiple bin configurations for all histone PTMs.

**Features:**
- Tests bin ranges from 10-500 with configurable steps
- Supports both global and individual min/max value approaches
- Generates comprehensive entropy reports and summaries
- Multi-organism processing with consolidated results

### 2. **Core MI Computation** (`multivariate_run.py`)
Performs the fundamental mutual information calculations using optimized bin configurations.

**Calculations:**
- Two-variable MI: `I(U; Y) = H(U) + H(Y) - H(U,Y)`
- Three-variable MI: `I(U; Y) - I(U; Y|V) = H(U) + H(V) + H(Y) - [H(U,V) + H(V,Y) + H(U,Y)] + H(U,V,Y)`

### 3. **Redundancy Removal** (`mi_red_rmvr.py`)
Eliminates redundant combinations from three-variable MI matrices to focus on unique relationships.

### 4. **Advanced Filtering** (`three_mi_value_splitter.py`)
Splits three-variable MI data based on organism-specific thresholds and generates targeted visualizations.

**Organism-Specific Ranges:**
- **Human**: Fine-grained ranges from -1.0 to +1.0
- **Yeast**: Specialized ranges optimized for yeast MI distributions
- **Custom Thresholds**: User-definable filtering criteria

---

## Usage Instructions

### Run Complete Pipeline
```bash
python multivariate_main_run.py -o human yeast -p
```

**Options:**
- `-o, --organisms`: Specify organisms to analyze (default: human, yeast)
- `-p, --parallel`: Enable parallel processing for faster computation
- `-s, --skip`: Skip specific analysis steps (e.g., 'entropy' for bin optimization)

### Run Individual Modules

#### 1. Entropy Optimization
```bash
python bin_num_based_entropy_calculation.py -o human yeast -g -b 10 -e 500 -s 10
```
- `-g, --global_minmax`: Use global min/max values across all features
- `-b, --start_bin`: Starting bin number (default: 10)
- `-e, --end_bin`: Ending bin number (default: 500)
- `-s, --step`: Step size for bin progression (default: 10)

#### 2. Core MI Analysis
```bash
python multivariate_run.py -o human
```

#### 3. Redundancy Removal
```bash
python mi_red_rmvr.py -o yeast
```

#### 4. Three-way MI Filtering & Visualization
```bash
python three_mi_value_splitter.py -o human
```

---

## Advanced Features

### **Scientific Visualization**
The visualization system includes multiple sophisticated tools:

- **Color Management**: Scientifically validated color palettes with colorblind accessibility
- **Interactive Elements**: Hover information showing MI values, covariance, and correlation data
- **Publication Quality**: High-resolution outputs suitable for academic publications
- **Memory Efficiency**: Optimized rendering for large datasets

### **Performance Optimization**
- **Parallel Processing**: Multi-core utilization for MI calculations
- **Memory Management**: Efficient handling of large histone modification datasets
- **Caching**: Intelligent caching of intermediate results
- **Progress Tracking**: Real-time progress bars and performance metrics

### **Data Integration**
Seamlessly integrates multiple data types:
- Mutual information matrices
- Covariance and correlation data
- Organism-specific metadata
- Custom threshold configurations

---

## Output Structure

Results are organized in the project's output directory:

```
output/files/multivariate_analysis/
├── human/
│   ├── mi_btw_u_and_y.csv                    # Pairwise MI matrix
│   ├── diff_uy_and_uy_givenv_entro.csv       # Three-way MI matrix
│   ├── masked_human_three_mi.csv             # Filtered unique combinations
│   └── three_mi_filtered_*.csv               # Threshold-filtered results
└── yeast/
    ├── mi_btw_u_and_y.csv
    ├── diff_uy_and_uy_givenv_entro.csv
    ├── masked_yeast_three_mi.csv
    └── three_mi_filtered_*.csv

output/images/multivariate_analysis/
├── human/
│   ├── *.eps                                 # High-resolution static plots
│   └── *.html                                # Interactive visualizations
└── yeast/
    ├── *.eps
    └── *.html

output/files/entropy_values/
├── human/
│   ├── all_ptms_bin_based_entropy_values_*.json
│   └── bin_entropy_summary_*.csv
└── yeast/
    ├── all_ptms_bin_based_entropy_values_*.json
    └── bin_entropy_summary_*.csv
```

---

## Configuration

### **Bin Configuration**
The system automatically loads optimal bin configurations from `data/cleaned_data/mi_bin_info.json`, which contains organism-specific parameters determined through entropy optimization.

### **Threshold Ranges**
Pre-configured threshold ranges for different analysis scenarios:
- **Human**: -1.0 to +1.0 with fine-grained subdivisions
- **Yeast**: Optimized ranges based on yeast-specific MI distributions
- **Custom**: User-definable ranges for specialized analyses

---


## Performance Notes

- **Memory Usage**: Optimized for datasets with 20-50 histone modifications
- **Computation Time**: Parallel processing reduces analysis time by 60-80%
- **Storage**: Output files range from MB to GB depending on data size
- **Scalability**: Handles both small research datasets and large-scale genomic data

---

## Troubleshooting

**Common Issues:**
- **Memory Errors**: Use global min/max approach for large datasets
- **Unicode Errors**: Visualization modules handle special characters automatically
- **Missing Files**: Ensure all prerequisite data files are properly generated

**Performance Tips:**
- Use `-p` flag for parallel processing on multi-core systems
- Consider skipping entropy calculation for quick analysis iterations
- Use global min/max values for memory-constrained environments

---

## Example Workflow

```bash
# 1. Optimize bin numbers (optional, can skip with -s entropy)
python bin_num_based_entropy_calculation.py -o human yeast -g

# 2. Run complete pipeline with parallel processing
python multivariate_main_run.py -o human yeast -p

# 3. Generate custom threshold visualizations
python three_mi_value_splitter.py -o human
```

---

For questions or issues, check individual script help messages using the `-h` flag.
