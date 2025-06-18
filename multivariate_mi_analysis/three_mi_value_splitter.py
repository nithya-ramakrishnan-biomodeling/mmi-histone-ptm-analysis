"""
Three-Variable Mutual Information Value Splitter

This script splits and visualizes three-variable mutual information values based on
user-defined thresholds for human and yeast organisms. It generates heatmaps showing
relationships between different histone modifications.
"""
import os
import argparse
import pandas as pd
import seaborn as sns

# Local imports
import notebook__init__b
from datahandler import data_handler, dir_handler
from path_utils.path_handler import ProjectPaths
from three_var_mi_utils.visualization import visulaizer_mod
from three_var_mi_utils.notebooks import data_processor

# Constants
OUTPUT_FOLDER_NAME = "multivariate analysis"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split and visualize three-variable mutual information values"
    )
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Organism to analyze (e.g., 'human', 'yeast')",
    )
    return parser.parse_args()


def setup_directories(organism):
    """Set up the required directories for input/output files."""
    cleaned_dir = os.path.join(ProjectPaths.get_cleaned_data_dir(), organism)
    output_dir = os.path.join(
        ProjectPaths.get_output_dir(), OUTPUT_FOLDER_NAME, organism
    )
    output_file_dir = os.path.join(
        ProjectPaths.get_output_files_dir(), OUTPUT_FOLDER_NAME, organism
    )
    output_img_dir = os.path.join(
        ProjectPaths.get_output_images_dir(), OUTPUT_FOLDER_NAME, organism
    )

    # Create output directories if they don't exist
    dir_handler.dir_maker(output_file_dir)
    dir_handler.dir_maker(output_img_dir)

    return cleaned_dir, output_dir, output_file_dir, output_img_dir


def load_data_files(organism, cleaned_dir, output_file_dir):
    """Load all required data files."""
    # Raw histone data file
    raw_path_file = os.path.join(cleaned_dir, f"{organism}_zero_mean.csv")

    # Masked three MI file
    three_mi_input_file_path = os.path.join(
        output_file_dir, f"masked_{organism}_three_mi.csv"
    )

    # Two-variable MI file
    two_mi_input_file_path = os.path.join(output_file_dir, "mi_btw_u_and_y.csv")

    # Three-variable MI all combinations
    three_mi_full_file_path = os.path.join(
        output_file_dir, "diff_uy_and_uy_givenv_entro.csv"
    )

    # Load CSV files
    raw_df = data_handler.csv_loader(raw_path_file)
    three_mi_df = data_handler.csv_loader(three_mi_input_file_path)
    all_permut_df = data_handler.csv_loader(three_mi_full_file_path)
    two_mi_df = data_handler.csv_loader(two_mi_input_file_path)

    return raw_df, three_mi_df, all_permut_df, two_mi_df


def get_threshold_dictionary(organism):
    """Get appropriate threshold dictionary based on organism."""
    # Common thresholds for visualization
    common_thresholds = {
        # Very small values around zero
        "zero_vicinity": {"threshold": (0.05, -0.05), "oprtr": ("le", "ge")},
        # Full range
        "full_range": {"threshold": (1.0, -1.0), "oprtr": ("le", "ge")},
    }

    # Human-specific thresholds
    human_thresholds = {
        # Negative ranges
        "small_negative": {"threshold": (0, -0.1), "oprtr": ("le", "ge")},
        "medium_negative": {"threshold": (-0.1, -0.5), "oprtr": ("le", "ge")},
        "large_negative": {"threshold": (-0.5, -1.0), "oprtr": ("le", "ge")},
        # Positive ranges
        "small_positive": {"threshold": (0.2, 0), "oprtr": ("le", "ge")},
        "medium_positive": {"threshold": (0.5, 0.2), "oprtr": ("le", "ge")},
        "large_positive": {"threshold": (1.0, 0.5), "oprtr": ("le", "ge")},
    }

    # Yeast-specific thresholds
    yeast_thresholds = {
        # Negative ranges
        "small_negative": {"threshold": (0, -0.1), "oprtr": ("le", "ge")},
        "medium_negative": {"threshold": (-0.1, -0.3), "oprtr": ("le", "ge")},
        "large_negative": {"threshold": (-0.3, -0.5), "oprtr": ("le", "ge")},
        "very_large_negative": {"threshold": (-0.5, -1.0), "oprtr": ("le", "ge")},
        # Positive ranges
        "small_positive": {"threshold": (0.5, 0), "oprtr": ("le", "ge")},
        "medium_positive": {"threshold": (0.9, 0.5), "oprtr": ("le", "ge")},
        "large_positive": {"threshold": (1.5, 0.9), "oprtr": ("le", "ge")},
    }

    # Merge appropriate thresholds based on organism
    if organism.lower() == "human":
        return {**common_thresholds, **human_thresholds}
    else:
        return {**common_thresholds, **yeast_thresholds}


def generate_visualizations(
    organism, three_mi_df, two_mi_df, raw_df, threshold_dict, output_dir
):
    """Generate visualizations for each threshold range."""
    # Calculate covariance and correlation matrices
    cov_df = raw_df.cov()
    corr_df = raw_df.corr()

    print(f"\nGenerating visualizations for {organism}...")

    # Process each threshold in the dictionary
    for range_name, params in threshold_dict.items():
        print(f"Processing range: {range_name}")

        # Select colormap based on threshold values
        if params["threshold"][1] < 0:
            color_map = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
        else:
            color_map = "RdBu_r"

        # Generate data and annotations
        mi_dict = data_processor.two_mi_value_combn_three_mi_value_gen(
            three_mi_df=three_mi_df,
            two_mi_df=two_mi_df,
            cond_value=params["oprtr"],
            oprtr=params["threshold"],
            cov_df=cov_df,
            corr_df=corr_df,
        )

        result_df, annotations = data_processor.df_and_text_generator(
            mi_dict, df_new_col=three_mi_df.columns, df_new_indx=three_mi_df.index
        )

        # Generate visualization
        output_filename = f"{organism}_{range_name}"
        visulaizer_mod.three_var_filtrd_mi_visualizer(
            df=result_df.T,
            cbar=color_map,
            filename_to_sav=output_filename,
            foldername=OUTPUT_FOLDER_NAME,
            organism=organism,
            additional_info=annotations,
            output_dir=output_dir,
        )

        # Save the filtered data as CSV
        output_csv_path = os.path.join(
            ProjectPaths.get_output_files_dir(),
            OUTPUT_FOLDER_NAME,
            organism,
            f"three_mi_filtered_{range_name}.csv",
        )
        result_df.to_csv(output_csv_path)
        print(f"Saved filtered data to {output_csv_path}")


def main():
    """Main function to run the script."""
    # Parse command line arguments
    args = parse_arguments()
    organism = args.organism

    print(f"\n{'='*80}\nThree-Variable MI Value Splitter\n{'='*80}")
    print(f"Analyzing organism: {organism}")

    # Setup directories
    cleaned_dir, output_dir, output_file_dir, output_img_dir = setup_directories(organism)

    # Load data
    raw_df, three_mi_df, all_permut_df, two_mi_df = load_data_files(
        organism, cleaned_dir, output_file_dir
    )

    # Get threshold dictionary
    threshold_dict = get_threshold_dictionary(organism)

    # Generate visualizations
    generate_visualizations(
        organism, three_mi_df, two_mi_df, raw_df, threshold_dict, output_dir
    )

    print(f"\n{'='*80}\nAnalysis complete!\n{'='*80}")


if __name__ == "__main__":
    main()
