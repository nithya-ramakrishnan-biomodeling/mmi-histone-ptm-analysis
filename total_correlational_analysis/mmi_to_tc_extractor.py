import pandas as pd
import numpy as np
import json
import argparse
import os
import ast
from typing import List, Tuple, Optional, Dict
from datahandler import data_handler
import notebook__init__b
from path_utils.path_handler import ProjectPaths

#  This file is used to:
#  mmi value cutoffs as a input to extract triplets from masked MMI CSV files.
# using those triplets, it retrieves their total correlation values
# input csv file: MMI masked_mi.csv


class MaskedMITotalCorrelationExtractor:
    """
    A class to extract triplets from masked MMI CSV files based on MMI-value cutoffs and
    retrieve their total correlation values and initial MMI values.

    Attributes:
        organism (str): Organism name
        csv_file_path (str): Path to the masked MMI CSV file
        total_corr_json_path (str): Path to total correlation JSON file
        df (pd.DataFrame): Loaded masked MMI CSV data
        total_corr_data (dict): Total correlation data
    """

    def __init__(self, organism: str, csv_file_path: str = None):
        """
        Initialize the MaskedMITotalCorrelationExtractor.

        Parameters:
            organism (str): Organism name (e.g., 'human', 'yeast')
            csv_file_path (str, optional): Path to the masked MMI CSV file. If None, uses ProjectPaths
        """
        self.organism = organism
        self.csv_file_path = csv_file_path or self._get_default_csv_path()
        self.total_corr_json_path = self._get_total_corr_json_path()
        self.df = self._load_csv_data()
        self.total_corr_data = self._load_total_correlation_data()

    def _get_default_csv_path(self) -> str:
        """
        Get the default path for the masked MMI CSV file using ProjectPaths.

        Returns:
            str: Path to the masked MMI CSV file
        """
        return os.path.join(
            ProjectPaths.get_output_files_dir(),
            "multivariate analysis",
            self.organism,
            f"masked_{self.organism}_three_mi.csv",
        )

    def _get_total_corr_json_path(self) -> str:
        """
        Get the path for the total correlation JSON file using ProjectPaths.

        Returns:
            str: Path to the total correlation JSON file
        """
        return os.path.join(
            ProjectPaths.get_output_files_dir(),
            "total correlation analysis",
            self.organism,
            "total_correlation_triplets.json",
        )

    def _load_csv_data(self) -> pd.DataFrame:
        """
        Load masked MMI CSV data with proper index handling.

        Returns:
            pd.DataFrame: Loaded CSV data
        """
        try:
            df = pd.read_csv(self.csv_file_path, index_col=0)
            print(f"Loaded masked MMI CSV data from: {self.csv_file_path}")
            print(f"MMI data shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            print(f"Expected path: {self.csv_file_path}")
            return pd.DataFrame()

    def _load_total_correlation_data(self) -> dict:
        """
        Load total correlation data from JSON file.

        Returns:
            dict: Total correlation data with tuple keys
        """
        try:
            with open(self.total_corr_json_path, "r") as f:
                data = json.load(f)

            # Convert string keys back to tuples
            total_corr_data = {ast.literal_eval(k): v for k, v in data.items()}
            print(
                f"Loaded {len(total_corr_data)} total correlation triplets from: {self.total_corr_json_path}"
            )
            return total_corr_data

        except Exception as e:
            print(f"Error loading total correlation data: {e}")
            print(f"Expected path: {self.total_corr_json_path}")
            return {}

    def parse_row_index(self, row_name: str) -> Tuple[str, str]:
        """
        Parse row index to extract the two histone modifications.

        Parameters:
            row_name (str): Row name in format "histone1_histone2"

        Returns:
            Tuple[str, str]: Tuple of two histone names
        """
        if "_" in row_name:
            parts = row_name.split("_")
            if len(parts) >= 2:
                return parts[0], parts[1]
        return row_name, ""

    def sort_triplet(self, triplet: Tuple[str, str, str]) -> Tuple[str, str, str]:
        """
        Sort triplet elements alphabetically to match stored keys.

        Parameters:
            triplet (Tuple[str, str, str]): Input triplet to sort

        Returns:
            Tuple[str, str, str]: Sorted triplet
        """
        return tuple(sorted(triplet))

    def extract_triplets_by_cutoff(
        self, cutoff: float, operator: str = "ge"
    ) -> List[Dict]:
        """
        Extract triplets from masked MMI CSV based on value cutoff.

        Parameters:
            cutoff (float): Cutoff value for filtering
            operator (str): Comparison operator ('le', 'ge', 'lt', 'gt', 'eq')

        Returns:
            List[Dict]: List of dictionaries with triplet information
        """
        triplet_data = []

        # Apply cutoff filter
        if operator == "le":
            mask = self.df <= cutoff
        elif operator == "ge":
            mask = self.df >= cutoff
        elif operator == "lt":
            mask = self.df < cutoff
        elif operator == "gt":
            mask = self.df > cutoff
        elif operator == "eq":
            mask = self.df == cutoff
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        # Find positions where condition is met
        for row_idx, row_name in enumerate(self.df.index):
            for col_idx, col_name in enumerate(self.df.columns):
                if mask.iloc[row_idx, col_idx] and not pd.isna(
                    self.df.iloc[row_idx, col_idx]
                ):
                    # Parse row name to get two histone modifications
                    row_histone1, row_histone2 = self.parse_row_index(row_name)

                    if row_histone2:  # Valid pair found in row
                        # Create triplet: (row_histone1, row_histone2, column_histone)
                        triplet = (row_histone1, row_histone2, col_name)
                        sorted_triplet = self.sort_triplet(triplet)

                        # Get MMI value
                        mmi_value = self.df.iloc[row_idx, col_idx]

                        # Get total correlation value
                        total_corr_value = self.total_corr_data.get(sorted_triplet)

                        triplet_info = {
                            "original_triplet": triplet,
                            "sorted_triplet": sorted_triplet,
                            "row_pair": f"{row_histone1}_{row_histone2}",
                            "column_histone": col_name,
                            "mmi_value": mmi_value,
                            "total_correlation": total_corr_value,
                            "row_index": row_name,
                            "column_index": col_name,
                        }

                        triplet_data.append(triplet_info)

        print(f"Found {len(triplet_data)} triplets with {operator} {cutoff}")
        return triplet_data

    def create_results_dataframe(self, triplet_data: List[Dict]) -> pd.DataFrame:
        """
        Create a results DataFrame from triplet data.

        Parameters:
            triplet_data (List[Dict]): List of triplet information dictionaries

        Returns:
            pd.DataFrame: Results DataFrame
        """
        # Convert to DataFrame
        results_df = pd.DataFrame(triplet_data)

        # Reorder columns for better readability
        column_order = [
            "original_triplet",
            "sorted_triplet",
            "row_pair",
            "column_histone",
            "mmi_value",
            "total_correlation",
            "row_index",
            "column_index",
        ]

        results_df = results_df[column_order]

        # Sort by MMI value (descending) and then by total correlation (descending)
        results_df = results_df.sort_values(
            ["mmi_value", "total_correlation"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        return results_df

    def get_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Get statistics about the results.

        Parameters:
            results_df (pd.DataFrame): Results DataFrame

        Returns:
            Dict: Statistics dictionary
        """
        total_triplets = len(results_df)
        has_total_corr = results_df["total_correlation"].notna().sum()

        mmi_values = results_df["mmi_value"].dropna()
        total_corr_values = results_df["total_correlation"].dropna()

        stats = {
            "total_triplets_found": total_triplets,
            "triplets_with_total_correlation": has_total_corr,
            "coverage_percentage": (
                (has_total_corr / total_triplets * 100) if total_triplets > 0 else 0
            ),
            "mmi_value_stats": {
                "count": len(mmi_values),
                "min": mmi_values.min() if len(mmi_values) > 0 else np.nan,
                "max": mmi_values.max() if len(mmi_values) > 0 else np.nan,
                "mean": mmi_values.mean() if len(mmi_values) > 0 else np.nan,
                "std": mmi_values.std() if len(mmi_values) > 0 else np.nan,
            },
            "total_correlation_stats": {
                "count": len(total_corr_values),
                "min": (
                    total_corr_values.min() if len(total_corr_values) > 0 else np.nan
                ),
                "max": (
                    total_corr_values.max() if len(total_corr_values) > 0 else np.nan
                ),
                "mean": (
                    total_corr_values.mean() if len(total_corr_values) > 0 else np.nan
                ),
                "std": (
                    total_corr_values.std() if len(total_corr_values) > 0 else np.nan
                ),
            },
        }

        return stats

    def export_results(self, results_df: pd.DataFrame, output_path: str = None):
        """
        Export results to CSV file with only required columns: original_triplet, mmi_value, total_correlation_value.

        Parameters:
            results_df (pd.DataFrame): Results DataFrame
            output_path (str, optional): Path to output file. If None, uses ProjectPaths
        """
        if output_path is None:
            output_path = os.path.join(
                ProjectPaths.get_output_files_dir(),
                "total correlation analysis",
                self.organism,
                f"mmi_to_tc_results_{self.organism}.csv",
            )

        # Select only the required columns
        export_df = results_df[
            ["original_triplet", "mmi_value", "total_correlation"]
        ].copy()

        # Rename the total_correlation column to total_correlation_value
        export_df = export_df.rename(
            columns={"total_correlation": "total_correlation_value"}
        )

        export_df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        print(f"Exported {len(export_df)} rows with columns: {list(export_df.columns)}")

    def create_summary_table(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary table with key statistics.

        Parameters:
            results_df (pd.DataFrame): Results DataFrame

        Returns:
            pd.DataFrame: Summary table
        """
        # Create summary by unique sorted triplets
        summary_data = []

        for sorted_triplet, group in results_df.groupby("sorted_triplet"):
            summary_info = {
                "sorted_triplet": str(sorted_triplet),
                "histone_1": sorted_triplet[0],
                "histone_2": sorted_triplet[1],
                "histone_3": sorted_triplet[2],
                "occurrences": len(group),
                "mmi_value_mean": group["mmi_value"].mean(),
                "mmi_value_std": group["mmi_value"].std(),
                "mmi_value_min": group["mmi_value"].min(),
                "mmi_value_max": group["mmi_value"].max(),
                "total_correlation": (
                    group["total_correlation"].iloc[0]
                    if group["total_correlation"].notna().any()
                    else None
                ),
            }
            summary_data.append(summary_info)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(
            "mmi_value_mean", ascending=False
        ).reset_index(drop=True)

        return summary_df


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Extract triplets from masked MMI CSV and get total correlation values"
    )

    # Required arguments
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Organism name (e.g., 'human', 'yeast')",
    )
    parser.add_argument(
        "-c", "--cutoff", type=float, required=True, help="Cutoff value for filtering"
    )

    # Optional arguments
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        help="Path to input masked MMI CSV file (optional, uses ProjectPaths by default)",
    )
    parser.add_argument(
        "-op",
        "--operator",
        type=str,
        default="ge",
        choices=["le", "ge", "lt", "gt", "eq"],
        help="Comparison operator (default: ge)",
    )
    parser.add_argument(
        "--export", type=str, help="Export detailed results to CSV file"
    )
    parser.add_argument(
        "--summary", type=str, help="Export summary results to CSV file"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # Initialize extractor
    extractor = MaskedMITotalCorrelationExtractor(args.organism, args.input_csv)

    if extractor.df.empty or not extractor.total_corr_data:
        print("Error: Could not load required data files. Exiting.")
        return

    # Extract triplets based on cutoff
    triplet_data = extractor.extract_triplets_by_cutoff(args.cutoff, args.operator)

    if not triplet_data:
        print("No triplets found with the specified criteria.")
        return

    # Create results DataFrame
    results_df = extractor.create_results_dataframe(triplet_data)

    # Display basic results
    print(f"\nFound {len(results_df)} triplet entries for {args.organism}:")
    print("\nTop 10 results:")
    print(results_df[["sorted_triplet", "mmi_value", "total_correlation"]].head(10))

    # Show statistics if requested
    if args.stats:
        stats = extractor.get_statistics(results_df)
        print(f"\nStatistics for {args.organism}:")
        print(f"  Total triplet entries found: {stats['total_triplets_found']}")
        print(
            f"  Triplets with total correlation: {stats['triplets_with_total_correlation']}"
        )
        print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        print(
            f"  MMI value range: {stats['mmi_value_stats']['min']:.6f} to {stats['mmi_value_stats']['max']:.6f}"
        )
        print(
            f"  MMI value mean: {stats['mmi_value_stats']['mean']:.6f} ± {stats['mmi_value_stats']['std']:.6f}"
        )
        if not np.isnan(stats["total_correlation_stats"]["min"]):
            print(
                f"  Total correlation range: {stats['total_correlation_stats']['min']:.6f} to {stats['total_correlation_stats']['max']:.6f}"
            )
            print(
                f"  Total correlation mean: {stats['total_correlation_stats']['mean']:.6f} ± {stats['total_correlation_stats']['std']:.6f}"
            )

    # Export detailed results if requested
    if args.export:
        extractor.export_results(results_df, args.export)
    else:
        extractor.export_results(results_df)  # Uses default path

    # Export summary results if requested
    if args.summary:
        summary_df = extractor.create_summary_table(results_df)
        if args.summary:
            summary_df.to_csv(args.summary, index=False)
        else:
            # Use default path for summary
            summary_path = os.path.join(
                ProjectPaths.get_output_files_dir(),
                "total correlation analysis",
                args.organism,
                f"mmi_to_tc_summary_{args.organism}.csv",
            )
            summary_df.to_csv(summary_path, index=False)
        print(f"Summary results exported to: {args.summary or summary_path}")

        print(f"\nSummary: {len(summary_df)} unique triplets")
        print("Top 10 triplets by average MMI value:")
        print(
            summary_df[["sorted_triplet", "mmi_value_mean", "total_correlation"]].head(
                10
            )
        )

    # Show some examples
    print(f"\nExamples of triplets with both MMI and Total Correlation values:")
    has_both = results_df[results_df["total_correlation"].notna()]
    if len(has_both) > 0:
        for _, row in has_both.head(5).iterrows():
            print(
                f"  {row['sorted_triplet']}: MMI={row['mmi_value']:.6f}, Total_Corr={row['total_correlation']:.6f}"
            )
    else:
        print("  No triplets found with both values.")


if __name__ == "__main__":
    main()

# Usage examples:
# Extract triplets with MMI values >= 0.5 for yeast (using default path)
# python mmi_to_tc_extractor.py -o yeast -c 0.5 --stats

# Extract triplets with MMI values >= 0.3 for human and export results
# python mmi_to_tc_extractor.py -o human -c 0.3 -op ge --export "human_mmi_tc_results.csv" --stats

# Extract triplets with custom MMI CSV file
# python mmi_to_tc_extractor.py -o yeast -c 0.1 -i "custom_masked_yeast.csv" --stats
