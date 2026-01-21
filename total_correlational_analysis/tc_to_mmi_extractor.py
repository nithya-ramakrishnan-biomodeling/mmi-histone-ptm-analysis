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

"""'
This module is used to:
1. using tc-value cutoffs a input to extract triplets from total correlation CSV files.
2. Retrieve their tc values and MMI values.
"""


class TotalCorrelationMIExtractor:
    """
    A class to extract triplets from total correlation CSV file based on tc-value cutoffs and
    retrieve their MMI values and total correlation values.

    Attributes:
        organism (str): Organism name
        total_corr_df (pd.DataFrame): Loaded total correlation CSV data
        mi_df (pd.DataFrame): Loaded masked MMI CSV data
    """

    def __init__(self, organism: str):
        """
        Initialize the TotalCorrelationMIExtractor.

        Parameters:
            organism (str): Organism name (e.g., 'human', 'yeast')
        """
        self.organism = organism
        self.total_corr_df = self._load_total_correlation_csv()
        self.mi_df = self._load_mi_csv()

    def _load_total_correlation_csv(self) -> pd.DataFrame:
        """
        Load total correlation CSV data using ProjectPaths.

        Returns:
            pd.DataFrame: Loaded total correlation CSV data
        """
        try:
            # Construct path using ProjectPaths
            total_corr_path = os.path.join(
                ProjectPaths.get_output_files_dir(),
                "total correlation analysis",
                self.organism,
                "total_correlation_results.csv",
            )

            df = pd.read_csv(total_corr_path, index_col=0)
            print(f"Loaded total correlation CSV data from: {total_corr_path}")
            print(f"Total correlation data shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading total correlation CSV file: {e}")
            print(f"Expected path: {total_corr_path}")
            return pd.DataFrame()

    def _load_mi_csv(self) -> pd.DataFrame:
        """
        Load masked MMI CSV data using ProjectPaths.

        Returns:
            pd.DataFrame: Loaded MMI CSV data
        """
        try:
            # Construct path using ProjectPaths
            mi_path = os.path.join(
                ProjectPaths.get_output_files_dir(),
                "multivariate analysis",
                self.organism,
                f"masked_{self.organism}_three_mi.csv",
            )

            df = pd.read_csv(mi_path, index_col=0)
            print(f"Loaded masked MMI CSV data from: {mi_path}")
            print(f"MMI data shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading MMI CSV file: {e}")
            print(f"Expected path: {mi_path}")
            return pd.DataFrame()

    def parse_row_index(self, row_name: str) -> Tuple[str, str]:
        """
        Parse row index to extract the two histone modifications.

        Parameters:
            row_name (str): Row name in format "histone1_histone2" or "histone1.histone2_histone3"

        Returns:
            Tuple[str, str]: Tuple of two histone names
        """
        if "_" in row_name:
            parts = row_name.split("_")
            if len(parts) >= 2:
                return parts[0], parts[1]
        elif "." in row_name:
            # Handle cases like "H2A.H4R3me2_H2AZ"
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

    def find_mmi_value(
        self, triplet: Tuple[str, str, str]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Find MMI value for a given triplet in the masked MMI CSV.

        Parameters:
            triplet (Tuple[str, str, str]): Triplet to search for

        Returns:
            Tuple[Optional[float], Optional[str]]: MMI value and the row_pair used (if found)
        """
        # Try all possible combinations of the triplet for row_pair + column
        histone1, histone2, histone3 = triplet

        # Try different row pair combinations
        possible_row_pairs = [
            f"{histone1}_{histone2}",
            f"{histone2}_{histone1}",
            f"{histone1}_{histone3}",
            f"{histone3}_{histone1}",
            f"{histone2}_{histone3}",
            f"{histone3}_{histone2}",
        ]

        for row_pair in possible_row_pairs:
            if row_pair in self.mi_df.index:
                # Determine the column (the remaining histone)
                if row_pair == f"{histone1}_{histone2}":
                    col = histone3
                elif row_pair == f"{histone2}_{histone1}":
                    col = histone3
                elif row_pair == f"{histone1}_{histone3}":
                    col = histone2
                elif row_pair == f"{histone3}_{histone1}":
                    col = histone2
                elif row_pair == f"{histone2}_{histone3}":
                    col = histone1
                elif row_pair == f"{histone3}_{histone2}":
                    col = histone1

                if col in self.mi_df.columns:
                    value = self.mi_df.loc[row_pair, col]
                    if not pd.isna(value):
                        return value, row_pair

        return None, None

    def extract_triplets_by_cutoff(
        self, cutoff: float, operator: str = "ge"
    ) -> List[Dict]:
        """
        Extract unique triplets from total correlation CSV based on value cutoff.

        Parameters:
            cutoff (float): Cutoff value for filtering
            operator (str): Comparison operator ('le', 'ge', 'lt', 'gt', 'eq')

        Returns:
            List[Dict]: List of dictionaries with unique triplet information
        """
        triplet_data = []
        unique_triplets = set()  # Track unique sorted triplets

        # Apply cutoff filter
        if operator == "le":
            mask = self.total_corr_df <= cutoff
        elif operator == "ge":
            mask = self.total_corr_df >= cutoff
        elif operator == "lt":
            mask = self.total_corr_df < cutoff
        elif operator == "gt":
            mask = self.total_corr_df > cutoff
        elif operator == "eq":
            mask = self.total_corr_df == cutoff
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        # Find positions where condition is met
        for row_idx, row_name in enumerate(self.total_corr_df.index):
            for col_idx, col_name in enumerate(self.total_corr_df.columns):
                if mask.iloc[row_idx, col_idx] and not pd.isna(
                    self.total_corr_df.iloc[row_idx, col_idx]
                ):
                    # Parse row name to get two histone modifications
                    row_histone1, row_histone2 = self.parse_row_index(row_name)

                    if row_histone2:  # Valid pair found in row
                        # Create triplet: (row_histone1, row_histone2, column_histone)
                        triplet = (row_histone1, row_histone2, col_name)
                        sorted_triplet = self.sort_triplet(triplet)

                        # Only process if this sorted triplet hasn't been seen before
                        if sorted_triplet not in unique_triplets:
                            unique_triplets.add(sorted_triplet)

                            # Get total correlation value
                            total_corr_value = self.total_corr_df.iloc[row_idx, col_idx]

                            # Get MMI value and the row_pair used
                            mmi_value, mmi_row_pair = self.find_mmi_value(triplet)

                            triplet_info = {
                                "sorted_triplet": sorted_triplet,
                                "histone_1": sorted_triplet[0],
                                "histone_2": sorted_triplet[1],
                                "histone_3": sorted_triplet[2],
                                "total_correlation": total_corr_value,
                                "mmi_value": mmi_value,
                                "mmi_row_pair": mmi_row_pair,
                                "original_row_index": row_name,
                                "original_column_index": col_name,
                            }

                            triplet_data.append(triplet_info)

        print(f"Found {len(triplet_data)} unique triplets with {operator} {cutoff}")
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
            "sorted_triplet",
            "histone_1",
            "histone_2",
            "histone_3",
            "total_correlation",
            "mmi_value",
            "mmi_row_pair",
            "original_row_index",
            "original_column_index",
        ]

        results_df = results_df[column_order]

        # Sort by total correlation (descending) and then by MMI value (descending)
        results_df = results_df.sort_values(
            ["total_correlation", "mmi_value"],
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
        has_mmi = results_df["mmi_value"].notna().sum()

        total_corr_values = results_df["total_correlation"].dropna()
        mmi_values = results_df["mmi_value"].dropna()

        stats = {
            "unique_triplets_found": total_triplets,
            "triplets_with_mmi_value": has_mmi,
            "coverage_percentage": (
                (has_mmi / total_triplets * 100) if total_triplets > 0 else 0
            ),
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
            "mmi_value_stats": {
                "count": len(mmi_values),
                "min": mmi_values.min() if len(mmi_values) > 0 else np.nan,
                "max": mmi_values.max() if len(mmi_values) > 0 else np.nan,
                "mean": mmi_values.mean() if len(mmi_values) > 0 else np.nan,
                "std": mmi_values.std() if len(mmi_values) > 0 else np.nan,
            },
        }

        return stats

    def export_results(self, results_df: pd.DataFrame, output_path: str = None):
        """
        Export results to CSV file with only required columns: sorted_triplet, total_correlation_value, mmi_value.

        Parameters:
            results_df (pd.DataFrame): Results DataFrame
            output_path (str): Path to output file. If None, uses ProjectPaths
        """
        if output_path is None:
            output_path = os.path.join(
                ProjectPaths.get_output_files_dir(),
                "total correlation analysis",
                self.organism,
                f"tc_to_mmi_results_{self.organism}.csv",
            )

        # Select only the required columns
        export_df = results_df[
            ["sorted_triplet", "total_correlation", "mmi_value"]
        ].copy()

        # Rename the total_correlation column to total_correlation_value
        export_df = export_df.rename(
            columns={"total_correlation": "total_correlation_value"}
        )

        export_df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        print(f"Exported {len(export_df)} rows with columns: {list(export_df.columns)}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Extract unique triplets from total correlation CSV and get MMI values"
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
        "-op",
        "--operator",
        type=str,
        default="ge",
        choices=["le", "ge", "lt", "gt", "eq"],
        help="Comparison operator (default: ge)",
    )
    parser.add_argument(
        "--export", type=str, help="Export results to specific CSV file path"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # Initialize extractor with organism
    extractor = TotalCorrelationMIExtractor(args.organism)

    if extractor.total_corr_df.empty or extractor.mi_df.empty:
        print("Error: Could not load required data files. Exiting.")
        return

    # Extract unique triplets based on cutoff
    triplet_data = extractor.extract_triplets_by_cutoff(args.cutoff, args.operator)

    if not triplet_data:
        print("No triplets found with the specified criteria.")
        return

    # Create results DataFrame
    results_df = extractor.create_results_dataframe(triplet_data)

    # Display basic results
    print(f"\nFound {len(results_df)} unique triplets for {args.organism}:")
    print("\nTop 10 results:")
    print(results_df[["sorted_triplet", "total_correlation", "mmi_value"]].head(10))

    # Show statistics if requested
    if args.stats:
        stats = extractor.get_statistics(results_df)
        print(f"\nStatistics for {args.organism}:")
        print(f"  Unique triplets found: {stats['unique_triplets_found']}")
        print(f"  Triplets with MMI values: {stats['triplets_with_mmi_value']}")
        print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        print(
            f"  Total correlation range: {stats['total_correlation_stats']['min']:.6f} to {stats['total_correlation_stats']['max']:.6f}"
        )
        print(
            f"  Total correlation mean: {stats['total_correlation_stats']['mean']:.6f} ± {stats['total_correlation_stats']['std']:.6f}"
        )
        if not np.isnan(stats["mmi_value_stats"]["min"]):
            print(
                f"  MMI value range: {stats['mmi_value_stats']['min']:.6f} to {stats['mmi_value_stats']['max']:.6f}"
            )
            print(
                f"  MMI value mean: {stats['mmi_value_stats']['mean']:.6f} ± {stats['mmi_value_stats']['std']:.6f}"
            )

    # Export results if requested
    if args.export:
        extractor.export_results(results_df, args.export)
    else:
        extractor.export_results(results_df)  # Uses default path

    # Show some examples
    print(f"\nExamples of unique triplets with both Total Correlation and MMI values:")
    has_both = results_df[results_df["mmi_value"].notna()]
    if len(has_both) > 0:
        for _, row in has_both.head(5).iterrows():
            print(
                f"  {row['sorted_triplet']}: Total_Corr={row['total_correlation']:.6f}, MMI={row['mmi_value']:.6f}"
            )
            if row["mmi_row_pair"]:
                print(f"    (MMI found in row: {row['mmi_row_pair']})")
    else:
        print("  No triplets found with both values.")

    # Show triplets without MMI values
    no_mmi = results_df[results_df["mmi_value"].isna()]
    if len(no_mmi) > 0:
        print(f"\nExamples of triplets without MMI values ({len(no_mmi)} total):")
        for _, row in no_mmi.head(3).iterrows():
            print(
                f"  {row['sorted_triplet']}: Total_Corr={row['total_correlation']:.6f}, MMI=Not Found"
            )


if __name__ == "__main__":
    main()

# Usage examples:
# Extract triplets with total correlation values >= 0.5 for yeast
# python tc_to_mmi_extractor.py -o yeast -c 0.5 --stats

# Extract triplets with total correlation values >= 0.3 for human and export results
# python tc_to_mmi_extractor.py -o human -c 0.3 -op ge --export "human_tc_mmi_results.csv" --stats

# Extract triplets with total correlation values <= 0.1 for yeast (low correlations)
# python tc_to_mmi_extractor.py -o yeast -c 0.1 -op le --stats
