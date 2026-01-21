import json
import ast
import argparse
import os
from typing import Union, Optional, List, Tuple
from datahandler import data_handler
import notebook__init__b
from path_utils.path_handler import ProjectPaths

""""
# This will explain about the range of the total correlation values
# and how to use the TotalCorrelationFinder class to find specific triplet correlations.
# python tc_value_key_finder.py -o yeast  
# or 
# python tc_value_key_finder.py -o human
#  This will show the top 10 highest total correlation values for triplets in the specified organism.
# you can use the --lowest option to get the lowest values instead.
"""


class TotalCorrelationFinder:
    """
    A class to find total correlation values for triplets from JSON data.

    Attributes:
        triplet_data (dict): Dictionary with triplet tuples as keys and correlation values
        file_path (str): Path to the JSON file containing triplet data
    """

    def __init__(self, json_file_path: str):
        """
        Initialize the TotalCorrelationFinder with a JSON file.

        Parameters:
            json_file_path (str): Path to the JSON file containing triplet data
        """
        self.file_path = json_file_path
        self.triplet_data = self._load_triplet_data()

    def _load_triplet_data(self) -> dict:
        """
        Load and convert JSON data to dictionary with tuple keys.

        Returns:
            dict: Dictionary with triplet tuples as keys and correlation values
        """
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)

            # Convert string keys back to tuples
            triplet_data = {ast.literal_eval(k): v for k, v in data.items()}
            print(f"Loaded {len(triplet_data)} triplets from {self.file_path}")
            return triplet_data

        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.file_path}")
            return {}
        except Exception as e:
            print(f"Error loading data: {e}")
            return {}

    def sort_triplet(self, triplet: Union[tuple, list]) -> tuple:
        """
        Sort triplet elements alphabetically to match stored keys.

        Parameters:
            triplet (Union[tuple, list]): Input triplet to sort

        Returns:
            tuple: Sorted triplet
        """
        return tuple(sorted(triplet))

    def find_value(self, triplet: Union[tuple, list]) -> Optional[float]:
        """
        Find total correlation value for a given triplet.

        Parameters:
            triplet (Union[tuple, list]): Triplet to search for

        Returns:
            Optional[float]: Total correlation value if found, None otherwise
        """
        sorted_triplet = self.sort_triplet(triplet)
        return self.triplet_data.get(sorted_triplet)

    def find_multiple_values(self, triplets: List[Union[tuple, list]]) -> dict:
        """
        Find total correlation values for multiple triplets.

        Parameters:
            triplets (List[Union[tuple, list]]): List of triplets to search for

        Returns:
            dict: Dictionary with triplets as keys and correlation values
        """
        results = {}
        for triplet in triplets:
            sorted_triplet = self.sort_triplet(triplet)
            value = self.find_value(triplet)
            results[sorted_triplet] = value
        return results

    def search_by_histone(self, histone_name: str) -> dict:
        """
        Find all triplets containing a specific histone modification.

        Parameters:
            histone_name (str): Name of histone modification to search for

        Returns:
            dict: Dictionary with matching triplets and their correlation values
        """
        matching_triplets = {}
        for triplet, value in self.triplet_data.items():
            if histone_name in triplet:
                matching_triplets[triplet] = value
        return matching_triplets

    def get_top_correlations(self, n: int = 10, ascending: bool = False) -> dict:
        """
        Get top N triplets by correlation value.

        Parameters:
            n (int): Number of top triplets to return
            ascending (bool): If True, return lowest values; if False, return highest

        Returns:
            dict: Dictionary with top N triplets and their correlation values
        """
        sorted_items = sorted(
            self.triplet_data.items(), key=lambda x: x[1], reverse=not ascending
        )
        return dict(sorted_items[:n])

    def get_correlation_range(self, min_val: float, max_val: float) -> dict:
        """
        Get triplets within a specific correlation value range.

        Parameters:
            min_val (float): Minimum correlation value
            max_val (float): Maximum correlation value

        Returns:
            dict: Dictionary with triplets in the specified range
        """
        filtered_triplets = {}
        for triplet, value in self.triplet_data.items():
            if min_val <= value <= max_val:
                filtered_triplets[triplet] = value
        return filtered_triplets

    def get_statistics(self) -> dict:
        """
        Get statistics about the correlation data.

        Returns:
            dict: Dictionary containing statistical information
        """
        if not self.triplet_data:
            return {}

        values = list(self.triplet_data.values())
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
        }

    def export_results(self, results: dict, output_path: str):
        """
        Export results to JSON file.

        Parameters:
            results (dict): Results dictionary to export
            output_path (str): Path to output file
        """
        # Convert tuple keys to strings for JSON compatibility
        json_results = {str(k): v for k, v in results.items()}

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=4)
        print(f"Results exported to {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Find total correlation values by triplet"
    )

    # Required arguments
    parser.add_argument(
        "-o",
        "--organism",
        type=str,
        required=True,
        help="Organism name (e.g., 'human', 'yeast')",
    )

    # Optional arguments
    parser.add_argument(
        "-t", "--triplet", nargs=3, help="Triplet to search for (3 histone names)"
    )
    parser.add_argument(
        "-s", "--search", type=str, help="Search for triplets containing this histone"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Get top N correlations (default: 10)"
    )
    parser.add_argument(
        "--lowest",
        action="store_true",
        help="Get lowest correlations instead of highest",
    )
    parser.add_argument(
        "--range", nargs=2, type=float, help="Get correlations in range [min max]"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show statistics about the data"
    )
    parser.add_argument("--export", type=str, help="Export results to file")

    args = parser.parse_args()

    # Construct file path
    total_corr_dir = "total correlation analysis"
    json_file_path = os.path.join(
        ProjectPaths.get_output_files_dir(),
        total_corr_dir,
        args.organism,
        "total_correlation_triplets.json",
    )

    # Initialize finder
    finder = TotalCorrelationFinder(json_file_path)

    if not finder.triplet_data:
        print("No data loaded. Exiting.")
        return

    results = {}

    # Handle different search modes
    if args.triplet:
        # Search for specific triplet
        triplet = tuple(args.triplet)
        value = finder.find_value(triplet)
        sorted_triplet = finder.sort_triplet(triplet)

        if value is not None:
            print(f"Triplet {sorted_triplet}: {value:.6f}")
            results[sorted_triplet] = value
        else:
            print(f"Triplet {sorted_triplet} not found")

    elif args.search:
        # Search by histone name
        print(f"Searching for triplets containing '{args.search}':")
        results = finder.search_by_histone(args.search)

        if results:
            for triplet, value in sorted(
                results.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {triplet}: {value:.6f}")
        else:
            print(f"  No triplets found containing '{args.search}'")

    elif args.range:
        # Search by value range
        min_val, max_val = args.range
        print(f"Triplets with correlation between {min_val} and {max_val}:")
        results = finder.get_correlation_range(min_val, max_val)

        if results:
            for triplet, value in sorted(
                results.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {triplet}: {value:.6f}")
        else:
            print(f"  No triplets found in range [{min_val}, {max_val}]")

    else:
        # Get top correlations
        print(f"Top {args.top} {'lowest' if args.lowest else 'highest'} correlations:")
        results = finder.get_top_correlations(args.top, args.lowest)

        for triplet, value in results.items():
            print(f"  {triplet}: {value:.6f}")

    # Show statistics if requested
    if args.stats:
        stats = finder.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total triplets: {stats['count']}")
        print(f"  Min correlation: {stats['min']:.6f}")
        print(f"  Max correlation: {stats['max']:.6f}")
        print(f"  Mean correlation: {stats['mean']:.6f}")
        print(f"  Median correlation: {stats['median']:.6f}")

    # Export results if requested
    if args.export and results:
        finder.export_results(results, args.export)


if __name__ == "__main__":
    # Example usage without command line
    if len(os.sys.argv) == 1:
        # Demo mode - replace with your actual file path
        demo_file = "total_correlation_triplets.json"

        if os.path.exists(demo_file):
            finder = TotalCorrelationFinder(demo_file)

            # Example searches
            print("=== Demo Mode ===")

            # Example 1: Find specific triplet
            example_triplet = ("H3K4me1", "H3K4me3", "H3K27me3")
            value = finder.find_value(example_triplet)
            if value:
                print(f"Triplet {finder.sort_triplet(example_triplet)}: {value:.6f}")

            # Example 2: Get top 5 correlations
            print("\nTop 5 highest correlations:")
            top_5 = finder.get_top_correlations(5)
            for triplet, value in top_5.items():
                print(f"  {triplet}: {value:.6f}")

            # Example 3: Show statistics
            stats = finder.get_statistics()
            print(f"\nStatistics:")
            print(f"  Total triplets: {stats['count']}")
            print(f"  Correlation range: {stats['min']:.6f} to {stats['max']:.6f}")
        else:
            print("Demo file not found. Use command line mode.")
    else:
        main()
