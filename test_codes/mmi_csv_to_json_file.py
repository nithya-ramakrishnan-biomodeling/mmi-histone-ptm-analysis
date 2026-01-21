import pandas as pd
import json
import numpy as np


def convert_csv_to_json(path):
    # Read the CSV file
    df = pd.read_csv(path, index_col=0)

    # Initialize the result dictionary
    result = {}

    # Get all the column names (PTMs)
    ptms = df.columns.tolist()

    # Process each row in the dataframe
    for index, row in df.iterrows():
        # Parse the index to get the two PTMs that form the pair
        pair_ptms = index.split("_")

        if len(pair_ptms) == 2:
            ptm1, ptm2 = pair_ptms

            # For each column (third PTM), create triplets
            for col_idx, ptm3 in enumerate(ptms):
                value = row.iloc[col_idx]

                # Skip NaN values
                if pd.isna(value):
                    continue

                # Create the triplet as a sorted tuple
                triplet = tuple(sorted([ptm1, ptm2, ptm3]))

                # Store the value
                result[triplet] = float(value)

    return result


def sort_dictionary_by_values(data, reverse=True):
    """
    Sort dictionary by values

    Parameters:
    data (dict): Dictionary to sort
    reverse (bool): If True, sort in descending order (highest values first)
                   If False, sort in ascending order (lowest values first)

    Returns:
    dict: Sorted dictionary
    """
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=reverse)
    return dict(sorted_items)


def save_to_json(data, filename, sort_by_values=True, descending=True):
    """
    Save data to JSON file with optional sorting

    Parameters:
    data (dict): Data to save
    filename (str): Output filename
    sort_by_values (bool): Whether to sort by values
    descending (bool): Sort order (True for descending, False for ascending)
    """
    # Sort the data if requested
    if sort_by_values:
        data = sort_dictionary_by_values(data, reverse=descending)
        print(f"Data sorted by values ({'descending' if descending else 'ascending'})")

    # Convert tuple keys to strings for JSON serialization
    json_data = {}
    for triplet_tuple, value in data.items():
        # Convert tuple to string representation
        triplet_str = str(triplet_tuple)
        json_data[triplet_str] = value

    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Data saved to {filename}")
    print(f"Total triplets: {len(json_data)}")


def print_top_bottom_values(data, n=5):
    """
    Print top and bottom n values from the sorted dictionary
    """
    items_list = list(data.items())

    print(f"\nTop {n} highest values:")
    for i, (triplet, value) in enumerate(items_list[:n]):
        print(f"  {i+1}. {triplet}: {value:.6f}")

    print(f"\nTop {n} lowest values:")
    for i, (triplet, value) in enumerate(items_list[-n:]):
        print(f"  {len(items_list)-n+i+1}. {triplet}: {value:.6f}")


# Main execution
if __name__ == "__main__":
    try:
        # Define the path to the CSV file
        csv_path = r"C:\Users\dhana\OneDrive\Documents\multivariate_ptms_analysis-\output\files\multivariate analysis\human\masked_human_three_mi.csv"

        # Convert CSV to dictionary
        triplet_data = convert_csv_to_json(csv_path)

        print(f"Original data loaded: {len(triplet_data)} triplets")

        # Save sorted data (highest values first)
        # save_to_json(triplet_data, 'yeast_mmi_sorted_desc.json', sort_by_values=True, descending=True)

        # Save sorted data (lowest values first)
        save_to_json(
            triplet_data,
            "human_mmi_sorted_asc.json",
            sort_by_values=True,
            descending=False,
        )

        # Save unsorted data (original order)
        # save_to_json(triplet_data, 'yeast_mmi_unsorted.json', sort_by_values=False)

        # Sort the data for display
        sorted_data = sort_dictionary_by_values(triplet_data, reverse=True)

        # Print top and bottom values
        print_top_bottom_values(sorted_data, n=5)

        # Print some statistics
        values = list(triplet_data.values())
        print(f"\nValue statistics:")
        print(f"  Min value: {min(values):.6f}")
        print(f"  Max value: {max(values):.6f}")
        print(f"  Mean value: {np.mean(values):.6f}")
        print(f"  Median value: {np.median(values):.6f}")

    except Exception as e:
        print(f"Error: {e}")
