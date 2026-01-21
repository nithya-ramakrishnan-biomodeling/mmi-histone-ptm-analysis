import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import seaborn as sns


def plot_colorbar_from_csv(csv_path, save_path=None, figsize=(12, 8)):
    """
    Read CSV file and plot a colorbar showing the range of values (min-max)

    Parameters:
    csv_path (str): Path to the CSV file
    save_path (str): Path to save the figure (optional)
    figsize (tuple): Figure size (width, height)
    """

    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)

    # Convert all columns to numeric, replacing non-numeric values with NaN
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Calculate min and max values (excluding NaN)
    min_val = numeric_df.min().min()
    max_val = numeric_df.max().max()

    print(f"Data range: {min_val:.6f} to {max_val:.6f}")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [4, 1]}
    )

    # Plot 1: Heatmap of the data
    # Create a masked array to handle NaN values
    masked_data = np.ma.masked_invalid(numeric_df.values)

    # Create heatmap
    im = ax1.imshow(
        masked_data, cmap="viridis", aspect="auto", vmin=min_val, vmax=max_val
    )

    # Set labels and title
    ax1.set_title("Heatmap of Data Values", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Columns", fontsize=12)
    ax1.set_ylabel("Rows", fontsize=12)

    # Set ticks (show every 5th tick to avoid crowding)
    if len(df.columns) > 20:
        ax1.set_xticks(range(0, len(df.columns), 5))
        ax1.set_xticklabels(df.columns[::5], rotation=45, ha="right")
    else:
        ax1.set_xticks(range(len(df.columns)))
        ax1.set_xticklabels(df.columns, rotation=45, ha="right")

    if len(df.index) > 20:
        ax1.set_yticks(range(0, len(df.index), 5))
        ax1.set_yticklabels(df.index[::5])
    else:
        ax1.set_yticks(range(len(df.index)))
        ax1.set_yticklabels(df.index)

    # Plot 2: Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=1.0, aspect=20)
    cbar.set_label("Value Range", rotation=270, labelpad=20, fontsize=12)

    # Remove the second axis
    ax2.remove()

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return min_val, max_val


def plot_simple_colorbar(csv_path, save_path=None, figsize=(8, 6)):
    """
    Plot just a colorbar showing the data range

    Parameters:
    csv_path (str): Path to the CSV file
    save_path (str): Path to save the figure (optional)
    figsize (tuple): Figure size (width, height)
    """

    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)

    # Convert all columns to numeric, replacing non-numeric values with NaN
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Calculate min and max values (excluding NaN)
    min_val = numeric_df.min().min()
    max_val = numeric_df.max().max()

    print(f"Data range: {min_val:.6f} to {max_val:.6f}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create a simple colorbar
    norm = Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Value Range", rotation=270, labelpad=20, fontsize=14)

    # Add title and statistics
    ax.text(
        0.5,
        0.7,
        f"Data Range Analysis",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.5,
        f"Minimum: {min_val:.6f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0.5,
        0.4,
        f"Maximum: {max_val:.6f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0.5,
        0.3,
        f"Range: {max_val - min_val:.6f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return min_val, max_val


def plot_distribution_with_colorbar(csv_path, save_path=None, figsize=(12, 8)):
    """
    Plot data distribution histogram with colorbar

    Parameters:
    csv_path (str): Path to the CSV file
    save_path (str): Path to save the figure (optional)
    figsize (tuple): Figure size (width, height)
    """

    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)

    # Convert all columns to numeric, replacing non-numeric values with NaN
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Flatten all values and remove NaN
    all_values = numeric_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]

    # Calculate statistics
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    mean_val = np.mean(all_values)
    median_val = np.median(all_values)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Histogram
    n, bins, patches = ax1.hist(all_values, bins=50, alpha=0.7, edgecolor="black")

    # Color the histogram bars according to their position
    norm = Normalize(vmin=min_val, vmax=max_val)
    for i, (bin_val, patch) in enumerate(zip(bins[:-1], patches)):
        color = plt.cm.viridis(norm(bin_val))
        patch.set_facecolor(color)

    ax1.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.4f}",
    )
    ax1.axvline(
        median_val,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_val:.4f}",
    )

    ax1.set_title("Distribution of Values", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Values", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label("Value Range", rotation=270, labelpad=20, fontsize=12)

    # Add statistics text
    ax2.text(
        0.1,
        0.8,
        f"Statistics:",
        fontsize=12,
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(0.1, 0.7, f"Min: {min_val:.6f}", fontsize=10, transform=ax2.transAxes)
    ax2.text(0.1, 0.6, f"Max: {max_val:.6f}", fontsize=10, transform=ax2.transAxes)
    ax2.text(0.1, 0.5, f"Mean: {mean_val:.6f}", fontsize=10, transform=ax2.transAxes)
    ax2.text(
        0.1, 0.4, f"Median: {median_val:.6f}", fontsize=10, transform=ax2.transAxes
    )
    ax2.text(
        0.1,
        0.3,
        f"Range: {max_val - min_val:.6f}",
        fontsize=10,
        transform=ax2.transAxes,
    )

    # Remove axis elements for the colorbar subplot
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return min_val, max_val, mean_val, median_val


# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file_path = r"C:\Users\dhana\OneDrive\Documents\multivariate_ptms_analysis-\output\files\multivariate analysis\yeast\masked_yeast_three_mi.csv"
    # print("Option 1: Heatmap with colorbar")
    # min_val, max_val = plot_colorbar_from_csv(csv_file_path,
    #                                          save_path="heatmap_colorbar.png")

    print("\nOption 2: Simple colorbar with statistics")
    min_val, max_val = plot_simple_colorbar(
        csv_file_path, save_path="simple_colorbar.png"
    )

    # print("\nOption 3: Distribution with colorbar")
    # min_val, max_val, mean_val, median_val = plot_distribution_with_colorbar(csv_file_path,
    #                                                                        save_path="distribution_colorbar.png")
