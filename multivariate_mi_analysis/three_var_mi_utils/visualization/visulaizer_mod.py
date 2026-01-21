import pandas as pd
import numpy as np
from datahandler import plotter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


def annotate_colorbar(cbar, label, ticks, fontsize=12, labelpad=10):
    """Annotate colorbar with custom formatting"""
    cbar.set_label(label, fontsize=fontsize, labelpad=labelpad)
    cbar.ax.tick_params(labelsize=fontsize)
    # Format ticks to 2 decimal places
    cbar.ax.set_yticklabels([f"{x:.2f}" for x in ticks])


def three_var_filtrd_mi_visualizer(
    df: pd.DataFrame,
    cbar: str,
    filename_to_sav: str = None,
    foldername: str = None,
    organism: str = None,
    additional_info=None,
    output_dir: str = None,
):
    """
    Create and save a heatmap visualization of filtered mutual information data.

    Args:
        df: DataFrame containing the data to visualize
        cbar: Colorbar specification (colormap name)
        filename_to_sav: Base filename for saving outputs
        foldername: Folder name for organizing outputs
        organism: Organism name for file organization
        additional_info: Additional information to include
        output_dir: Base directory for saving outputs
    """
    # Input validation: Check if DataFrame is empty
    if df.empty:
        print("The input DataFrame is empty. Skipping visualization.")
        return

    # Ensure all data is numeric
    try:
        df = df.apply(pd.to_numeric, errors="raise")
    except ValueError as e:
        raise ValueError(
            "DataFrame contains non-numeric values. Please clean your data."
        ) from e

    # Dynamically adjust figure size based on data dimensions
    if len(df.index) >= 200:
        figsize = (20, 50)  # Large dataset
    elif 100 <= len(df.index) < 200:
        figsize = (20, 25)  # Medium dataset
    else:
        figsize = (28, 8)  # Small dataset (human)
        # figsize = (29, 10)  # Small dataset (yeast)

    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap - FIXED: Only create one heatmap
    heatmap = sns.heatmap(
        df.values,
        annot=True,  # Show values in cells
        cmap=cbar,  # Use the cbar parameter for colormap
        xticklabels=df.columns,
        yticklabels=df.index,
        ax=ax,
        annot_kws={"fontsize": 13, "fontweight": "bold"},  # Bold cell annotations
        # linecolor="black",  # Color for cell borders
        # linewidths=2,  # Thickness of cell borders
        # square=True,  # Ensure square cells
    )

    # Set up colorbar with proper dimensions - FIXED: Get colorbar from heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)

    # FIXED: Get the mappable from the heatmap collections
    cbar_obj = plt.colorbar(heatmap.collections[0], cax=cax, aspect=30)

    # Add strong outer border lines
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axhline(y=df.shape[0], color="black", linewidth=1)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=df.shape[1], color="black", linewidth=1)

    # Customize axis labels
    ax.set_xlabel("Histone PTMs", fontsize=20, fontweight="bold")
    ax.set_ylabel("Histone PTMs", fontsize=20, fontweight="bold")

    # Customize tick labels
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", fontsize=20, fontweight="bold"
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=20, fontweight="bold")

    # Configure colorbar - FIXED: Use the colorbar object directly
    ticks = cbar_obj.get_ticks()
    annotate_colorbar(
        cbar_obj,
        label="MI Values",
        ticks=ticks,
        fontsize=10,
        labelpad=20,
    )

    # Make colorbar tick labels bold
    for label in cbar_obj.ax.get_yticklabels():
        label.set_fontweight("bold")

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save visualization as EPS file
    if output_dir and filename_to_sav:
        # Create directory for images
        filename_to_sav_img = os.path.join(
            output_dir, "images", foldername, organism, f"{filename_to_sav}.eps"
        )
        os.makedirs(os.path.dirname(filename_to_sav_img), exist_ok=True)
        plt.savefig(filename_to_sav_img, format="eps", dpi=300)

        # Save data as CSV file
        filename_to_sav_csv = os.path.join(
            output_dir, "files", foldername, organism, f"{filename_to_sav}.csv"
        )
        os.makedirs(os.path.dirname(filename_to_sav_csv), exist_ok=True)
        df.to_csv(filename_to_sav_csv)

    # Clean up
    plt.close()
