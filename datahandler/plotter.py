import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime
from typing import Union, Sequence
from matplotlib.colors import LinearSegmentedColormap

Arraylike = Union[Sequence, pd.Series]
import seaborn as sns
from plotly import figure_factory as ff

# Visualization Parameters
font_dictionary = {"family": "serif", "weight": "bold", "size": "16", "color": "black"}
font_dictionary_ticks = {"family": "serif", "size": "11", "color": "black"}
# plt.rc("text", usetex=True)
# plt.rc("font", family="arial")


def create_custom_colormap(name, colors):
    """
    Create a custom colormap.

    Parameters:
    name (str): The name of the colormap.
    colors (list): A list of color codes.

    Returns:
    LinearSegmentedColormap: The custom colormap.
    """
    return LinearSegmentedColormap.from_list(name, colors)


# Example usage
# custom_map = create_custom_colormap("custom_map", ["#964d18", "white", "#1d6f59"])


def plot_mod(
    ax,
    x_label: str = None,
    y_label: str = None,
    title: str = None,
    x_ticks: str = None,
    y_ticks: str = None,
    x_tick_label: str = None,
    y_tick_label: str = None,
    font: str = None,
):
    """Function
    -----------
    used for modifying the plot output

    parameter:
    ----------
      ax, x-axis label, y-axis label

    """

    font_label = {"size": 14, "family": "serif", "weight": "bold"}
    font_ticks = {"size": 14, "family": "serif", "weight": "normal"}
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if x_label:
        ax.set_xlabel(x_label, fontdict=font_label)

    if y_label:
        ax.set_ylabel(y_label, fontdict=font_label)

    if title:
        ax.set_title(title, fontdict=font_label)

    plt.xticks(**font_ticks)
    plt.yticks(**font_ticks)

    if x_ticks is not None:

        ax.set_xticks(x_ticks)

    if y_ticks is not None:

        ax.set_yticks(y_ticks)

    if x_tick_label is not None:

        ax.set_xticklabels(x_tick_label)
    if y_tick_label is not None:

        ax.set_yticklabels(y_tick_label)


def scatter_plot_gen(
    x_value: Arraylike,
    y_value: Arraylike,
    x_label: str = "x_value",
    y_label: str = "y_value",
    file_sav: bool = False,
    file_name: str = None,
    out_dir: str = None,
    close: bool = False,
):
    """Generate the scatterplot between histone modification."""

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_value, x_value, alpha=0.5, c="black")
    ax.scatter(x_value, y_value, c="#a8a8a8")
    plot_mod(ax=ax, x_label=x_label, y_label=y_label, font="serif")

    if file_sav:
        cur_dir = os.getcwd()

        scatter_dir = "scatter_plot"

        now = datetime.now()

        # conversion of date and time in the string format.
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = os.path.join(scatter_dir, f"picture{date_time}.jpeg")

        if file_name:
            output_file_name = os.path.join(cur_dir, scatter_dir, f"{file_name}.jpeg")

        if out_dir and file_name:
            output_file_name = os.path.join(out_dir, scatter_dir, f"{file_name}.jpeg")

        dir_path = os.path.dirname(output_file_name)
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path, exist_ok=True)

        fig.savefig(output_file_name, dpi=400, bbox_inches="tight")

    if close:
        plt.close()
    else:
        plt.show()


def heatmap_plotter(
    dataframe: pd.DataFrame,
    color_bar: str = "rocket_r",
    sav: bool = False,
    output_image_path: str = None,
    x_label: str = None,
    y_label: str = None,
    colorbar_name: str = None,
    diag_cmap: str = None,
    number_of_bins: int = None,
    tick_font_size: int = 14,
    annot_font_size: int = 9,
    figsize: tuple = (17, 9),
    lower_limit: float = None,
    upper_limit: float = None,
) -> None:
    """Generate the heatmap plot from a given dataframe.

    Parameters
    ----------
    number_of_bins
    dataframe : Dataframe
        Input dataframe for MI calculation.
    sav : bool, optional
        option for whether to sav image or not, by default True
    output_image_path : str, optionally
         provides the output_dir path, by default None
    """
    #
    mpl.rcParams["pgf.texsystem"] = (
        "xelatex"  # using the latex backend for good rendering
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create masks for diagonal and non-diagonal entries
    mask_diag = np.eye(dataframe.shape[0], dtype=bool)  # Mask for diagonal
    mask_non_diag = ~mask_diag  # Mask for non-diagonal

    # Draw the heatmap for diagonal elements
    sns.heatmap(
        dataframe,
        mask=mask_non_diag,
        cmap=diag_cmap,
        annot=True,
        cbar=False,
        fmt=".2f",
        linewidths=0.001,
        ax=ax,
    )

    # Draw the heatmap for non-diagonal elements without a color bar
    heatmap = sns.heatmap(
        data=dataframe,
        cmap=color_bar,
        mask=mask_diag,
        annot=True,
        fmt=".2f",
        annot_kws={"weight": "normal", "family": "arial", "size": annot_font_size},
        cbar=False,  # Temporarily disable color bar
        linewidths=0.001,
        ax=ax,
    )

    # Create and customize the color bar for the non-diagonal elements
    cbar = fig.colorbar(
        heatmap.collections[1],  # Reference the non-diagonal heatmap
        ax=ax,
        pad=0.02,  # Adjust the distance between the heatmap and the color bar
        shrink=0.6,  # Scale the color bar height
    )

    # Customize the color bar label
    cbar.set_label(colorbar_name, fontdict=font_dictionary)

    # Customize color bar ticks and labels (optional)
    # cbar.set_ticks([0.2, 0.5, 0.8])
    # cbar.set_ticklabels(["Low", "Medium", "High"])

    heatmap.collections[1].set_clim(lower_limit, upper_limit)
    # Adjust color bar tick labels
    cbar.ax.tick_params(labelsize=16)  # Adjust tick label size
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    # Customize axis ticks and labels
    font_dictionary_ticks = {"size": tick_font_size, "weight": "bold"}
    plt.xticks(**font_dictionary_ticks)
    plt.xticks(rotation=90)
    plt.yticks(**font_dictionary_ticks)
    plt.xlabel(x_label, fontdict=font_dictionary_ticks)
    plt.ylabel(y_label, fontdict=font_dictionary_ticks)

    # Adjust layout
    plt.subplots_adjust(0.5, 0.5, 1.5, 1.5)

    # Save or display the plot
    if sav:
        plt.savefig(
            os.path.join(f"{output_image_path}.eps"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(f"{output_image_path}.jpeg"),
            dpi=700,
            bbox_inches="tight",
        )
    plt.show()


def line_plot(
    x_value: Arraylike,
    y_value: Arraylike,
    x_label: str = "x_value",
    y_label: str = "y_value",
    file_sav: bool = False,
    file_name: str = None,
    out_dir: str = None,
    close: bool = False,
):

    # x_range = np.ptp(x_value)
    # y_range = np.ptp(y_value)

    # aspect_ratio = y_range/x_range
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(x_value, y_value)  # Plotting the data.
    plt.xticks(rotation=45)
    # plt.yticks

    for indx, txt in enumerate(y_value):
        ax.annotate(
            txt,
            (x_value[indx], y_value[indx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=12,
        )

    plot_mod(ax=ax, x_label=x_label, y_label=y_label)

    if file_sav:
        cur_dir = os.getcwd()

        scatter_dir = "line_plot"

        now = datetime.now()

        # conversion of date and time in the string format.
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_file_name = os.path.join(scatter_dir, f"picture{date_time}.jpeg")

        if file_name:
            output_file_name = os.path.join(cur_dir, scatter_dir, f"{file_name}.jpeg")

        if out_dir and file_name:
            output_file_name = os.path.join(out_dir, scatter_dir, f"{file_name}.jpeg")

        dir_path = os.path.dirname(output_file_name)
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path, exist_ok=True)

        fig.savefig(output_file_name, dpi=400, bbox_inches="tight")

    if close:
        plt.close()
    else:
        plt.show()


import subprocess


def convert_pdf_to_eps(pdf_path, eps_path):
    command = [
        "gs",
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-sDEVICE=eps2write",
        f"-sOutputFile={eps_path}",
        pdf_path,
    ]
    subprocess.run(command, check=True)


def save_dataframe_image_plotly(df: pd.DataFrame, output_filepath=None, indx_title=""):

    fig = ff.create_table(df, index=True, index_title=indx_title)
    # Adjust text alignment to center
    # for annotation in fig.layout.annotations:
    #     annotation.update(xanchor="center")

    fig.update_layout(
        autosize=True,
        width=600 + (len(df.columns) * 100),
        height=200 + (len(df) * 25),
        font=dict(family="arial", size=20),  # Adjust height based on number of rows
    )

    if output_filepath is not None:

        extension = os.path.splitext(output_filepath)[1]
        if extension == ".eps":

            eps_file = output_filepath
            fig.write_image(eps_file, format="eps", engine="kaleido")
            # convert_pdf_to_eps(pdf_path=pdf_file, eps_path=eps_file)

        else:
            fig.write_image(output_filepath, scale=2)

    fig.show()
