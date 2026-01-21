import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List
from datetime import datetime
from typing import Union, Sequence

Arraylike = Union[Sequence, pd.Series]


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

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    title : str, optional
        Title of the plot.
    x_ticks : str, optional
        Ticks for the x-axis.
    y_ticks : str, optional
        Ticks for the y-axis.
    x_tick_label : str, optional
        Labels for the x-axis ticks.
    y_tick_label : str, optional
        Labels for the y-axis ticks.
    font : str, optional
        Font for the labels.
    """

    font_label = {"size": 15, "family": "arial", "weight": "bold"}
    font_ticks = {"size": 14, "family": "arial", "weight": "bold"}
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
    """Generate the scatterplot between histone modifications.

    Parameters:
    ----------
    x_value : Arraylike
        Values for the x-axis.
    y_value : Arraylike
        Values for the y-axis.
    x_label : str, optional
        Label for the x-axis, by default "x_value".
    y_label : str, optional
        Label for the y-axis, by default "y_value".
    file_sav : bool, optional
        Whether to save the file, by default False.
    file_name : str, optional
        Name of the file to save, by default None.
    out_dir : str, optional
        Directory to save the file, by default None.
    close : bool, optional
        Whether to close the plot after saving, by default False.
    """

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
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        fig.savefig(output_file_name, dpi=400, bbox_inches="tight")

    if close:
        plt.close()
    else:
        plt.show()
