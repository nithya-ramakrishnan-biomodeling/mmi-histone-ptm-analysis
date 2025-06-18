import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd


font_label = {"size": 14, "family": "serif", "weight": "bold"}
font_ticks = {"size": 14, "family": "serif", "weight": "normal"}
cbar_label = {"size": 12, "family": "serif", "weight": "normal"}


def plot_mod(
    ax,
    x_label=None,
    y_label=None,
    title=None,
    x_ticks=None,
    y_ticks=None,
    x_tick_label=None,
    y_tick_label=None,
    font=None,
):
    if x_label:
        ax.set_xlabel(x_label, fontdict=font_label)

    if y_label:
        ax.set_ylabel(y_label, fontdict=font_label)

    if title:
        ax.set_title(title, fontdict=font_label)

    plt.xticks(**font_ticks)
    plt.yticks(**font_ticks)

    if x_ticks:
        ax.set_xticks(x_ticks)

    if y_ticks:
        ax.set_yticks(y_ticks)

    if x_tick_label:
        ax.set_xticklabels(x_tick_label)

    if y_tick_label:
        ax.set_yticklabels(y_tick_label)


def plotter(df: pd.DataFrame, filename: str = None) -> None:
    """generate heatmap from the MI pandas dataframe

    Parameters
    ----------
    df : pd.DataFrame
        input Mi pandas dataframe.
    filename : str, optional
        name for saving the heatmap, by default None
    """

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(df, cmap="YlOrBr", ax=ax, annot=True, fmt=".2f", cbar=False)

    # customization in the heatmap color bar:
    colorbar_obj = plt.colorbar(
        ax.collections[0],
        ax=ax,
        extend="both",
        shrink=0.5,
        orientation="horizontal",
        location="top",
        pad=0.01,
    )

    colorbar_obj.set_label("MI values", rotation=0, **cbar_label)
    colorbar_obj.ax.tick_params(labelsize=12, labelfontfamily="serif")
    plot_mod(ax)  # Standardization of the graph plotting.

    if filename:
        current_dir = pathlib.Path.cwd()
        output_file_path = pathlib.Path.joinpath(
            current_dir, "output", "image_files", f"{filename}.jpeg"
        )
        plt.savefig(output_file_path, dpi=500, bbox_inches="tight")

    plt.close()


# creating function for the plotting three var mutual information bar graph


def three_var_mi_barplot(df: pd.DataFrame, var_1: str, var_2: str):

    fig, ax = plt.subplots(figsize=(14, 10))
    data_to_plot = pd.DataFrame(df.loc[f"{var_1}_{var_2}"].sort_values().round(3))

    # Ensure index is treated as categorical (if it's not numerical)
    data_to_plot.index = pd.Categorical(data_to_plot.index)
    data_to_plot = data_to_plot.drop(data_to_plot.index[-2:])

    # Create the bar plot
    data_to_plot.plot(kind="bar", ax=ax, color="#CACACA", legend=False)

    # Annotate bars with their values
    for i, value in enumerate(data_to_plot.iloc[:, 0]):

        if value < 0:
            height = value - 0.015
        else:
            height = value + 0.015
        # Assuming you are plotting one column
        ax.text(
            i,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Third Histone modifications", fontsize=12)
    plt.ylabel(" Three var MI", fontsize=12)
    plt.title(f"{var_1}_{var_2}")
    plot_mod(ax)
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    input_file_path = r"/home/group_nithya01/Desktop/dhana_ibab_intern/file_downloads/cluster_work/multivariate_mi_current/difference_between_mi and_cond_mi.csv"
    plotter(pd.read_csv(input_file_path, index_col=0))
