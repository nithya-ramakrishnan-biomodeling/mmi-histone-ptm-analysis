import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
from typing import Literal, Union, Any
import os
from datahandler import dir_handler


def three_var_filtrd_mi_visualizer(
    df: pd.DataFrame,
    cbar: str,
    filename_to_sav: str = None,
    foldername: str = None,
    organism: str = None,
    additional_info=None,
    output_dir: str = None,
):

    data = df.T.to_numpy()
    if len(df.index) >= 200:
        height = 800
        width = 10000

    elif len(df.index) >= 100 and len(df.index) <= 200:
        height = 500
        width = 8000
    else:
        height = 500
        width = 1800

    # Create heatmap using Plotly
    heatmap = go.Heatmap(
        z=data,
        y=df.columns,
        x=df.index,
        hovertemplate="X: %{x}<br>Y: %{y}<br> 3_mi: %{z}<br> Info: %{text}<extra></extra>",
        text=additional_info,
        colorscale=cbar,
        colorbar=dict(
            orientation="v",
            thickness=10,  # Adjust thickness of the colorbar
            len=0.7,  # Adjust length of the colorbar
            x=1.02,  # Adjust x position of the colorbar
            y=0.5,  # Adjust y position of the colorbar for landscape mode
            title="Mi Values",
        ),
    )

    layout = go.Layout(
        # xaxis=dict(title="X Axis"),
        xaxis=dict(
            title="Histone PTMs",
            side="top",
            tickangle=-45,
            tickfont=dict(
                size=15,
                family="Arial, sans-serif",
                color="black",
                weight="bold",
            ),
        ),
        yaxis=dict(
            title="Y Axis",
            tickfont=dict(
                size=15,
                family="Arial, sans-serif",
                color="black",
                weight="bold",
            ),
        ),
        width=width,
        height=height,
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    filename_to_sav_img = os.path.join(
        output_dir, "images", foldername, organism, f"{filename_to_sav}.html"
    )
    filename_to_sav_csv = os.path.join(
        output_dir, "files", foldername, organism, f"{filename_to_sav}.csv"
    )

    pio.write_html(fig, file=filename_to_sav_img, auto_open=True)

    # eps format
    filename_to_sav_img_eps = os.path.join(
        output_dir, "images", foldername, organism, f"{filename_to_sav}.eps"
    )
    fig.write_image(filename_to_sav_img_eps, format="eps", engine="kaleido", scale=2)
    df.to_csv(f"{filename_to_sav_csv}.csv")
