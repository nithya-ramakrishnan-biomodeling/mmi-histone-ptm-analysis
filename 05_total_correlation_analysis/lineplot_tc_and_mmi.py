import plotly.graph_objects as go
import pandas as pd
import numpy as np
from multivariate_utils import ProjectPaths, load_data


# tc_mmi_path: output\files\total_correlation_analysis\tc_values_bin_based\yeast\histone_triplet_tc_bin70.csv

# 

tc_mmi_path =  ProjectPaths().get_output_files_dir().joinpath(
    "total_correlation_analysis",
    "tc_values_bin_based",
    "human",  # change the name of the organism, when needed
    "histone_triplet_tc_bin100.csv"
) 

df= pd.read_csv(tc_mmi_path, index_col=0)

# sort the index of the dataframe based on TC values
df_sorted = df.sort_values(by="tc_value", ascending=True)

triplets = df_sorted.index.tolist()
tc_values = df_sorted["tc_value"].tolist()
mmi_values = df_sorted["mmi_value"].tolist()

# Clean up triplet names for better display 
clean_triplets = []
for triplet in triplets:
    clean_triplet = triplet.strip("()").replace("'", "").replace(" ", "")
    clean_triplets.append(clean_triplet)

# Alternative: High-contrast color palette
colors = {
    "tc": "#1f77b4",  # Blue
    "mmi": "#ff7f0e",  # Brighter red
    "tc_light": "#87CEEB",
    "mmi_light": "#DDA0DD",
    "positive": "#2ca02c",  # Green
    "negative": "#d62728",
    "bg_positive": "rgba(144, 238, 144, 0.15)",  # Reduced opacity
    "bg_negative": "rgba(255, 182, 193, 0.15)",  # Reduced opacity
    "grid": "#E5E5E5",
    "zero_line": "#696969",
}

# 12. Colorblind-friendly colors

colors_cb = {
    "tc": "#1f77b4",  # Blue
    "mmi": "#ff7f0e",  # Orange (instead of red)
    "positive": "#2ca02c",  # Green
    "negative": "#d62728",  # Red
}

# colors(**colors_cb)  # Use colorblind-friendly palette

# Professional journal colors (Nature, Science, Cell style)
colors_nature = {
    "tc": "#0173b2",  # Deep blue (primary data)
    "mmi": "#de8f05",  # Amber/orange (secondary data)
    "positive": "#029e73",  # Teal green (positive regions)
    "negative": "#cc78bc",  # Muted purple (negative regions)
    "bg_positive": "rgba(2, 158, 115, 0.08)",  # Very light teal
    "bg_negative": "rgba(204, 120, 188, 0.08)",  # Very light purple
    "grid": "#E8E8E8",
    "zero_line": "#666666",
}

# High contrast for academic publications
colors_academic = {
    "tc": "#000080",  # Navy blue (excellent for print)
    "mmi": "#DC143C",  # Crimson red (high visibility)
    "positive": "#006400",  # Dark green (clear distinction)
    "negative": "#8B0000",  # Dark red (strong contrast)
    "bg_positive": "rgba(240, 255, 240, 0.4)",  # Light green wash
    "bg_negative": "rgba(255, 240, 240, 0.4)",  # Light red wash
    "grid": "#D3D3D3",
    "zero_line": "#404040",
}

# Scientific publication inspired by Viridis (perceptually uniform)
colors_viridis = {
    "tc": "#440154",  # Deep purple
    "mmi": "#21908c",  # Teal
    "positive": "#35b779",  # Green
    "negative": "#fde725",  # Yellow
    "bg_positive": "rgba(53, 183, 121, 0.1)",  # Light green
    "bg_negative": "rgba(253, 231, 37, 0.1)",  # Light yellow
    "grid": "#E8E8E8",
    "zero_line": "#666666",
}

colors = colors_nature  # Use high contrast colors
# Create figure with academic styling
fig = go.Figure()

# Find MMI value range for background regions
mmi_min, mmi_max = min(mmi_values), max(mmi_values)
x_range = len(mmi_values)

# Add background regions FIRST (so they appear behind the data)
if mmi_min < 0:
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=x_range - 0.5,
        y0=mmi_min * 1.05,
        y1=0,
        xref="x",
        yref="y2",
        fillcolor=colors["bg_negative"],
        line=dict(width=0),
        layer="below",
    )

if mmi_max > 0:
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=x_range - 0.5,
        y0=0,
        y1=mmi_max * 1.05,
        xref="x",
        yref="y2",
        fillcolor=colors["bg_positive"],
        line=dict(width=0),
        layer="below",
    )

# Add Total Correlation trace with bolder styling
fig.add_trace(
    go.Scatter(
        x=list(range(len(tc_values))),
        y=tc_values,
        mode="lines+markers",
        name="TC Value",
        line=dict(
            color=colors["tc"],
            width=3,
            dash="solid",
        ),
        marker=dict(
            size=8,
            color=colors["tc"],
            line=dict(width=0),
            opacity=1.0,
        ),
        yaxis="y",
        hovertemplate="<b>Triplet:</b> %{customdata}<br><b>TC Value:</b> %{y:.4f}<extra></extra>",
        customdata=clean_triplets,
        showlegend=True,
    )
)

# Add MMI trace with bolder styling
fig.add_trace(
    go.Scatter(
        x=list(range(len(mmi_values))),
        y=mmi_values,
        mode="lines+markers",
        name="MMI Value",
        line=dict(
            color=colors["mmi"],
            width=3,
            dash="dot",
        ),
        marker=dict(
            size=7,
            color=colors["mmi"],
            line=dict(width=0),
            opacity=1.0,
        ),
        yaxis="y2",
        hovertemplate="<b>Triplet:</b> %{customdata}<br><b>MMI Value:</b> %{y:.4f}<extra></extra>",
        customdata=clean_triplets,
        showlegend=True,
    )
)

# Update layout with bold, slightly larger fonts everywhere
fig.update_layout(
    xaxis=dict(
        title=dict(
            text="<b>Histone PTM Triplets (Increasing Order of TC)</b>",
            font=dict(size=16, family="Arial Black, Arial, sans-serif", color="black"),
        ),
        showgrid=False,
        linecolor="black",
        linewidth=2,
        showline=True,
        mirror=False,
        ticks="",
        showticklabels=False,
        tickfont=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
    ),
    yaxis=dict(
        title=dict(
            text="<b>Total Correlation (TC, bits)</b>",
            font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
        ),
        tickfont=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
        showgrid=False,
        side="left",
        linecolor="white",
        linewidth=0,
        showline=False,
        mirror=False,
        ticks="outside",
        tickwidth=1,
        tickcolor="black",
        zeroline=True,
        zerolinecolor=colors["zero_line"],
        zerolinewidth=1,
    ),
    yaxis2=dict(
        title=dict(
            text="<b>MMI (bits)</b>",
            font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
            standoff=15,
        ),
        tickfont=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
        overlaying="y",
        side="right",
        showgrid=False,
        linecolor="white",
        linewidth=0,
        showline=False,
        mirror=False,
        ticks="outside",
        tickwidth=1.5,
        ticklen=6,
        tickcolor="black",
        tickmode="auto",
        nticks=6,
        tickangle=0,
        ticklabelposition="outside right",
        zeroline=True,
        zerolinecolor=colors["zero_line"],
        zerolinewidth=2,
        anchor="x",
        constraintoward="right",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1000,
    height=600,
    font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
    showlegend=True,
    legend=dict(
        x=0.04,
        y=0.995,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
    ),
    hovermode="x unified",
    margin=dict(l=60, r=60, t=10, b=60),
)

# Add academic-quality annotations for extreme values with bolder fonts
min_tc_idx = np.argmin(tc_values)
max_tc_idx = np.argmax(tc_values)
min_mmi_idx = np.argmin(mmi_values)
max_mmi_idx = np.argmax(mmi_values)

fig.add_annotation(
    x=min_tc_idx,
    y=tc_values[min_tc_idx],
    text=f"<b>Min TC</b><br><b>{tc_values[min_tc_idx]:.3f}</b>",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["tc"],
    arrowwidth=1.5,
    ax=80,
    ay=-60,
    bgcolor="white",
    bordercolor=colors["tc"],
    borderwidth=1,
    yref="y",
    font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
)

fig.add_annotation(
    x=max_tc_idx,
    y=tc_values[max_tc_idx],
    text=f"<b>Max TC</b><br><b>{tc_values[max_tc_idx]:.3f}</b>",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["tc"],
    arrowwidth=1.5,
    ax=-80,
    ay=-60,
    bgcolor="white",
    bordercolor=colors["tc"],
    borderwidth=1,
    yref="y",
    font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
)

fig.add_annotation(
    x=min_mmi_idx,
    y=mmi_values[min_mmi_idx],
    text=f"<b>Min MMI</b><br><b>{mmi_values[min_mmi_idx]:.3f}</b>",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["mmi"],
    arrowwidth=1.5,
    ax=50,
    ay=60,
    bgcolor="white",
    bordercolor=colors["mmi"],
    borderwidth=1,
    yref="y2",
    font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
)

fig.add_annotation(
    x=max_mmi_idx,
    y=mmi_values[max_mmi_idx],
    text=f"<b>Max MMI</b><br><b>{mmi_values[max_mmi_idx]:.3f}</b>",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["mmi"],
    arrowwidth=1.5,
    ax=-80,
    ay=60,
    bgcolor="white",
    bordercolor=colors["mmi"],
    borderwidth=1,
    yref="y2",
    font=dict(size=18, family="Arial Black, Arial, sans-serif", color="black"),
)

# Save outputs
# fig.write_image("yeast_tc_vs_mmi_clean.png", width=1000, height=600, scale=3)
# fig.write_image("yeast_tc_vs_mmi_clean.pdf", width=1000, height=600)

fig.write_image("human_tc_vs_mmi_clean.png", width=1000, height=600, scale=3)
fig.write_image("human_tc_vs_mmi_clean.pdf", width=1000, height=600)