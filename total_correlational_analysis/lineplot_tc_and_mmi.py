import plotly.graph_objects as go
import os
import json
import numpy as np
from plotly.subplots import make_subplots
from scientificplotter import PublicationFormatterConfig
from path_utils.path_handler import ProjectPaths


# Load the total correlation triplets data
with open(
    r"C:\Users\dhana\OneDrive\Documents\multivariate_ptms_analysis-\output\files\total correlation analysis\human\total_correlation_triplets.json",
    "r",
) as f:
    # with open(r'C:\Users\dhana\OneDrive\Documents\multivariate_ptms_analysis-\output\files\total correlation analysis\yeast\total_correlation_triplets.json', 'r') as f:
    tc_data = json.load(f)

# Load the MMI data, MMI values are sorted in the ascending order.
# with open('yeast_mmi_sorted_asc.json', 'r') as f:
output_file_folder_name = "total correlation analysis"
output_abs_path = os.path.join(
    ProjectPaths.get_output_files_dir(),
    output_file_folder_name,
    "human_mmi_sorted_asc.json",
)  # change the name of the organism, when needed
with open(output_abs_path, "r") as f:
    mmi_data = json.load(f)

# loading plot config
with open("total_correlational_analysis/tc_line_plot_config.json", "r") as f:
    plot_config = json.load(f)

# Find common triplets between both datasets
tc_keys = set(tc_data.keys())
mmi_keys = set(mmi_data.keys())
common_keys = tc_keys.intersection(mmi_keys)

print(f"Total correlation triplets: {len(tc_keys)}")
print(f"MMI triplets: {len(mmi_keys)}")
print(f"Common triplets: {len(common_keys)}")

# Create data for common triplets only
common_triplets_data = []
for triplet in common_keys:
    common_triplets_data.append((triplet, tc_data[triplet], mmi_data[triplet]))

# Sort by total correlation values (ascending order)
common_triplets_data.sort(key=lambda x: x[1])

# Extract sorted data
triplets = [item[0] for item in common_triplets_data]
tc_values = [item[1] for item in common_triplets_data]
mmi_values = [item[2] for item in common_triplets_data]

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

# Fix the trace styling - remove conflicting marker line settings:

# Add Total Correlation trace with proper color visibility
fig.add_trace(
    go.Scatter(
        x=list(range(len(tc_values))),
        y=tc_values,
        mode="lines+markers",
        name="TC Value",
        line=dict(
            color=colors["tc"],  # This should show the blue color
            width=2,  # Increased width for better visibility
            dash="solid",  # Solid line style
        ),
        marker=dict(
            size=6,
            color=colors["tc"],  # Same color as line
            line=dict(width=0),  # Remove white border - this was causing the issue!
            opacity=1.0,
        ),
        yaxis="y",
        hovertemplate="<b>Triplet:</b> %{customdata}<br>"
        + "<b>TC Value:</b> %{y:.4f}<extra></extra>",
        customdata=clean_triplets,
        showlegend=True,
    )
)

# Add MMI trace with proper color visibility
fig.add_trace(
    go.Scatter(
        x=list(range(len(mmi_values))),
        y=mmi_values,
        mode="lines+markers",
        name="MMI Value",
        line=dict(
            color=colors["mmi"],  # This should show the red/magenta color
            width=2,  # Increased width for better visibility
            dash="dot",  # Dot line style for distinction
        ),
        marker=dict(
            size=6,
            color=colors["mmi"],  # Same color as line
            line=dict(width=0),  # Remove white border - this was causing the issue!
            opacity=1.0,
        ),
        yaxis="y2",
        hovertemplate="<b>Triplet:</b> %{customdata}<br>"
        + "<b>MMI Value:</b> %{y:.4f}<extra></extra>",
        customdata=clean_triplets,
        showlegend=True,
    )
)

# Update layout with minimal styling - only x-axis line visible
fig.update_layout(
    # X-axis styling - remove tick labels, update title
    xaxis=dict(
        title=dict(
            text="Histone PTM Triplets (Increasing Order of TC)",
            font=dict(
                size=14, family="Arial, sans-serif", color="black", weight="bold"
            ),  # Added bold
        ),
        showgrid=False,  # Remove grid
        linecolor="black",
        linewidth=2,
        showline=True,  # Keep x-axis line
        mirror=False,  # Remove top line
        ticks="",  # Remove tick marks
        showticklabels=False,  # Remove tick labels
        tickfont=dict(
            size=12, family="Arial, sans-serif", color="black", weight="bold"
        ),  # Added bold
    ),
    # Primary Y-axis (Total Correlation) - remove all lines
    yaxis=dict(
        title=dict(
            text="Total Correlation (TC, bits)",
            font=dict(
                size=14, family="Arial, sans-serif", color=colors["tc"], weight="bold"
            ),  # Added bold
        ),
        titlefont=dict(color=colors["tc"], weight="bold"),  # Added bold
        tickfont=dict(
            size=12, family="Arial, sans-serif", color=colors["tc"], weight="bold"
        ),  # Added bold
        showgrid=False,  # Remove grid
        side="left",
        linecolor="white",  # Hide y-axis line
        linewidth=0,
        showline=False,  # Hide y-axis line
        mirror=False,  # Remove right line
        ticks="outside",
        tickwidth=1,
        tickcolor=colors["tc"],
        zeroline=True,
        zerolinecolor=colors["zero_line"],
        zerolinewidth=1,
    ),
    # Secondary Y-axis (MMI) - remove duplicate 'overlaying' parameter
    yaxis2=dict(
        title=dict(
            text="MMI (bits)",
            font=dict(
                size=14, family="Arial, sans-serif", color=colors["mmi"], weight="bold"
            ),
            standoff=15,  # Distance from axis to title
        ),
        titlefont=dict(color=colors["mmi"], weight="bold"),
        tickfont=dict(
            size=12, family="Arial, sans-serif", color=colors["mmi"], weight="bold"
        ),
        overlaying="y",  # Keep only this one
        side="right",
        showgrid=False,
        linecolor="white",
        linewidth=0,
        showline=False,
        mirror=False,
        ticks="outside",
        tickwidth=1.5,  # Slightly thicker tick marks
        ticklen=6,  # Length of tick marks
        tickcolor=colors["mmi"],
        tickmode="auto",  # Automatic tick spacing
        nticks=6,  # Approximate number of ticks
        # tickformat='.3f',   # 3 decimal places for ticks
        tickangle=0,  # Horizontal tick labels
        ticklabelposition="outside right",  # Position labels outside on right
        zeroline=True,
        zerolinecolor=colors["zero_line"],
        zerolinewidth=2,  # Slightly thicker zero line
        # Fine-tune positioning
        anchor="x",  # Anchor to x-axis
        constraintoward="right",  # Constrain expansion toward right (removed duplicate overlaying)
    ),
    # General layout
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1000,
    height=600,
    font=dict(
        size=12, family="Arial, sans-serif", color="black", weight="bold"
    ),  # Added bold
    # Legend styling
    showlegend=True,
    legend=dict(
        x=0.04,  # Close to right edge
        y=0.995,  # High up but below annotations
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.92)",  # Slightly transparent
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12, family="Arial, sans-serif", weight="bold"),
    ),
    # Hover settings
    hovermode="x unified",
    # Margins for academic presentation
    margin=dict(l=60, r=60, t=10, b=60),  # Reduced from l=80, r=80, t=60, b=80
)

# Add academic-quality annotations for extreme values with more breathing space
min_tc_idx = np.argmin(tc_values)
max_tc_idx = np.argmax(tc_values)
min_mmi_idx = np.argmin(mmi_values)
max_mmi_idx = np.argmax(mmi_values)

# Minimal, clean annotations for extreme values with increased spacing
fig.add_annotation(
    x=min_tc_idx,
    y=tc_values[min_tc_idx],
    text=f"Min TC<br>{tc_values[min_tc_idx]:.3f}",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["tc"],
    arrowwidth=1.5,
    ax=80,  # Increased from 40 to 80
    ay=-60,  # Increased from -40 to -80
    bgcolor="white",
    bordercolor=colors["tc"],
    borderwidth=1,
    yref="y",
    font=dict(size=14, family="Arial, sans-serif"),
)

fig.add_annotation(
    x=max_tc_idx,
    y=tc_values[max_tc_idx],
    text=f"Max TC<br>{tc_values[max_tc_idx]:.3f}",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["tc"],
    arrowwidth=1.5,
    ax=-80,  # Increased from -40 to -80
    ay=-60,  # Increased from -40 to -40
    bgcolor="white",
    bordercolor=colors["tc"],
    borderwidth=1,
    yref="y",
    font=dict(size=14, family="Arial, sans-serif"),
)

fig.add_annotation(
    x=min_mmi_idx,
    y=mmi_values[min_mmi_idx],
    text=f"Min MMI<br>{mmi_values[min_mmi_idx]:.3f}",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["mmi"],
    arrowwidth=1.5,
    ax=50,  # Increased from 40 to 80
    ay=60,  # Increased from 40 to 80
    bgcolor="white",
    bordercolor=colors["mmi"],
    borderwidth=1,
    yref="y2",
    font=dict(size=14, family="Arial, sans-serif"),
)

fig.add_annotation(
    x=max_mmi_idx,
    y=mmi_values[max_mmi_idx],
    text=f"Max MMI<br>{mmi_values[max_mmi_idx]:.3f}",
    showarrow=True,
    arrowhead=2,
    arrowcolor=colors["mmi"],
    arrowwidth=1.5,
    ax=-80,  # Increased from -40 to -80
    ay=60,  # Increased from 40 to 80
    bgcolor="white",
    bordercolor=colors["mmi"],
    borderwidth=1,
    yref="y2",
    font=dict(size=14, family="Arial, sans-serif"),
)

# # Add simple region labels with more spacing from edges
# if mmi_max > 0:
#     fig.add_annotation(
#         x=x_range * 0.01,  # Increased from 0.08 to 0.12
#         y=mmi_max * 0.5,
#         text="+ve MMI",
#         showarrow=False,
#         font=dict(size=12, family='Arial, sans-serif', color=colors['positive']),
#         bgcolor="rgba(255,255,255,0.9)",  # Slightly more opaque
#         bordercolor=colors['positive'],
#         borderwidth=1,
#         yref='y2'
#     )

# if mmi_min < 0:
#     fig.add_annotation(
#         x=x_range * 0.01,  # Increased from 0.08 to 0.12
#         y=mmi_min * 0.75,
#         text="-ve MMI",
#         showarrow=False,
#         font=dict(size=12, family='Arial, sans-serif', color=colors['negative']),
#         bgcolor="rgba(255,255,255,0.9)",  # Slightly more opaque
#         bordercolor=colors['negative'],
#         borderwidth=1,
#         yref='y2'
#     )

# # Journal-quality annotations with mathematical notation
# if mmi_max > 0:
#     fig.add_annotation(
#         x=x_range * 1.02,
#         y=mmi_max * 0.75,
#         # text="<b>I</b>(<b>X</b>;<b>Y</b>;<b>Z</b>) > 0<br><i>Synergistic Information</i>",
#         text = "+Ve MMI",
#         showarrow=False,
#         font=dict(size=11, family='Computer Modern, serif', color='black'),
#         bgcolor="rgba(245, 255, 245, 0.98)",
#         bordercolor='#1e7d32',
#         borderwidth=1.5,
#         borderpad=12,
#         yref='y2',
#         align='center',
#         # Professional spacing
#         xanchor='center',
#         yanchor='middle'
#     )

# if mmi_min < 0:
#     fig.add_annotation(
#         x=x_range * 1.02,
#         y=mmi_min * 0.75,
#         # text="<b>I</b>(<b>X</b>;<b>Y</b>;<b>Z</b>) < 0<br><i>Redundant Information</i>",
#         text= "-Ve MMI",
#         showarrow=False,
#         font=dict(size=11, family='Computer Modern, serif', color='black'),
#         bgcolor="rgba(255, 245, 245, 0.98)",
#         bordercolor='#c62828',
#         borderwidth=1.5,
#         borderpad=12,
#         yref='y2',
#         align='center',
#         # Professional spacing
#         xanchor='center',
#         yanchor='middle'
#     )

plot_config["legend_loc"] = (0.02, 1)
# fig.show()
# apply publication formatter config
formatter = PublicationFormatterConfig(**plot_config)

formatter.apply_to_figure(fig)


# Show the plot


# Save in academic publication formats
# fig.write_html("human_tc_vs_mmi_clean.html")
fig.write_image("human_tc_vs_mmi_clean.png", width=1000, height=600, scale=3)
fig.write_image("human_tc_vs_mmi_clean.pdf", width=1000, height=600)
# # fig.write_image("human_tc_vs_mmi_clean.svg", width=1000, height=600)

# # for yeats
# fig.write_image("yeast_tc_vs_mmi_clean.png", width=1000, height=600, scale=3)
# fig.write_image("yeast_tc_vs_mmi_clean.pdf", width=1000, height=600)
# fig.write_image("human_tc_vs_mmi_clean.svg", width=1000, height=600)


print(f"\nClean plots saved as:")
print(f"- human_tc_vs_mmi_clean.html (interactive)")
print(f"- human_tc_vs_mmi_clean.png (300 DPI)")
print(f"- human_tc_vs_mmi_clean.pdf (vector)")
print(f"- human_tc_vs_mmi_clean.svg (vector)")

# Print comprehensive statistics
print(f"\n=== FIGURE STATISTICS ===")
print(f"Total common triplets analyzed: {len(tc_values)}")
print(f"Total Correlation range: [{min(tc_values):.4f}, {max(tc_values):.4f}]")
print(f"MMI range: [{min(mmi_values):.4f}, {max(mmi_values):.4f}]")
print(
    f"Correlation coefficient (TC vs MMI): {np.corrcoef(tc_values, mmi_values)[0, 1]:.4f}"
)
