import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class ScatterConfig:
    """Configuration for scatter plot styling"""

    figure_width: int = 12
    figure_height: int = 8
    font_size: int = 10
    marker_size: int = 60
    alpha: float = 0.7
    edge_width: float = 0.8
    grid: bool = True
    grid_alpha: float = 0.3
    show_regression: bool = False
    show_correlation: bool = False
    confidence_interval: bool = False
    ci_level: float = 0.95
    dpi: int = 300
    legend_location: str = "best"


class ScatterColorPalette:
    """Enhanced color palette management for scatter plots"""

    # Scientific color schemes
    SCIENTIFIC = {
        "experiment": {"control": "#4e79a7", "treatment": "#e15759"},
        "correlation": {
            "positive": "#59a14f",
            "negative": "#e15759",
            "neutral": "#4e79a7",
        },
        "significance": {"high": "#e31a1c", "medium": "#fd8d3c", "low": "#fecc5c"},
    }

    # Journal color schemes
    PUBLICATION = {
        "nature": ["#4477AA", "#66CCEE", "#228833"],
        "science": ["#3B4992", "#EE0000", "#008B45"],
        "cell": ["#2271B2", "#D55E00", "#359B73"],
    }

    # Statistical color schemes
    STATISTICAL = {"pearson": "#4e79a7", "spearman": "#59a14f", "kendall": "#9c755f"}


class ScatterPlotter:
    """Scatter plot with dynamic styling"""

    def __init__(self, config: Optional[ScatterConfig] = None):
        self.config = config or ScatterConfig()

    def plot_scatter(
        self,
        x_data: Union[List[float], np.ndarray],
        y_data: Union[List[float], np.ndarray],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Union[str, Path],
        color_scheme: str = "pearson",
        palette_type: str = "STATISTICAL",
        show_plot: bool = True,
        custom_color: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Create publication-quality scatter plot
        """
        try:
            # Setup figure
            fig, ax = plt.subplots(
                figsize=(self.config.figure_width, self.config.figure_height)
            )

            # Get color
            color = (
                custom_color
                if custom_color
                else getattr(ScatterColorPalette, palette_type)[color_scheme]
            )

            # Create scatter plot
            scatter = ax.scatter(
                x_data,
                y_data,
                s=self.config.marker_size,
                alpha=self.config.alpha,
                c=color,
                linewidth=self.config.edge_width,
                edgecolor="white",
            )

            # Add regression line if requested
            if self.config.show_regression:
                sns.regplot(
                    x=x_data,
                    y=y_data,
                    scatter=False,
                    color="red",
                    ax=ax,
                    ci=(
                        self.config.ci_level
                        if self.config.confidence_interval
                        else None
                    ),
                )

            # Add correlation coefficient if requested
            if self.config.show_correlation:
                corr = np.corrcoef(x_data, y_data)[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"r = {corr:.3f}",
                    transform=ax.transAxes,
                    fontsize=self.config.font_size,
                    verticalalignment="top",
                )

            # Customize appearance
            if self.config.grid:
                ax.grid(True, alpha=self.config.grid_alpha, linestyle="--")

            # Set labels and title
            ax.set_title(title, pad=20, fontsize=self.config.font_size + 2)
            ax.set_xlabel(xlabel, fontsize=self.config.font_size)
            ax.set_ylabel(ylabel, fontsize=self.config.font_size)

            # Add legend if labels provided
            if labels is not None:
                legend_elements = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=8,
                        label=label,
                    )
                    for label, color in zip(
                        labels, [color] if isinstance(color, str) else color
                    )
                ]
                ax.legend(
                    handles=legend_elements,
                    loc=self.config.legend_location,
                    fontsize=self.config.font_size,
                )

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

            if show_plot:
                plt.show()

        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            raise
        finally:
            plt.close()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 0.5 * x + np.random.normal(0, 0.5, 100)

    # Basic scatter plot
    config = ScatterConfig(
        show_regression=True, show_correlation=True, confidence_interval=True
    )
    plotter = ScatterPlotter(config)

    # Create correlation plot
    plotter.plot_scatter(
        x_data=x,
        y_data=y,
        title="Correlation Scatter Plot",
        xlabel="X Values",
        ylabel="Y Values",
        save_path="correlation_scatter.png",
        color_scheme="pearson",
        palette_type="STATISTICAL",
    )
