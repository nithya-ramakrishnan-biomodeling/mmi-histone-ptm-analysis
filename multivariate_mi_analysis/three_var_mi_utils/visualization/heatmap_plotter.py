import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class HeatmapConfig:
    """Configuration for heatmap styling"""

    figure_width: int = 15
    figure_height: int = 10
    font_size: int = 10
    rotation: int = 45
    annotation_size: int = 8
    dpi: int = 300
    cbar_width: float = 0.03
    cbar_pad: float = 0.04


class HeatmapColorPalette:
    """Scientific heatmap color palette management"""

    SEQUENTIAL = {
        "viridis": ["#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"],
        "plasma": ["#0D0887", "#7E03A8", "#CC4778", "#F89540", "#F0F921"],
        "magma": ["#000004", "#51127C", "#B63679", "#FB8861", "#FCFDBF"],
    }

    DIVERGING = {
        "rwb": ["#67001F", "#B2182B", "#F4A582", "#F5F5F5", "#92C5DE", "#2166AC"],
        "coolwarm": ["#4A148C", "#7B1FA2", "#E1BEE7", "#F5F5F5", "#B3E5FC", "#01579B"],
    }


class HeatmapPlotter:
    """Scientific heatmap plotter with dynamic styling"""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        self.config = config or HeatmapConfig()
        plt.style.use("seaborn")

    def plot_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str,
        save_path: Union[str, Path],
        palette_type: str = "sequential",
        scheme: str = "viridis",
        show_annotations: bool = True,
        show_plot: bool = True,
    ) -> None:
        """
        Create publication-quality heatmap

        Args:
            data: 2D numpy array of values
            row_labels: List of row labels
            col_labels: List of column labels
            title: Plot title
            save_path: Path to save the plot
            palette_type: Type of color palette ('sequential' or 'diverging')
            scheme: Specific color scheme name
            show_annotations: Whether to show value annotations
            show_plot: Whether to display the plot
        """
        try:
            # Set figure size based on data dimensions
            aspect_ratio = data.shape[1] / data.shape[0]
            fig_width = self.config.figure_width
            fig_height = fig_width / aspect_ratio

            # Create figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Get color palette
            palette = getattr(HeatmapColorPalette, palette_type.upper())[scheme]
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom", palette)

            # Create heatmap
            sns.heatmap(
                data,
                xticklabels=col_labels,
                yticklabels=row_labels,
                cmap=cmap,
                annot=show_annotations,
                fmt=".2f",
                annot_kws={"size": self.config.annotation_size},
                cbar_kws={"shrink": 0.5},
                ax=ax,
            )

            # Customize appearance
            ax.set_title(title, pad=20, fontsize=self.config.font_size + 2)
            plt.xticks(rotation=self.config.rotation, ha="right")
            plt.yticks(rotation=0)

            # Adjust colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=self.config.font_size)

            # Adjust layout
            plt.tight_layout()

            # Save plot
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

            if show_plot:
                plt.show()

        except Exception as e:
            print(f"Error creating heatmap: {e}")
            raise
        finally:
            plt.close()


# Example usage:
if __name__ == "__main__":
    # Custom configuration
    config = HeatmapConfig(
        figure_width=15,
        figure_height=10,
        font_size=10,
        rotation=45,
        annotation_size=8,
        dpi=300,
    )

    # Initialize plotter
    plotter = HeatmapPlotter(config)

    # Create sample data
    data = np.random.rand(10, 8)
    row_labels = [f"Row_{i}" for i in range(10)]
    col_labels = [f"Col_{i}" for i in range(8)]

    # Create heatmap
    plotter.plot_heatmap(
        data=data,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Sample Heatmap",
        save_path="heatmap.png",
        palette_type="sequential",
        scheme="viridis",
    )
