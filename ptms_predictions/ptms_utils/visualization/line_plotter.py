import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class BaseConfig:
    """Base configuration for all plot types"""

    figure_width: int = 12
    figure_height: int = 8
    font_size: int = 10
    dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.3


@dataclass
class LineConfig(BaseConfig):
    """Enhanced configuration for line plot styling"""

    line_width: float = 2.0
    marker_size: int = 6
    alpha: float = 1.0
    show_markers: bool = True
    show_annotations: bool = False
    annotation_offset: float = 0.1
    annotation_size: int = 8
    legend_location: str = "best"
    x_label_rotation: int = 45
    error_bars: bool = False
    error_bar_alpha: float = 0.3
    confidence_interval: bool = False
    ci_level: float = 0.95
    legend_columns: int = 1
    legend_frame: bool = True
    y_limit: Optional[Tuple[float, float]] = None
    x_limit: Optional[Tuple[float, float]] = None


class LineColorPalette:
    """Enhanced color palette management for scientific visualization"""

    SCIENTIFIC = {
        "nature": ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"],
        "science": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021"],
        "cell": ["#2271B2", "#D55E00", "#359B73", "#E69F00", "#CC79A7", "#0072B2"],
    }

    CATEGORICAL = {
        "pastel": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff"],
        "deep": ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"],
        "muted": ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4"],
    }

    SEQUENTIAL = {
        "blues": ["#deebf7", "#9ecae1", "#3182bd"],
        "reds": ["#fee0d2", "#fc9272", "#de2d26"],
        "greens": ["#e5f5e0", "#a1d99b", "#31a354"],
    }

    R2_SPECIFIC = {
        "correlation": [
            "#001F3F",
            "#0074D9",
            "#7FDBFF",
            "#FFFFFF",
            "#FF851B",
            "#FF4136",
            "#85144B",
        ],
        "evaluation": [
            "#4e79a7",
            "#f28e2c",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#edc949",
        ],
        "comparison": ["#3B9AB2", "#FFFFFF", "#E31A1C"],
    }

    COLORBLIND = {
        "wong": [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ],
        "okabe": [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ],
    }

    # Add more journal-specific color schemes
    JOURNALS = {
        # Existing
        "nature": ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"],
        "science": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021"],
        "cell": ["#2271B2", "#D55E00", "#359B73", "#E69F00", "#CC79A7", "#0072B2"],
        # New additions
        "pnas": ["#FF9900", "#109618", "#990099", "#3B3EAC", "#0099C6", "#DD4477"],
        "nejm": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD"],
        "lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91"],
        "jama": ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599"],
        "bmj": ["#005AAA", "#E64A19", "#FBC02D", "#00897B", "#7B1FA2", "#455A64"],
    }

    # Field-specific palettes
    FIELD_SPECIFIC = {
        "biochemistry": ["#2E8B57", "#8B0000", "#4682B4", "#FF8C00", "#9370DB"],
        "molecular": ["#4DAF4A", "#984EA3", "#FF7F00", "#377EB8", "#E41A1C"],
        "genomics": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"],
        "proteomics": ["#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0"],
        "immunology": ["#CBD5E8", "#B3E2CD", "#FDCDAC", "#F4CAE4", "#E6F5C9"],
    }

    # High-contrast palettes for accessibility
    ACCESSIBLE = {
        "high_contrast": [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
        ],
        "deuteranopia": [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
        ],
        "protanopia": [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#CC79A7",
            "#0072B2",
        ],
        "tritanopia": [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#D55E00",
        ],
    }

    # Specialized statistical palettes
    STATISTICAL = {
        "p_values": ["#FFFFFF", "#FEE0D2", "#FC9272", "#DE2D26"],
        "correlation": [
            "#67001F",
            "#B2182B",
            "#F4A582",
            "#FFFFFF",
            "#92C5DE",
            "#2166AC",
        ],
        "significance": [
            "#FFFFFF",
            "#FFF5F0",
            "#FEE0D2",
            "#FCC5C0",
            "#FA9FB5",
            "#F768A1",
            "#DD3497",
            "#AE017E",
        ],
        "regression": [
            "#4A148C",
            "#7B1FA2",
            "#E1BEE7",
            "#F5F5F5",
            "#B3E5FC",
            "#01579B",
        ],
    }


class PlotLogger:
    """Logging for plot creation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


class PlotValidator:
    """Validate plot inputs and configurations"""

    @staticmethod
    def validate_data(data: Dict[str, Tuple[List[float], List[float]]]) -> bool:
        """Validate input data"""
        if not data:
            raise ValueError("Empty data dictionary provided")
        for name, (x, y) in data.items():
            if len(x) != len(y):
                raise ValueError(f"X and Y lengths don't match for {name}")
        return True


class StyleManager:
    """Manage plot styles across different types"""

    @staticmethod
    def set_publication_style(
        ax: plt.Axes, journal: str = "nature", fontsize: int = 10
    ) -> None:
        """Apply journal-specific styling"""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=fontsize)


class PlotExporter:
    """Handle plot saving and format conversion"""

    @staticmethod
    def save_plot(
        fig: plt.Figure,
        path: Union[str, Path],
        formats: List[str] = ["png", "pdf", "svg", "eps"],
        dpi: int = 300,
    ) -> None:
        """Save plot in multiple formats"""
        base_path = str(path).rsplit(".", 1)[0]
        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            if fmt == "eps":
                fig.savefig(
                    save_path,
                    format="eps",
                    dpi=dpi,
                    bbox_inches="tight",
                    transparent=True,
                )
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


class LinePlotter:
    """Enhanced line plotter with multiple line support"""

    def __init__(self, config: Optional[LineConfig] = None):
        self.config = config or LineConfig()
        self.logger = PlotLogger().logger
        self.validator = PlotValidator()
        self.exporter = PlotExporter()

    def plot_lines(
        self,
        data: Dict[str, Tuple[List[float], List[float]]],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Union[str, Path],
        palette_type: str = "SCIENTIFIC",
        scheme: str = "nature",
        markers: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, List[str]]] = None,
        publication_style: bool = False,
        export_formats: Optional[List[str]] = None,
    ) -> None:
        """
        Create publication-quality line plot with multiple lines

        Args:
            data: Dictionary with line names and (x, y) value tuples
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
            palette_type: Type of color palette
            scheme: Color scheme name
            markers: Dictionary of line names and marker styles
            annotations: Dictionary of line names and annotation texts
            publication_style: Whether to use publication styling
            export_formats: List of export formats
        """
        try:
            # Setup figure
            fig, ax = plt.subplots(
                figsize=(self.config.figure_width, self.config.figure_height)
            )

            # Get colors
            colors = getattr(LineColorPalette, palette_type)[scheme]

            # Plot each line
            for idx, (line_name, (x_values, y_values)) in enumerate(data.items()):
                color = colors[idx % len(colors)]
                marker = markers.get(line_name, "o") if markers else "o"

                # Plot line
                line = ax.plot(
                    x_values,
                    y_values,
                    label=line_name,
                    color=color,
                    linewidth=self.config.line_width,
                    alpha=self.config.alpha,
                    marker=marker if self.config.show_markers else None,
                    markersize=self.config.marker_size,
                )

                # Add annotations if provided
                if (
                    self.config.show_annotations
                    and annotations
                    and line_name in annotations
                ):
                    for x, y, text in zip(x_values, y_values, annotations[line_name]):
                        ax.annotate(
                            text,
                            (x, y),
                            xytext=(0, self.config.annotation_offset),
                            textcoords="offset points",
                            ha="center",
                            fontsize=self.config.annotation_size,
                        )

            # Customize appearance
            if self.config.grid:
                ax.grid(True, alpha=self.config.grid_alpha, linestyle="--")

            # Set labels and title
            ax.set_title(title, pad=20, fontsize=self.config.font_size + 2)
            ax.set_xlabel(xlabel, fontsize=self.config.font_size)
            ax.set_ylabel(ylabel, fontsize=self.config.font_size)

            # Rotate x-labels if needed
            plt.xticks(rotation=self.config.x_label_rotation)

            # Add legend
            self._setup_legend(ax, len(data))

            # Publication styling
            if publication_style:
                StyleManager.set_publication_style(ax, fontsize=self.config.font_size)

            # Save plot
            if export_formats:
                self.exporter.save_plot(
                    fig, save_path, formats=export_formats, dpi=self.config.dpi
                )
            else:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

            self.logger.info(f"Successfully created line plot: {save_path}")

        except Exception as e:
            self.logger.error(f"Error creating line plot: {e}")
            raise
        finally:
            plt.close()

    def _setup_legend(self, ax: plt.Axes, n_lines: int) -> None:
        """Configure legend based on number of lines"""
        if n_lines <= 3:
            legend_config = {
                "bbox_to_anchor": (1.05, 1),
                "loc": "upper left",
                "borderaxespad": 0,
                "ncol": self.config.legend_columns,
            }
        else:
            legend_config = {
                "bbox_to_anchor": (0.5, -0.2),
                "loc": "upper center",
                "borderaxespad": 0,
                "ncol": min(3, n_lines // 2),
            }
        ax.legend(**legend_config, frameon=self.config.legend_frame)

    def add_error_bars(
        self,
        ax: plt.Axes,
        x_values: List[float],
        y_values: List[float],
        errors: List[float],
        color: str,
    ) -> None:
        """Add error bars or confidence intervals"""
        if self.config.error_bars:
            ax.fill_between(
                x_values,
                [y - e for y, e in zip(y_values, errors)],
                [y + e for y, e in zip(y_values, errors)],
                alpha=self.config.error_bar_alpha,
                color=color,
            )


# Example usage
if __name__ == "__main__":
    # Generate sample data
    x = np.linspace(0, 10, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = -np.sin(x)

    # Prepare data dictionary
    data = {
        "Sin": (x.tolist(), y1.tolist()),
        "Cos": (x.tolist(), y2.tolist()),
        "Negative Sin": (x.tolist(), y3.tolist()),
    }

    # Optional markers
    markers = {"Sin": "o", "Cos": "s", "Negative Sin": "^"}

    # Optional annotations
    annotations = {
        "Sin": [
            "max" if i == np.argmax(y1) else "min" if i == np.argmin(y1) else ""
            for i in range(len(x))
        ]
    }

    # Add error data
    errors = {"Sin": [0.1] * len(x), "Cos": [0.15] * len(x)}

    config = LineConfig(
        show_markers=True,
        show_annotations=True,
        line_width=2.0,
        marker_size=8,
        error_bars=True,
        error_bar_alpha=0.2,
        legend_columns=2,
        y_limit=(-1.5, 1.5),
    )

    plotter = LinePlotter(config)

    # Create R² style plot
    plotter.plot_lines(
        data=data,
        title="R² Comparison",
        xlabel="Features",
        ylabel="R² Score",
        save_path="r2_comparison_plot",
        palette_type="R2_SPECIFIC",
        scheme="evaluation",
        markers=markers,
        annotations=annotations,
        publication_style=True,
        export_formats=["png", "pdf", "svg", "eps"],
    )

    # Using journal-specific palette
    plotter.plot_lines(
        data=data,
        title="Journal-Style Plot",
        xlabel="Time",
        ylabel="Expression",
        save_path="journal_plot",
        palette_type="JOURNALS",
        scheme="nejm",
    )

    # Using field-specific palette
    plotter.plot_lines(
        data=data,
        title="Proteomics Analysis",
        xlabel="Protein ID",
        ylabel="Abundance",
        save_path="proteomics_plot",
        palette_type="FIELD_SPECIFIC",
        scheme="proteomics",
    )
