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
    rotation: int = 45
    grid: bool = True
    grid_alpha: float = 0.3


@dataclass
class BarGraphConfig(BaseConfig):
    """Configuration for bar graph styling extending BaseConfig"""

    bar_width: float = 0.8
    annotation_size: int = 8
    error_bar_style: str = "bar"
    show_values: bool = True


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


class PlotValidator:
    """Validate plot inputs and configurations"""

    @staticmethod
    def validate_data(data: Dict[str, Union[float, Tuple[float, float]]]) -> bool:
        """Validate input data"""
        if not data:
            raise ValueError("Empty data dictionary provided")
        return True

    @staticmethod
    def validate_colors(colors: List[str], data_length: int) -> bool:
        """Validate color list length"""
        if len(colors) < data_length:
            raise ValueError(
                f"Not enough colors ({len(colors)}) for data points ({data_length})"
            )
        return True


class PlotExporter:
    """Handle plot saving and format conversion"""

    @staticmethod
    def save_plot(
        fig: plt.Figure,
        path: Union[str, Path],
        formats: List[str] = ["png", "pdf", "svg"],
        dpi: int = 300,
    ) -> None:
        """Save plot in multiple formats"""
        base_path = str(path).rsplit(".", 1)[0]
        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


class PlotLogger:
    """Logging for plot creation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


class BarGraphColorPalette:
    """Enhanced scientific color palette management for bar graphs"""

    # Single color schemes
    SOLID = {
        "blue": ["#4e79a7"],
        "red": ["#e15759"],
        "green": ["#59a14f"],
        "purple": ["#b07aa1"],
        "orange": ["#ff9d9a"],
    }

    # Multi-color schemes
    MULTI = {
        "default": [
            "#4e79a7",
            "#f28e2c",
            "#e15759",
            "#76b7b2",
            "#59a14f",
            "#edc949",
            "#af7aa1",
            "#ff9da7",
            "#9c755f",
            "#bab0ab",
        ],
        "pastel": [
            "#a1c9f4",
            "#ffb482",
            "#8de5a1",
            "#ff9f9b",
            "#d0bbff",
            "#debb9b",
            "#fab0e4",
            "#cfcfcf",
            "#fffea3",
            "#b9f2f0",
        ],
        "dark": [
            "#1b1b1b",
            "#242424",
            "#2d2d2d",
            "#363636",
            "#3f3f3f",
            "#484848",
            "#515151",
            "#5a5a5a",
            "#636363",
            "#6c6c6c",
        ],
    }

    # Publication color schemes
    PUBLICATION = {
        # Existing schemes
        "nature": ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"],
        "science": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021"],
        "cell": ["#2271B2", "#D55E00", "#359B73", "#E69F00", "#CC79A7", "#0072B2"],
        # New scientific schemes
        "nejm": [
            "#BC3C29",
            "#0072B5",
            "#E18727",
            "#20854E",
            "#7876B1",
            "#6F99AD",
        ],  # New England Journal
        "lancet": [
            "#00468B",
            "#ED0000",
            "#42B540",
            "#0099B4",
            "#925E9F",
            "#FDAF91",
        ],  # Lancet
        "jama": [
            "#374E55",
            "#DF8F44",
            "#00A1D5",
            "#B24745",
            "#79AF97",
            "#6A6599",
        ],  # JAMA
        "pnas": [
            "#FF9900",
            "#109618",
            "#990099",
            "#3B3EAC",
            "#0099C6",
            "#DD4477",
        ],  # PNAS
    }

    # Colorblind-friendly schemes
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
        ],  # Wong's palette
        "okabe_ito": [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ],  # Okabe-Ito
        "tol": [
            "#332288",
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#999933",
            "#DDCC77",
            "#CC6677",
            "#882255",
            "#AA4499",
        ],  # Paul Tol's
    }

    # Emphasis schemes (for highlighting specific bars)
    EMPHASIS = {
        "red_highlight": ["#E6E6E6", "#E6E6E6", "#E6E6E6", "#FF0000", "#E6E6E6"],
        "blue_highlight": ["#E6E6E6", "#E6E6E6", "#0066CC", "#E6E6E6", "#E6E6E6"],
        "contrast": ["#CCCCCC", "#CCCCCC", "#E31A1C", "#1F78B4", "#CCCCCC"],
    }

    # Sequential scientific schemes
    SEQUENTIAL = {
        "impact": ["#f0f0f0", "#bdbdbd", "#636363", "#252525"],  # Impact factor scale
        "pvalue": ["#fcbba1", "#fb6a4a", "#de2d26", "#a50f15"],  # P-value scale
        "quality": ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],  # Quality score
    }

    @staticmethod
    def create_emphasis_palette(
        highlight_positions: List[int],
        n_bars: int,
        highlight_color: str = "#FF0000",
        base_color: str = "#E6E6E6",
    ) -> List[str]:
        """Create custom emphasis palette

        Args:
            highlight_positions: Positions to highlight (0-based)
            n_bars: Total number of bars
            highlight_color: Color for highlighted bars
            base_color: Color for non-highlighted bars
        """
        colors = [base_color] * n_bars
        for pos in highlight_positions:
            if 0 <= pos < n_bars:
                colors[pos] = highlight_color
        return colors

    @staticmethod
    def create_gradient_palette(
        start_color: str, end_color: str, n_colors: int
    ) -> List[str]:
        """Create gradient palette between two colors"""
        import matplotlib.colors as mcolors

        # Convert hex to RGB
        rgb1 = mcolors.hex2color(start_color)
        rgb2 = mcolors.hex2color(end_color)

        # Create gradient
        colors = [
            tuple(r1 + (r2 - r1) * i / (n_colors - 1) for r1, r2 in zip(rgb1, rgb2))
            for i in range(n_colors)
        ]

        # Convert back to hex
        return [mcolors.rgb2hex(c) for c in colors]


class BarGraphPlotter:
    """Enhanced bar graph plotter with dynamic styling"""

    def __init__(self, config: Optional[BarGraphConfig] = None):
        self.config = config or BarGraphConfig()
        self.logger = PlotLogger().logger
        self.validator = PlotValidator()
        self.exporter = PlotExporter()

    def plot_bars(
        self,
        data: Dict[str, Union[float, Tuple[float, float]]],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: Union[str, Path],
        color_scheme: str = "default",
        palette_type: str = "MULTI",
        show_plot: bool = True,
        horizontal: bool = False,
        custom_colors: Optional[List[str]] = None,
        publication_style: bool = False,
        export_formats: Optional[List[str]] = None,
    ) -> None:
        """Create publication-quality bar graph with enhanced features"""
        try:
            # Validate inputs
            self.validator.validate_data(data)

            # Setup figure
            fig, ax = plt.subplots(
                figsize=(self.config.figure_width, self.config.figure_height)
            )

            # Apply publication style if requested
            if publication_style:
                StyleManager.set_publication_style(ax, fontsize=self.config.font_size)

            # Get colors - handle custom colors
            if custom_colors is not None:
                colors = custom_colors
            else:
                colors = getattr(BarGraphColorPalette, palette_type)[color_scheme]

            # Prepare data
            labels = list(data.keys())
            if isinstance(list(data.values())[0], tuple):
                values = [v[0] for v in data.values()]
                errors = [v[1] for v in data.values()]
            else:
                values = list(data.values())
                errors = None

            # Create bars
            if horizontal:
                bars = ax.barh(
                    range(len(values)),
                    values,
                    color=colors[: len(values)],
                    height=self.config.bar_width,
                )
            else:
                bars = ax.bar(
                    range(len(values)),
                    values,
                    color=colors[: len(values)],
                    width=self.config.bar_width,
                )

            # Add error bars if provided
            if errors:
                if self.config.error_bar_style == "bar":
                    ax.errorbar(
                        range(len(values)),
                        values,
                        yerr=errors,
                        fmt="none",
                        color="black",
                        capsize=5,
                    )
                else:  # band style
                    ax.fill_between(
                        range(len(values)),
                        [v - e for v, e in zip(values, errors)],
                        [v + e for v, e in zip(values, errors)],
                        alpha=0.2,
                        color="gray",
                    )

            # Customize appearance
            if self.config.grid:
                ax.grid(True, alpha=self.config.grid_alpha, linestyle="--")

            # Add value labels
            if self.config.show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=self.config.annotation_size,
                    )

            # Set labels and title
            ax.set_title(title, pad=20, fontsize=self.config.font_size + 2)
            ax.set_xlabel(xlabel, fontsize=self.config.font_size)
            ax.set_ylabel(ylabel, fontsize=self.config.font_size)

            # Set tick labels
            if horizontal:
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
            else:
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=self.config.rotation)

            # Adjust layout and save
            plt.tight_layout()

            # Enhanced saving with multiple formats
            if export_formats:
                self.exporter.save_plot(
                    fig, save_path, formats=export_formats, dpi=self.config.dpi
                )
            else:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

            self.logger.info(f"Successfully created bar graph: {save_path}")

            if show_plot:
                plt.show()

        except Exception as e:
            self.logger.error(f"Error creating bar graph: {e}")
            raise
        finally:
            plt.close()


# Example usage
if __name__ == "__main__":
    # Sample data
    data_simple = {"A": 25.3, "B": 42.1, "C": 31.7, "D": 55.4, "E": 38.9}

    # Data with error bars
    data_with_errors = {
        "A": (25.3, 2.1),
        "B": (42.1, 3.4),
        "C": (31.7, 2.8),
        "D": (55.4, 4.2),
        "E": (38.9, 3.1),
    }

    # Create plotter with default config
    plotter = BarGraphPlotter()

    # Simple bar plot
    plotter.plot_bars(
        data=data_simple,
        title="Simple Bar Graph",
        xlabel="Categories",
        ylabel="Values",
        save_path="simple_bars.png",
        color_scheme="default",
        palette_type="MULTI",
    )

    # Bar plot with error bars
    config_with_errors = BarGraphConfig(error_bar_style="band", show_values=True)
    plotter = BarGraphPlotter(config_with_errors)
    plotter.plot_bars(
        data=data_with_errors,
        title="Bar Graph with Error Bands",
        xlabel="Categories",
        ylabel="Values",
        save_path="bars_with_errors.png",
        color_scheme="science",
        palette_type="PUBLICATION",
    )

    # Colorblind-friendly plot
    config = BarGraphConfig(figure_width=10, figure_height=6)
    plotter = BarGraphPlotter(config)
    plotter.plot_bars(
        data=data_simple,
        title="Colorblind-Friendly Bar Graph",
        xlabel="Categories",
        ylabel="Values",
        save_path="colorblind_bars.png",
        color_scheme="wong",
        palette_type="COLORBLIND",
    )

    # Emphasis plot (highlighting specific bars)
    highlight_colors = BarGraphColorPalette.create_emphasis_palette(
        highlight_positions=[2, 4],  # Highlight 3rd and 5th bars
        n_bars=5,
        highlight_color="#FF0000",
        base_color="#E6E6E6",
    )
    plotter.plot_bars(
        data=data_simple,
        title="Emphasis Bar Graph",
        xlabel="Categories",
        ylabel="Values",
        save_path="emphasis_bars.png",
        color_scheme="default",  # This won't be used when custom_colors is provided
        palette_type="MULTI",  # This won't be used when custom_colors is provided
        custom_colors=highlight_colors,  # Pass the custom colors directly
    )

    # Example with new features
    config = BarGraphConfig(figure_width=10, figure_height=6)
    plotter = BarGraphPlotter(config)

    plotter.plot_bars(
        data=data_simple,
        title="Publication-Ready Bar Graph",
        xlabel="Categories",
        ylabel="Values",
        save_path="publication_bars",  # Extension will be added for each format
        color_scheme="nature",
        palette_type="PUBLICATION",
        publication_style=True,
        export_formats=["png", "pdf", "svg"],
    )
