import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from matplotlib.colors import LinearSegmentedColormap, Normalize


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
class HeatmapConfig(BaseConfig):
    """Configuration for heatmap styling extending BaseConfig"""

    annotation_size: int = 8
    cbar_width: float = 0.03
    cbar_pad: float = 0.04
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    center: Optional[float] = None
    robust: bool = False
    cmap_reverse: bool = False


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
    def validate_data(data: np.ndarray) -> bool:
        """Validate input data"""
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise ValueError("Invalid data array provided")
        if len(data.shape) != 2:
            raise ValueError("Data must be a 2D array")
        return True

    @staticmethod
    def validate_labels(labels: List[str], size: int) -> bool:
        """Validate label consistency"""
        if len(labels) != size:
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match data size ({size})"
            )
        return True


class PlotExporter:
    """Handle plot saving and format conversion"""

    @staticmethod
    def save_plot(
        fig: plt.Figure,
        path: Union[str, Path],
        formats: List[str] = ["png", "pdf", "svg", "eps"],
        dpi: int = 300,
    ) -> None:
        """
        Save plot in multiple formats including EPS

        Args:
            fig: matplotlib figure object
            path: base path for saving
            formats: list of formats to save in
            dpi: resolution for raster formats
        """
        base_path = str(path).rsplit(".", 1)[0]

        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            if fmt == "eps":
                # Special handling for EPS format
                fig.savefig(
                    save_path,
                    format="eps",
                    dpi=dpi,
                    bbox_inches="tight",
                    transparent=True,
                )
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


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


class HeatmapColorPalette:
    """Enhanced scientific heatmap color palette management"""

    # Sequential palettes for continuous data visualization
    SEQUENTIAL = {
        # Publication-quality sequential
        "viridis": ["#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"],
        "plasma": ["#0D0887", "#7E03A8", "#CC4778", "#F89540", "#F0F921"],
        "magma": ["#000004", "#51127C", "#B63679", "#FB8861", "#FCFDBF"],
        # Single-hue sequences
        "blues": [
            "#F7FBFF",
            "#DEEBF7",
            "#C6DBEF",
            "#9ECAE1",
            "#6BAED6",
            "#4292C6",
            "#2171B5",
            "#084594",
        ],
        "greens": [
            "#F7FCF5",
            "#E5F5E0",
            "#C7E9C0",
            "#A1D99B",
            "#74C476",
            "#41AB5D",
            "#238B45",
            "#005A32",
        ],
        "purples": [
            "#FCFBFD",
            "#EFEDF5",
            "#DADAEB",
            "#BCBDDC",
            "#9E9AC8",
            "#807DBA",
            "#6A51A3",
            "#4A1486",
        ],
        "oranges": [
            "#FFF5EB",
            "#FEE6CE",
            "#FDD0A2",
            "#FDAE6B",
            "#FD8D3C",
            "#F16913",
            "#D94801",
            "#8C2D04",
        ],
        "greys": [
            "#FFFFFF",
            "#F0F0F0",
            "#D9D9D9",
            "#BDBDBD",
            "#969696",
            "#737373",
            "#525252",
            "#252525",
        ],
        # Multi-hue sequences
        "YlOrRd": [
            "#FFFFCC",
            "#FFEDA0",
            "#FED976",
            "#FEB24C",
            "#FD8D3C",
            "#FC4E2A",
            "#E31A1C",
            "#B10026",
        ],
        "YlGnBu": [
            "#FFFFD9",
            "#EDF8B1",
            "#C7E9B4",
            "#7FCDBB",
            "#41B6C4",
            "#1D91C0",
            "#225EA8",
            "#0C2C84",
        ],
        "inferno": [
            "#000004",
            "#320A5A",
            "#781C6D",
            "#BC3754",
            "#ED6925",
            "#FBB41A",
            "#FCFFA4",
        ],
        "cividis": [
            "#00204C",
            "#213D6B",
            "#555B6C",
            "#7B7B78",
            "#A59C74",
            "#D6C271",
            "#FFE945",
        ],
    }

    # Diverging palettes for centered data
    DIVERGING = {
        # Classic diverging
        "rwb": ["#67001F", "#B2182B", "#F4A582", "#F5F5F5", "#92C5DE", "#2166AC"],
        "coolwarm": ["#4A148C", "#7B1FA2", "#E1BEE7", "#F5F5F5", "#B3E5FC", "#01579B"],
        "spectral": [
            "#9E0142",
            "#D53E4F",
            "#F46D43",
            "#FFFFBF",
            "#66C2A5",
            "#3288BD",
            "#5E4FA2",
        ],
        # Scientific diverging
        "PiYG": [
            "#8E0152",
            "#C51B7D",
            "#DE77AE",
            "#F1B6DA",
            "#FDE0EF",
            "#E6F5D0",
            "#B8E186",
            "#7FBC41",
            "#4D9221",
            "#276419",
        ],
        "BrBG": [
            "#543005",
            "#8C510A",
            "#BF812D",
            "#DFC27D",
            "#F6E8C3",
            "#F5F5F5",
            "#C7EAE5",
            "#80CDC1",
            "#35978F",
            "#01665E",
            "#003C30",
        ],
        "PRGn": [
            "#40004B",
            "#762A83",
            "#9970AB",
            "#C2A5CF",
            "#E7D4E8",
            "#F7F7F7",
            "#D9F0D3",
            "#A6DBA0",
            "#5AAE61",
            "#1B7837",
            "#00441B",
        ],
        "RdYlBu": [
            "#A50026",
            "#D73027",
            "#F46D43",
            "#FDAE61",
            "#FEE090",
            "#FFFFBF",
            "#E0F3F8",
            "#ABD9E9",
            "#74ADD1",
            "#4575B4",
            "#313695",
        ],
        "RdGy": [
            "#67001F",
            "#B2182B",
            "#D6604D",
            "#F4A582",
            "#FDDBC7",
            "#FFFFFF",
            "#E0E0E0",
            "#BABABA",
            "#878787",
            "#4D4D4D",
            "#1A1A1A",
        ],
    }

    # Publication-specific color schemes
    PUBLICATION = {
        # Journal standards
        "nature": ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"],
        "science": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021"],
        "cell": ["#2271B2", "#D55E00", "#359B73", "#E69F00", "#CC79A7", "#0072B2"],
        "nejm": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD"],
        "lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91"],
        # Colorblind-friendly
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

    # Special purpose palettes
    SPECIAL = {
        # Correlation matrices
        "correlation": [
            "#001F3F",
            "#0074D9",
            "#7FDBFF",
            "#FFFFFF",
            "#FF851B",
            "#FF4136",
            "#85144B",
        ],
        "correlation_simple": ["#3B9AB2", "#FFFFFF", "#E31A1C"],
        # Expression data
        "expression": [
            "#00441B",
            "#1B7837",
            "#5AAE61",
            "#FFFFFF",
            "#DFC27D",
            "#BF812D",
            "#8C510A",
        ],
        "expression_binary": ["#2166AC", "#FFFFFF", "#B2182B"],
        # P-value visualization
        "pvalue": ["#FFFFFF", "#FEE0D2", "#FC9272", "#DE2D26"],
        "significance": [
            "#FFFFFF",
            "#FFF5F0",
            "#FEE0D2",
            "#FCC5C0",
            "#FA9FB5",
            "#F768A1",
            "#DD3497",
            "#AE017E",
            "#7A0177",
        ],
    }

    @staticmethod
    def create_custom_cmap(
        colors: List[str], name: str = "custom", reverse: bool = False
    ) -> LinearSegmentedColormap:
        """Create custom colormap"""
        if reverse:
            colors = colors[::-1]
        return LinearSegmentedColormap.from_list(name, colors)

    @staticmethod
    def get_scale_normalizer(
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None,
        robust: bool = False,
    ) -> Normalize:
        """Create color scale normalizer"""
        if robust:
            q_low, q_high = np.percentile(data, [2, 98])
            vmin = vmin if vmin is not None else q_low
            vmax = vmax if vmax is not None else q_high

        if center is not None:
            return plt.cm.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        return plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)

    # Add new methods for custom palette creation
    @staticmethod
    def create_sequential_gradient(
        start_color: str, end_color: str, n_colors: int = 256
    ) -> List[str]:
        """Create custom sequential gradient between two colors"""
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

    @staticmethod
    def create_diverging_gradient(
        left_color: str, center_color: str, right_color: str, n_colors: int = 256
    ) -> List[str]:
        """Create custom diverging gradient with three colors"""
        n_half = n_colors // 2

        # Create two sequential gradients
        left_gradient = HeatmapColorPalette.create_sequential_gradient(
            left_color, center_color, n_half
        )
        right_gradient = HeatmapColorPalette.create_sequential_gradient(
            center_color, right_color, n_half
        )

        # Combine gradients
        return left_gradient + right_gradient[1:]


class HeatmapPlotter:
    """Enhanced scientific heatmap plotter with dynamic styling"""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        self.config = config or HeatmapConfig()
        self.logger = PlotLogger().logger
        self.validator = PlotValidator()
        self.exporter = PlotExporter()

    def plot_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str,
        save_path: Union[str, Path],
        palette_type: str = "SEQUENTIAL",
        scheme: str = "viridis",
        show_annotations: bool = True,
        show_plot: bool = True,
        center_value: Optional[float] = None,
        publication_style: bool = False,
        export_formats: Optional[List[str]] = None,
    ) -> None:
        """Create enhanced publication-quality heatmap"""
        try:
            # Validate inputs
            self.validator.validate_data(data)
            self.validator.validate_labels(row_labels, data.shape[0])
            self.validator.validate_labels(col_labels, data.shape[1])

            # Setup figure
            aspect_ratio = data.shape[1] / data.shape[0]
            fig_width = self.config.figure_width
            fig_height = fig_width / aspect_ratio
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Apply publication style if requested
            if publication_style:
                StyleManager.set_publication_style(ax, fontsize=self.config.font_size)

            # Color palette setup
            palette = getattr(HeatmapColorPalette, palette_type.upper())[scheme]
            cmap = HeatmapColorPalette.create_custom_cmap(
                palette, scheme, reverse=self.config.cmap_reverse
            )

            # Color scaling
            norm = HeatmapColorPalette.get_scale_normalizer(
                data,
                vmin=self.config.vmin,
                vmax=self.config.vmax,
                center=center_value or self.config.center,
                robust=self.config.robust,
            )

            # Create heatmap
            sns.heatmap(
                data,
                xticklabels=col_labels,
                yticklabels=row_labels,
                cmap=cmap,
                norm=norm,
                annot=show_annotations,
                fmt=".2f",
                annot_kws={"size": self.config.annotation_size},
                cbar_kws={"shrink": 0.5, "label": "Value", "pad": self.config.cbar_pad},
                ax=ax,
            )

            # Styling
            ax.set_title(title, pad=20, fontsize=self.config.font_size + 2)
            plt.xticks(rotation=self.config.rotation, ha="right")
            plt.yticks(rotation=0)

            # Colorbar customization
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=self.config.font_size)

            # Layout and save
            plt.tight_layout()

            # Enhanced saving with multiple formats
            if export_formats:
                self.exporter.save_plot(
                    fig, save_path, formats=export_formats, dpi=self.config.dpi
                )
            else:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")

            self.logger.info(f"Successfully created heatmap: {save_path}")

            if show_plot:
                plt.show()

        except Exception as e:
            self.logger.error(f"Error creating heatmap: {e}")
            raise
        finally:
            plt.close()


# Example usage:
if __name__ == "__main__":
    # Generate different types of sample data
    np.random.seed(42)

    # 1. Correlation matrix data
    base_data = np.random.randn(10, 8)  # 10x8 matrix
    correlation_data = np.corrcoef(base_data.T)  # 8x8 correlation matrix
    corr_labels = [f"Feature_{i+1}" for i in range(correlation_data.shape[0])]

    # 2. Expression-like data
    expression_data = np.random.uniform(-2, 2, (10, 8))  # 10x8 matrix
    row_labels = [f"Gene_{i+1}" for i in range(expression_data.shape[0])]
    col_labels = [f"Sample_{i+1}" for i in range(expression_data.shape[1])]

    # 3. P-value matrix
    pvalue_data = np.abs(np.random.uniform(0, 1, (8, 8)))  # 8x8 matrix
    pvalue_labels = [f"Test_{i+1}" for i in range(pvalue_data.shape[0])]

    # Initialize plotter with configuration
    config = HeatmapConfig(
        figure_width=12, figure_height=8, font_size=10, annotation_size=8, robust=True
    )
    plotter = HeatmapPlotter(config)

    # 1. Correlation matrix plot
    plotter.plot_heatmap(
        data=correlation_data,
        row_labels=corr_labels,
        col_labels=corr_labels,
        title="Correlation Matrix",
        save_path="correlation_heatmap",
        palette_type="SPECIAL",
        scheme="correlation",
        center_value=0,
        publication_style=True,
        export_formats=["png", "pdf", "svg", "eps"],
    )

    # 2. Expression data plot
    plotter.plot_heatmap(
        data=expression_data,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Expression Heatmap",
        save_path="expression_heatmap",
        palette_type="SPECIAL",
        scheme="expression",
        center_value=0,
        publication_style=True,
        export_formats=["png", "pdf"],
    )

    # 3. P-value matrix plot
    plotter.plot_heatmap(
        data=pvalue_data,
        row_labels=pvalue_labels,
        col_labels=pvalue_labels,
        title="P-value Matrix",
        save_path="pvalue_heatmap",
        palette_type="SPECIAL",
        scheme="pvalue",
        show_annotations=True,
        publication_style=True,
        export_formats=["png", "pdf"],
    )

    # 4. Custom diverging gradient example
    custom_colors = HeatmapColorPalette.create_diverging_gradient(
        left_color="#0000FF",  # Blue
        center_color="#FFFFFF",  # White
        right_color="#FF0000",  # Red
    )

    plotter.plot_heatmap(
        data=base_data,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Custom Diverging Heatmap",
        save_path="custom_heatmap",
        palette_type="DIVERGING",
        scheme="coolwarm",
        center_value=0,
        publication_style=True,
        export_formats=["png", "pdf"],
    )
