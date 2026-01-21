import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


class HeatmapColorPalette:
    """Scientific heatmap color palette management"""

    # Validated sequential palettes (low to high values)
    SEQUENTIAL = {
        "viridis": ["#440154", "#414487", "#2A788E", "#22A884", "#7AD151", "#FDE725"],
        "plasma": ["#0D0887", "#7E03A8", "#CC4778", "#F89540", "#F0F921"],
        "magma": ["#000004", "#51127C", "#B63679", "#FB8861", "#FCFDBF"],
    }

    # Validated diverging palettes (centered values)
    DIVERGING = {
        "rwb": [
            "#67001F",
            "#B2182B",
            "#F4A582",
            "#FDDBC7",
            "#D1E5F0",
            "#92C5DE",
            "#2166AC",
        ],
        "coolwarm": [
            "#4A148C",
            "#7B1FA2",
            "#E1BEE7",
            "#F5F5F5",
            "#B3E5FC",
            "#0288D1",
            "#01579B",
        ],
        "spectral": [
            "#9E0142",
            "#D53E4F",
            "#F46D43",
            "#FFFFBF",
            "#66C2A5",
            "#3288BD",
            "#5E4FA2",
        ],
    }

    # Validated continuous palettes (correlation matrices)
    CORRELATION = {
        "correlation": [
            "#001F3F",
            "#0074D9",
            "#7FDBFF",
            "#FFFFFF",
            "#FF851B",
            "#FF4136",
            "#85144B",
        ],
        "pearson": [
            "#052F60",
            "#0C5DC4",
            "#89C4FF",
            "#FFFFFF",
            "#FF9B9B",
            "#D92121",
            "#7A0000",
        ],
    }

    @staticmethod
    def get_heatmap_colors(
        n_colors: int, palette_type: str = "sequential", scheme: str = None
    ) -> List[str]:
        """
        Get scientifically validated heatmap colors

        Args:
            n_colors: Number of colors needed
            palette_type: 'sequential', 'diverging', or 'correlation'
            scheme: Specific color scheme name

        Returns:
            List of hex colors optimized for heatmaps
        """
        # Select palette based on type
        if palette_type == "sequential":
            base_colors = HeatmapColorPalette.SEQUENTIAL.get(scheme or "viridis")
        elif palette_type == "diverging":
            base_colors = HeatmapColorPalette.DIVERGING.get(scheme or "rwb")
        else:
            base_colors = HeatmapColorPalette.CORRELATION.get(scheme or "correlation")

        # Interpolate colors
        return HeatmapColorPalette._interpolate_colors(base_colors, n_colors)

    @staticmethod
    def _interpolate_colors(base_colors: List[str], n_colors: int) -> List[str]:
        """Interpolate between base colors to get smooth gradient"""
        try:
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt

            # Convert hex to RGB
            rgb_colors = [mcolors.hex2color(color) for color in base_colors]

            # Create color map
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                "custom", rgb_colors, N=n_colors
            )

            # Generate colors
            colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]

            # Convert back to hex
            return [mcolors.rgb2hex(color[:3]) for color in colors]

        except Exception as e:
            print(f"Error interpolating colors: {e}")
            return base_colors[:n_colors]

    @staticmethod
    def validate_colorblind(colors: List[str]) -> bool:
        """Validate colors for colorblind accessibility"""
        try:
            min_delta_e = 20  # Minimum color difference threshold

            for i, color1 in enumerate(colors[:-1]):
                rgb1 = sRGBColor.new_from_rgb_hex(color1)
                lab1 = convert_color(rgb1, LabColor)

                # Compare with next color
                rgb2 = sRGBColor.new_from_rgb_hex(colors[i + 1])
                lab2 = convert_color(rgb2, LabColor)

                if delta_e_cie2000(lab1, lab2) < min_delta_e:
                    return False

            return True

        except Exception as e:
            print(f"Error validating colors: {e}")
            return False


# Usage example:
if __name__ == "__main__":
    # Get sequential colors for standard heatmap
    seq_colors = HeatmapColorPalette.get_heatmap_colors(
        n_colors=256, palette_type="sequential", scheme="viridis"
    )

    # Get diverging colors for correlation matrix
    div_colors = HeatmapColorPalette.get_heatmap_colors(
        n_colors=256, palette_type="diverging", scheme="coolwarm"
    )
