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
class HistogramConfig(BaseConfig):
    """Configuration for histogram styling extending BaseConfig"""
    n_bins: int = 50
    density: bool = False
    cumulative: bool = False
    kde: bool = False
    kde_bandwidth: float = 0.5
    alpha: float = 0.75
    edge_color: str = 'black'
    edge_width: float = 0.8
    show_mean: bool = True
    show_median: bool = True
    mean_line_style: str = '--'
    median_line_style: str = ':'

class StyleManager:
    """Manage plot styles across different types"""
    
    @staticmethod
    def set_publication_style(ax: plt.Axes,
                            journal: str = 'nature',
                            fontsize: int = 10) -> None:
        """Apply journal-specific styling"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=fontsize)

class PlotValidator:
    """Validate plot inputs and configurations"""
    
    @staticmethod
    def validate_data(data: Union[List[float], np.ndarray]) -> bool:
        """Validate input data"""
        if not isinstance(data, (list, np.ndarray)) or len(data) == 0:
            raise ValueError("Invalid data provided")
        return True
    
    @staticmethod
    def validate_bins(bins: Optional[Union[int, List[float]]], data_size: int) -> bool:
        """Validate bin specification"""
        if isinstance(bins, int) and bins <= 0:
            raise ValueError("Number of bins must be positive")
        if isinstance(bins, list) and len(bins) < 2:
            raise ValueError("Bin edges must have at least 2 values")
        return True

class PlotExporter:
    """Handle plot saving and format conversion"""
    
    @staticmethod
    def save_plot(fig: plt.Figure,
                  path: Union[str, Path],
                  formats: List[str] = ['png', 'pdf', 'svg', 'eps'],
                  dpi: int = 300) -> None:
        """Save plot in multiple formats"""
        base_path = str(path).rsplit('.', 1)[0]
        for fmt in formats:
            save_path = f"{base_path}.{fmt}"
            if fmt == 'eps':
                fig.savefig(save_path, format='eps', 
                          dpi=dpi, bbox_inches='tight',
                          transparent=True)
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

class PlotLogger:
    """Logging for plot creation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

class HistogramColorPalette:
    """Enhanced color palette management for histograms"""
    
    # Single color schemes
    SOLID = {
        'blue': '#4e79a7',
        'red': '#e15759',
        'green': '#59a14f',
        'purple': '#b07aa1',
        'orange': '#ff9d9a'
    }
    
    # Journal color schemes
    PUBLICATION = {
        'nature': '#4477AA',
        'science': '#3B4992',
        'cell': '#2271B2',
        'nejm': '#BC3C29',
        'lancet': '#00468B'
    }
    
    # Statistical color schemes
    STATISTICAL = {
        'normal': '#3182bd',
        'poisson': '#31a354',
        'binomial': '#756bb1',
        'uniform': '#636363',
        'exponential': '#e6550d'
    }
    
    # Emphasis color schemes
    EMPHASIS = {
        'highlight': '#ff0000',
        'background': '#cccccc',
        'confidence': '#9ecae1'
    }

class HistogramPlotter:
    """Enhanced histogram plotter with dynamic styling"""
    
    def __init__(self, config: Optional[HistogramConfig] = None):
        self.config = config or HistogramConfig()
        self.logger = PlotLogger().logger
        self.validator = PlotValidator()
        self.exporter = PlotExporter()
    
    def plot_histogram(self,
                      data: Union[List[float], np.ndarray],
                      title: str,
                      xlabel: str,
                      ylabel: str,
                      save_path: Union[str, Path],
                      color_scheme: str = 'normal',
                      palette_type: str = 'STATISTICAL',
                      show_plot: bool = True,
                      custom_color: Optional[str] = None,
                      bins: Optional[Union[int, List[float]]] = None,
                      publication_style: bool = False,
                      export_formats: Optional[List[str]] = None) -> None:
        """Create enhanced publication-quality histogram"""
        try:
            # Validate inputs
            self.validator.validate_data(data)
            if bins:
                self.validator.validate_bins(bins, len(data))
            
            # Setup figure
            fig, ax = plt.subplots(figsize=(self.config.figure_width, 
                                           self.config.figure_height))
            
            # Get color
            color = (custom_color if custom_color 
                    else getattr(HistogramColorPalette, palette_type)[color_scheme])
            
            # Create histogram
            n, bins, patches = ax.hist(
                data,
                bins=bins or self.config.n_bins,
                density=self.config.density,
                cumulative=self.config.cumulative,
                color=color,
                alpha=self.config.alpha,
                edgecolor=self.config.edge_color,
                linewidth=self.config.edge_width
            )
            
            # Add KDE if requested
            if self.config.kde:
                kde_x = np.linspace(min(data), max(data), 200)
                kde = sns.kdeplot(
                    data=data,
                    bw_adjust=self.config.kde_bandwidth,
                    color='black',
                    alpha=0.5,
                    ax=ax
                )
            
            # Add mean and median lines
            if self.config.show_mean:
                mean = np.mean(data)
                ax.axvline(mean, color='red', 
                          linestyle=self.config.mean_line_style,
                          label=f'Mean: {mean:.2f}')
            
            if self.config.show_median:
                median = np.median(data)
                ax.axvline(median, color='green',
                          linestyle=self.config.median_line_style,
                          label=f'Median: {median:.2f}')
            
            # Customize appearance
            if self.config.grid:
                ax.grid(True, alpha=self.config.grid_alpha, linestyle='--')
            
            # Set labels and title
            ax.set_title(title, pad=20, fontsize=self.config.font_size + 2)
            ax.set_xlabel(xlabel, fontsize=self.config.font_size)
            ax.set_ylabel(ylabel, fontsize=self.config.font_size)
            
            if self.config.show_mean or self.config.show_median:
                ax.legend(fontsize=self.config.font_size)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Enhanced saving with multiple formats
            if export_formats:
                self.exporter.save_plot(fig, save_path, formats=export_formats, dpi=self.config.dpi)
            else:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            
            self.logger.info(f"Successfully created histogram: {save_path}")
            
            if show_plot:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error creating histogram: {e}")
            raise
        finally:
            plt.close()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(loc=0, scale=1, size=1000)
    
    # Publication-ready histogram
    config = HistogramConfig(
        n_bins=30,
        kde=True,
        show_mean=True,
        show_median=True
    )
    plotter = HistogramPlotter(config)
    
    # Create histogram with multiple export formats
    plotter.plot_histogram(
        data=normal_data,
        title='Publication-Ready Histogram',
        xlabel='Value',
        ylabel='Frequency',
        save_path='publication_histogram',
        color_scheme='nature',
        palette_type='PUBLICATION',
        publication_style=True,
        export_formats=['png', 'pdf', 'svg', 'eps']
    )