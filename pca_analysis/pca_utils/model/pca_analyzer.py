from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import combinations_with_replacement
import os

from .mi_calculator import mi_betwn_uandy
from ..visualization.pca_plotter import plot_mod, scatter_plot_gen

class PCAanalyzer:

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        method: str = "cov",
        n_components: Optional[int] = None,
        bin_edges: Sequence[float] = None,
    ) -> None:
        """Initializing the PCAanalyzer"""
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self.method = method.lower()
        self.n_components = n_components or min(self.data.shape)
        self.bin_edges = bin_edges
        self.data_scaler = StandardScaler()
        self.scaled_data = self.data_scaler.fit_transform(self.data)
        self._calculate_distance()
        self._eigen_decompostion()

        # validating the input methods
        if self.method not in ["cov", "mi"]:
            raise ValueError("Method should be either 'cov' or 'mi'")

    # Calculate the distance matrix based on the given method
    def _calculate_distance(self):
        """Calculate the distance matrix based on the given methods"""
        if self.method == "mi":
            self.dist = self._mutual_information()
        else:
            self.dist = np.cov(self.scaled_data, rowvar=False)

    # Mutual information calculation
    def _mutual_information(self):
        """Calculate the mutual information"""

        col_num = self.data.shape[1]

        # for calculating the upper triangular mi, because of its symmetry.
        all_col_indx_comb = list(combinations_with_replacement(range(col_num), r=2))

        # implementing the parallel computation for mutual information calculation
        mi_result = Parallel(n_jobs=-1)(
            delayed(mi_betwn_uandy)(self.data[:, i], self.data[:, j], self.bin_edges)
            for i, j in all_col_indx_comb
        )

        mi_mat = np.zeros((col_num, col_num))
        for indx, mi in zip(all_col_indx_comb, mi_result):
            mi_mat[indx[0], indx[1]] = mi_mat[indx[1], indx[0]] = mi

        return mi_mat

    # Perform Eigen decomposition and calculate eigen values and eigen vectors
    def _eigen_decompostion(self) -> None:
        """Perform Eigen decomposition and calculate:
        * Eigen values and Eigen vectors
        * Explained variance ratio & cumulative variance ratio
        """
        self.eigen_values, self.eigen_vectors = np.linalg.eigh(self.dist)

        # sorting the eigen_values
        sorted_indices = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[sorted_indices]
        self.eigen_vectors = self.eigen_vectors[sorted_indices]

        # explained variance ratio
        self.explained_variance_ratio = (
            self.eigen_values / np.sum(self.eigen_values)
        ) * 100

        # cumulative variance ratio
        self.cum_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)

    # Transform data using selected principal components
    def fit_transform(self):
        """Transform data using selected principal component"""
        return self.scaled_data @ self.eigen_vectors[:, : self.n_components]

    # Plot transformed data with respect to principal components
    def scatter_plot(
        self, components: Tuple[int, int] = (0, 1), fig_sav=None, out_dir=None
    ):
        """Plot transformed data with respect to principal component"""
        x_value, y_value = self.fit_transform()[:, components].T
        x_label, y_label = (
            f"PCA - {components[0]}",
            f"PCA - {components[1]}",
        )

        if fig_sav:
            file_sav = True
            file_name = (
                f"{self.method}_pca_components{components[0]}_{components[1]}.jpeg"
            )
            out_dir = out_dir

        else:
            file_sav = False
            file_name = None
            out_dir = None

        scatter_plot_gen(
            x_value=x_value,
            y_value=y_value,
            x_label=x_label,
            y_label=y_label,
            file_sav=file_sav,
            file_name=file_name,
            out_dir=out_dir,
        )

    # Cumulative plot that explains variance captured by the eigen values
    def cumsum_plot(
        self,
        fig_sav: bool = None,
        out_dir: str = None,
        indvid_label: str = None,
        cumulative_label: str = None,
    ):
        """Cumulative plot that explains variance captured by the eigen values"""
        fig, ax = plt.subplots(figsize=(9, 7))
        # plot to transparent
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")
        fig.patch.set_edgecolor("none")
        ax.bar(
            range(0 + 1, len(self.eigen_values) + 1),
            self.explained_variance_ratio,
            alpha=0.8,
            # color="#4f5d5d",
            color="#225283",
            label=indvid_label.lower(),
        )
        ax.plot(
            range(0 + 1, len(self.eigen_values) + 1),
            self.cum_explained_variance_ratio,
            marker="8",
            markersize=5,
            linestyle=":",
            color="#8b4d1f",
            label=f"cumulative {cumulative_label.lower()}",
        )

        legend = plt.legend()
        legend.get_frame().set_alpha(0.0)  # Make background transparent
        legend.get_frame().set_facecolor('none')  # Remove face color
        legend.get_frame().set_edgecolor('none')
        #  Modify font properties after creating the legend
        for text in legend.get_texts():
            text.set_fontsize(14)  # Set font size
            text.set_fontweight("bold")  # Set font weight
            text.set_fontname("Arial")

        plot_mod(
            ax=ax,
            x_label="Principal components",
            y_label=f"{indvid_label.capitalize()}(%)",
        )

        if fig_sav:
            if out_dir != None:
                out_dir = os.path.join(out_dir, "cumsum")
            else:
                out_dir = os.path.join(os.getcwd(), "cumsum")

            os.makedirs(out_dir, exist_ok=True)
            file_name = f"{self.method}_cumulative_{indvid_label}_plot.jpeg"
            file_name_png = f"{self.method}_cumulative_{indvid_label}_plot.png"
            file_name_eps = f"{self.method}_cumulative_{indvid_label}_plot.eps"
            
            # saving Png as transparent
            
            plt.savefig(os.path.join(out_dir, file_name), dpi=400, bbox_inches="tight", transparent=True)
            plt.savefig(os.path.join(out_dir, file_name_eps), dpi=700, bbox_inches="tight", transparent=True)
            if file_name_png:
                # fig.patch.set_alpha(0.0)
                # fig.patch.set_facecolor("none")
                # fig.patch.set_edgecolor("none")
                fig.savefig(
                    os.path.join(out_dir, file_name_png), dpi=400, bbox_inches="tight", transparent=True
                )

            plt.close()

        else:
            plt.show()
