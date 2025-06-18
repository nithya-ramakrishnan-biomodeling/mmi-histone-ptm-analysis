from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import warnings
from typing import Literal, List, Union, Optional, Tuple, Dict, Any

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Define consistent plot styling
PLOT_STYLE = {
    "fontsize": 12, 
    "family": "serif", 
    "weight": "bold"
}

class ManualOMPHistoneRegressor:
    """
    Regressor model for histone modification prediction with manual OMP feature selection.
    
    This class implements a two-stage approach that:
    1. Uses OMP directly to select features based on coefficient importance
    2. Builds an XGBoost model using only the selected features
    
    The implementation provides more direct control over feature selection than pipeline-based approaches.
    """
    
    def __init__(
        self, 
        features: np.ndarray, 
        target: np.ndarray, 
        feature_names: Optional[List[str]] = None,
        use_cross_validation: bool = True,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 10,
        cv_folds: int = 10,
        n_features_to_select: Optional[int] = None,
        min_coef_value: float = 1e-8,
        use_cv_omp: bool = True
    ) -> None:
        """
        Initialize the ManualOMPHistoneRegressor model.
        
        Args:
            features: Input feature matrix
            target: Target values to predict
            feature_names: Names of features (optional)
            use_cross_validation: Whether to use k-fold cross-validation for evaluation
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in XGBoost ensemble
            max_depth: Maximum depth of trees in XGBoost
            cv_folds: Number of cross-validation folds
            n_features_to_select: Number of features to select (if None, uses coefficient threshold)
            min_coef_value: Minimum absolute coefficient value for feature selection
            use_cv_omp: Whether to use cross-validated OMP instead of standard OMP
        """
        self.features = features
        self.target = target
        self.use_cross_validation = use_cross_validation
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.n_features_to_select = n_features_to_select
        self.min_coef_value = min_coef_value
        self.use_cv_omp = use_cv_omp
        
        # Initialize state variables
        self.selected_features = None
        self.selected_feature_names = None
        self.omp_coefficients = None
        self.is_trained = False
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # Generate default feature names if not provided
        if feature_names is None and isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
        elif feature_names is None:
            n_features = features.shape[1] if np.ndim(features) > 1 else 1
            self.feature_names = [f"feature_{i+1}" for i in range(n_features)]
        else:
            self.feature_names = feature_names

    def _ensure_2d_features(self, features: np.ndarray) -> np.ndarray:
        """
        Ensure features are in 2D format required by scikit-learn.
        
        Args:
            features: Input feature array
            
        Returns:
            Reshaped features array (if needed)
        """
        if np.ndim(features) < 2:
            return features.reshape(-1, 1)
        return features

    def _select_features_with_omp(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Perform feature selection using OMP.
        
        Args:
            x_train: Training features (already scaled)
            y_train: Training targets
            
        Returns:
            Tuple of (selected feature matrix, selected feature indices)
        """
        # Choose between standard OMP and CV-OMP based on configuration
        if self.use_cv_omp:
            omp = OrthogonalMatchingPursuitCV(cv=self.cv_folds)
        elif self.n_features_to_select is not None:
            # Use standard OMP with specified number of features
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_features_to_select)
        else:
            # Use standard OMP with max_iter set to number of features
            omp = OrthogonalMatchingPursuit(tol=self.min_coef_value)
        
        # Fit OMP model to get coefficients
        omp.fit(x_train, y_train)
        
        # Extract non-zero coefficients
        if self.n_features_to_select is not None:
            # Get top N features by coefficient magnitude
            coef_magnitudes = np.abs(omp.coef_)
            selected_indices = np.argsort(coef_magnitudes)[::-1][:self.n_features_to_select]
        else:
            # Select features above threshold
            selected_indices = np.where(np.abs(omp.coef_) > self.min_coef_value)[0]
        
        # Store coefficients for later use
        self.omp_coefficients = omp.coef_[selected_indices]
        
        # Store feature selection info
        self.selected_features = selected_indices
        if len(selected_indices) > 0 and len(self.feature_names) >= max(selected_indices) + 1:
            self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
        else:
            self.selected_feature_names = [f"feature_{i+1}" for i in selected_indices]
        
        # Return selected feature subset
        if len(selected_indices) > 0:
            return x_train[:, selected_indices], selected_indices
        else:
            # If no features were selected, use all features
            print("Warning: No features were selected. Using all features.")
            return x_train, np.arange(x_train.shape[1])

    def train_and_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train model and predict using train-test split approach.
        
        Returns:
            Tuple of (true values, predicted values)
        """
        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.features, 
            self.target, 
            shuffle=True, 
            random_state=self.random_state, 
            test_size=0.33
        )

        # Ensure features are in correct shape
        x_train = self._ensure_2d_features(x_train)
        x_test = self._ensure_2d_features(x_test)
        
        # Scale features
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        # Select features using OMP
        x_train_selected, selected_indices = self._select_features_with_omp(x_train_scaled, y_train)
        
        # Use only selected features for test set
        x_test_selected = x_test_scaled[:, selected_indices] if len(selected_indices) > 0 else x_test_scaled
        
        # Create and train XGBoost model on selected features
        self.xgb_model = xgb.XGBRegressor(
            n_jobs=-1,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth
        )
        
        # Train XGBoost model
        self.xgb_model.fit(x_train_selected, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.xgb_model.predict(x_test_selected)
        return y_test, y_pred
        
    def calculate_error(self, method: Literal["r2", "mse", "rmse"] = "r2") -> float:
        """
        Calculate error metrics with or without cross-validation.
        
        Args:
            method: Error metric to calculate ("r2", "mse", or "rmse")
            
        Returns:
            Calculated error value
        """
        if self.use_cross_validation:
            return self._calculate_error_with_cv(method)
        else:
            return self._calculate_error_standard(method)

    def _calculate_error_standard(self, method: Literal["r2", "mse", "rmse"]) -> float:
        """
        Calculate error metrics without cross-validation.
        
        Args:
            method: Error metric to calculate
            
        Returns:
            Calculated error value
        """
        y_test, y_pred = self.train_and_predict()
        
        if method == "mse":
            return np.round(mean_squared_error(y_test, y_pred), 3)
        elif method == "rmse":
            return np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        else:  # r2
            return np.round(r2_score(y_test, y_pred), 3)

    def _calculate_error_with_cv(self, method: Literal["r2", "mse", "rmse"]) -> float:
        """
        Calculate error metrics with cross-validation.
        
        Args:
            method: Error metric to calculate
            
        Returns:
            Calculated error value (mean across folds)
        """
        features = self._ensure_2d_features(self.features)
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        for train_idx, test_idx in kfold.split(features):
            # Split data
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = self.target[train_idx], self.target[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Select features using OMP
            X_train_selected, selected_indices = self._select_features_with_omp(X_train_scaled, y_train)
            
            # Use only selected features for test set
            X_test_selected = X_test_scaled[:, selected_indices] if len(selected_indices) > 0 else X_test_scaled
            
            # Train and predict with XGBoost
            xgb_model = xgb.XGBRegressor(
                n_jobs=-1,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_depth=self.max_depth
            )
            xgb_model.fit(X_train_selected, y_train)
            y_pred = xgb_model.predict(X_test_selected)
            
            # Calculate score based on method
            if method == "mse":
                score = mean_squared_error(y_test, y_pred)
            elif method == "rmse":
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            else:  # r2
                score = r2_score(y_test, y_pred)
                
            cv_scores.append(score)
        
        # When done with CV, train a final model on all data for feature extraction
        if not self.is_trained:
            self.train_and_predict()
            
        return np.round(np.mean(cv_scores), 3)

    def plot_predictions(
        self, 
        filename: Optional[str] = None, 
        title_info: Optional[Tuple[str, str]] = None, 
        output_dir: Optional[str] = None
    ) -> None:
        """
        Plot predicted vs. true values and save/display the figure.
        
        Args:
            filename: Name to save the plot (without extension)
            title_info: Tuple of (feature_name, target_name) for plot title
            output_dir: Directory to save the plot in
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_facecolor("#EBEBEB")
        fig.set_facecolor("#F2F2F2")

        # Get predictions and plot
        y_test, y_pred = self.train_and_predict()
        ax.scatter(y_test, y_pred, c="#3F7C85", alpha=0.7)
        
        # Add diagonal reference line
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="#A60A33")

        # Set title with R² score
        r2_value = f"{self.calculate_error('r2'):.2f}"
        if title_info:
            ax.set_title(
                f"Feature: {title_info[0]}, Target: {title_info[1]}, R²: {r2_value}",
                fontdict=PLOT_STYLE
            )
        else:
            ax.set_title(f"R²: {r2_value}", fontdict=PLOT_STYLE)
        
        # Set labels and style
        ax.set_xlabel("True Values", fontdict=PLOT_STYLE)
        ax.set_ylabel("Predicted Values", fontdict=PLOT_STYLE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Save or display plot
        if filename:
            # Build output path
            base_output_dir = os.path.abspath(os.path.join("..", "..", "output", "images"))
            if output_dir:
                full_output_dir = os.path.join(base_output_dir, output_dir)
                os.makedirs(full_output_dir, exist_ok=True)
                output_path = os.path.join(full_output_dir, f"{filename}.jpeg")
            else:
                os.makedirs(base_output_dir, exist_ok=True)
                output_path = os.path.join(base_output_dir, f"{filename}.jpeg")
            
            # Save and close
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_feature_importance(
        self,
        plot_type: Literal["omp", "xgboost", "both"] = "both",
        filename: Optional[str] = None,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Plot feature importance from OMP coefficients and/or XGBoost.
        
        Args:
            plot_type: Type of plot to generate ('omp', 'xgboost', or 'both')
            filename: Name to save the plot (without extension)
            output_dir: Directory to save the plot in
            figsize: Figure size as (width, height)
        """
        # Ensure model is trained
        if not self.is_trained:
            self.train_and_predict()
            
        if not self.selected_features.size:
            print("No features were selected by OMP. Cannot plot feature importance.")
            return
            
        # Set up figure based on plot type
        if plot_type == "both":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            fig.tight_layout(pad=4.0)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot OMP coefficients
        if plot_type in ["omp", "both"]:
            ax_omp = ax1 if plot_type == "both" else ax
            
            # Sort features by coefficient magnitude for better visualization
            sorted_indices = np.argsort(np.abs(self.omp_coefficients))[::-1]
            sorted_names = [self.selected_feature_names[i] for i in sorted_indices]
            sorted_coefs = self.omp_coefficients[sorted_indices]
            
            # Create colormap based on coefficient sign
            colors = ['#3F7C85' if c >= 0 else '#A60A33' for c in sorted_coefs]
            
            # Plot bars
            bars = ax_omp.bar(range(len(sorted_names)), np.abs(sorted_coefs), color=colors)
            
            # Add coefficient values as text
            for i, bar in enumerate(bars):
                height = bar.get_height()
                value = sorted_coefs[i]
                ax_omp.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.02 * max(np.abs(sorted_coefs)),
                    f'{value:.4f}',
                    ha='center', va='bottom', rotation=0, fontsize=8
                )
            
            # Set labels and title
            ax_omp.set_title("OMP Feature Coefficients", fontdict=PLOT_STYLE)
            ax_omp.set_xlabel("Selected Features", fontdict=PLOT_STYLE)
            ax_omp.set_ylabel("Coefficient Magnitude", fontdict=PLOT_STYLE)
            ax_omp.set_xticks(range(len(sorted_names)))
            ax_omp.set_xticklabels(sorted_names, rotation=45, ha='right')
            
            # Add legend for coefficient sign
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#3F7C85', label='Positive Coefficient'),
                Patch(facecolor='#A60A33', label='Negative Coefficient')
            ]
            ax_omp.legend(handles=legend_elements, loc='upper right')
            
            # Add text about number of features
            ax_omp.text(
                0.02, 0.95, 
                f"OMP selected {len(self.selected_features)} features",
                transform=ax_omp.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Remove top and right spines
            ax_omp.spines["top"].set_visible(False)
            ax_omp.spines["right"].set_visible(False)
        
        # Plot XGBoost feature importance
        if plot_type in ["xgboost", "both"]:
            # For 'both' plot type, use the second axis
            ax_xgb = ax2 if plot_type == "both" else ax
            
            # Get feature importance
            if hasattr(self.xgb_model, 'feature_importances_'):
                importances = self.xgb_model.feature_importances_
                
                if len(importances) > 0:
                    # Sort by importance
                    sorted_idx = np.argsort(importances)[::-1]
                    pos = np.arange(sorted_idx.shape[0])
                    
                    # Plot bars
                    ax_xgb.barh(
                        pos, 
                        importances[sorted_idx],
                        align='center',
                        color='#3F7C85'
                    )
                    
                    # Add feature names if possible
                    ax_xgb.set_yticks(pos)
                    if len(self.selected_feature_names) == len(importances):
                        # Map indices back to actual feature names
                        labels = [self.selected_feature_names[i] for i in sorted_idx]
                        ax_xgb.set_yticklabels(labels)
                    else:
                        # Use generic labels if names don't match
                        ax_xgb.set_yticklabels([f"Feature {i}" for i in sorted_idx])
                    
                    # Set title and labels
                    ax_xgb.set_title("XGBoost Feature Importance", fontdict=PLOT_STYLE)
                    ax_xgb.set_xlabel("Importance", fontdict=PLOT_STYLE)
                    
                    # Remove top and right spines
                    ax_xgb.spines["top"].set_visible(False)
                    ax_xgb.spines["right"].set_visible(False)
                else:
                    if plot_type == "both":
                        ax_xgb.set_title("XGBoost Feature Importance Not Available", fontdict=PLOT_STYLE)
                    else:
                        print("XGBoost feature importance not available.")
            else:
                if plot_type == "both":
                    ax_xgb.set_title("XGBoost Feature Importance Not Available", fontdict=PLOT_STYLE)
                else:
                    print("XGBoost feature importance not available.")
        
        # Save or display the plot
        if filename:
            # Build output path
            base_output_dir = os.path.abspath(os.path.join("..", "..", "output", "images"))
            if output_dir:
                full_output_dir = os.path.join(base_output_dir, output_dir)
                os.makedirs(full_output_dir, exist_ok=True)
                output_path = os.path.join(full_output_dir, f"{filename}.jpeg")
            else:
                os.makedirs(base_output_dir, exist_ok=True)
                output_path = os.path.join(base_output_dir, f"{filename}.jpeg")
            
            # Save and close
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary information about selected features.
        
        Returns:
            Dictionary with feature selection details
        """
        if not self.is_trained:
            self.train_and_predict()
            
        # Create summary dictionary
        summary = {
            "n_original_features": len(self.feature_names),
            "n_selected_features": len(self.selected_features) if self.selected_features is not None else 0,
            "selected_indices": self.selected_features.tolist() if self.selected_features is not None else [],
            "selected_names": self.selected_feature_names if self.selected_feature_names is not None else [],
        }
        
        # Add coefficient information if available
        if self.omp_coefficients is not None and self.selected_feature_names is not None:
            coef_dict = {}
            for name, coef in zip(self.selected_feature_names, self.omp_coefficients):
                coef_dict[name] = round(float(coef), 4)
            summary["coefficients"] = coef_dict
            
        return summary
        
    def print_feature_summary(self) -> None:
        """
        Print detailed information about selected features.
        """
        if not self.is_trained:
            self.train_and_predict()
            
        # Print summary header
        print("\n===== Feature Selection Summary =====")
        print(f"Total features: {len(self.feature_names)}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Selection ratio: {len(self.selected_features)/len(self.feature_names):.2%}")
        
        # Print selection method details
        print(f"Selection method: {'CV-OMP' if self.use_cv_omp else 'Standard OMP'}")
        if self.n_features_to_select is not None:
            print(f"Selection criteria: Top {self.n_features_to_select} features by coefficient magnitude")
        else:
            print(f"Selection criteria: Coefficient magnitude > {self.min_coef_value:.2e}")
        
        # Print selected feature details
        if len(self.selected_features) > 0:
            print("\n----- Selected Feature Details -----")
            print("Index | Feature Name | OMP Coefficient")
            print("-" * 50)
            
            for i, idx in enumerate(self.selected_features):
                feature_name = self.selected_feature_names[i] if self.selected_feature_names else f"feature_{idx}"
                coef = self.omp_coefficients[i]
                print(f"{idx:5d} | {feature_name:30s} | {coef:+.6f}")
        
        # Print XGBoost feature importance if available
        if hasattr(self.xgb_model, 'feature_importances_') and len(self.xgb_model.feature_importances_) > 0:
            print("\n----- XGBoost Feature Importance -----")
            for i, importance in enumerate(self.xgb_model.feature_importances_):
                feature_name = self.selected_feature_names[i] if i < len(self.selected_feature_names) else f"Feature {i}"
                print(f"{feature_name}: {importance:.4f} ({importance*100:.2f}%)")
                
        print("\n===== End of Summary =====")


# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    # Create feature names
    feature_names = [f"histone_mark_{i+1}" for i in range(n_features)]
    
    # Generate random features with some correlation structure
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with dependency on only a few features
    y = 3.0 * X[:, 2] + 2.0 * X[:, 5] - 1.5 * X[:, 10] + 0.5 * np.random.randn(n_samples)
    
    # Create and train the model with default settings (using coefficient threshold)
    model = ManualOMPHistoneRegressor(features=X, target=y, feature_names=feature_names)
    
    # Evaluate model
    r2 = model.calculate_error(method="r2")
    print(f"R² Score (threshold-based selection): {r2:.3f}")
    
    # Print feature summary
    model.print_feature_summary()
    
    # Create and train a model with fixed number of features
    model_fixed = ManualOMPHistoneRegressor(
        features=X, 
        target=y, 
        feature_names=feature_names,
        n_features_to_select=5  # Select exactly 5 features
    )
    
    # Evaluate fixed feature count model
    r2_fixed = model_fixed.calculate_error(method="r2")
    print(f"R² Score (fixed 5 features): {r2_fixed:.3f}")
    
    # Print feature summary for fixed model
    model_fixed.print_feature_summary()
    
    # Plot feature importance for both models
    model.plot_feature_importance(plot_type="both", filename="threshold_based_importance")
    model_fixed.plot_feature_importance(plot_type="both", filename="fixed_feature_importance")