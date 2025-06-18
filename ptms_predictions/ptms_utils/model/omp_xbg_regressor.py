
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import StandardScaler
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

class HistoneRegressor:
    """
    Regressor model for histone modification prediction with OMP feature selection.
    
    This class implements a machine learning pipeline that:
    1. Standardizes features
    2. Performs feature selection using Orthogonal Matching Pursuit (OMP)
    3. Builds a prediction model using XGBoost Regression
    
    The implementation supports both regular training/testing and k-fold cross-validation.
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
        omp_cv: int = 10,
        feature_threshold: float = 1e-8
    ) -> None:
        """
        Initialize the HistoneRegressor model.
        
        Args:
            features: Input feature matrix
            target: Target values to predict
            feature_names: Names of features (optional)
            use_cross_validation: Whether to use k-fold cross-validation for evaluation
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in XGBoost ensemble
            max_depth: Maximum depth of trees in XGBoost
            omp_cv: Number of cross-validation folds for OMP
            feature_threshold: Threshold for feature selection
        """
        self.features = features
        self.target = target
        self.use_cross_validation = use_cross_validation
        self.random_state = random_state
        self.selected_features = None
        self.selected_feature_names = None
        self.omp_coefficients = None
        self.is_trained = False
        
        # Generate default feature names if not provided
        if feature_names is None and isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
        elif feature_names is None:
            n_features = features.shape[1] if np.ndim(features) > 1 else 1
            self.feature_names = [f"feature_{i+1}" for i in range(n_features)]
        else:
            self.feature_names = feature_names
        
        # Define pipeline: scaling -> OMP feature selection -> XGBoost
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selector', SelectFromModel(
                OrthogonalMatchingPursuitCV(cv=omp_cv, max_iter=10), 
                threshold=feature_threshold
            )),
            ('regressor', xgb.XGBRegressor(
                n_jobs=-1,
                n_estimators=n_estimators,
                random_state=random_state,
                max_depth=max_depth
            ))
        ])

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

        # Fit the model
        self.model.fit(x_train, y_train)
        self.is_trained = True
        
        # Extract OMP feature information
        self._extract_omp_information()

        # Make predictions
        y_pred = self.model.predict(x_test)
        return y_test, y_pred
    
    def _extract_omp_information(self) -> None:
        """
        Extract OMP feature selection information after model training.
        
        This method extracts selected feature indices, names, and coefficients
        from the trained OMP model in the pipeline.
        """
        # Get the OMP estimator from the pipeline
        omp_estimator = self.model.named_steps['feature_selector'].estimator_
        
        # Extract selected feature indices (non-zero coefficients)
        self.selected_features = np.where(omp_estimator.coef_ != 0)[0]
        
        # Get the coefficient values for selected features
        self.omp_coefficients = omp_estimator.coef_[self.selected_features]
        
        # Get feature names if available
        if self.feature_names is not None:
            if len(self.selected_features) > 0:
                # Ensure feature_names has correct length
                if len(self.feature_names) >= max(self.selected_features) + 1:
                    self.selected_feature_names = [self.feature_names[i] for i in self.selected_features]
                else:
                    self.selected_feature_names = [f"feature_{i+1}" for i in self.selected_features]
            else:
                self.selected_feature_names = []
                
        # Store number of selected features
        self.n_selected_features = len(self.selected_features)

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
            Calculated error value
        """
        # Ensure features are in correct format
        features = self._ensure_2d_features(self.features)
        
        # Create KFold cross-validator
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        # Select appropriate scoring metric
        if method == "mse":
            scoring = "neg_mean_squared_error"
        elif method == "rmse":
            scoring = "neg_root_mean_squared_error"
        else:  # r2
            scoring = "r2"
        
        # Perform cross-validation
        scores = cross_val_score(
            self.model, 
            features, 
            self.target,
            scoring=scoring, 
            cv=kfold, 
            n_jobs=-1
        )
        
        # Process scores based on method
        if method in ["mse", "rmse"]:
            # Convert negative scores to positive
            score = np.mean(-scores)
        else:  # r2
            score = np.mean(scores)
            
        # Train the model after cross-validation to extract feature information
        if not self.is_trained:
            self.train_and_predict()
            
        return np.round(score, 3)

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
            # Access the XGBoost model from the pipeline
            xgb_model = self.model.named_steps['regressor']
            
            # For 'both' plot type, use the second axis
            ax_xgb = ax2 if plot_type == "both" else ax
            
            # Get feature importance for selected features
            if hasattr(xgb_model, 'feature_importances_'):
                # Get feature importances from model
                # Note: XGBoost importance is for features AFTER OMP selection
                importances = xgb_model.feature_importances_
                
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
                    
                    # Add labels - these are positional indices after OMP selection
                    # We can't map directly to original feature names
                    ax_xgb.set_yticks(pos)
                    ax_xgb.set_yticklabels([f"Feature {i}" for i in sorted_idx])
                    
                    # Set title and labels
                    ax_xgb.set_title("XGBoost Feature Importance (after OMP selection)", fontdict=PLOT_STYLE)
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
            
        # Get OMP estimator from pipeline
        omp_estimator = self.model.named_steps['feature_selector'].estimator_
        
        # Print summary header
        print("\n===== Feature Selection Summary =====")
        print(f"Total features: {len(self.feature_names)}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Selection ratio: {len(self.selected_features)/len(self.feature_names):.2%}")
        
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
        xgb_model = self.model.named_steps['regressor']
        if hasattr(xgb_model, 'feature_importances_') and len(xgb_model.feature_importances_) > 0:
            print("\n----- XGBoost Feature Importance -----")
            # Note: XGBoost importance is for features AFTER OMP selection
            for i, importance in enumerate(xgb_model.feature_importances_):
                # print them in percentage 
                print(f"Feature {i}: {importance*100}")
                
        print("\n===== End of Summary =====")


def calculate_r2_score(feature_data: np.ndarray, target_data: np.ndarray) -> float:
    """
    Calculate R² score using the HistoneRegressor model.
    
    Args:
        feature_data: Input features
        target_data: Target values
        
    Returns:
        R² score
    """
    model = HistoneRegressor(features=feature_data, target=target_data)
    return model.calculate_error(method="r2")


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
    
    # Create and train the model
    model = HistoneRegressor(features=X, target=y, feature_names=feature_names)
    
    # Evaluate model
    r2 = model.calculate_error(method="r2")
    print(f"R² Score: {r2:.3f}")
    
    # Print feature summary
    model.print_feature_summary()
    
    # Plot feature importance
    model.plot_feature_importance(plot_type="both")
    
    # Plot predictions
    model.plot_predictions(title_info=("Histone Marks", "Modification Level"))
