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
        feature_threshold: float = 1e-8,
        n_features_to_select: Optional[int] = None  
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
            n_features_to_select: Number of top features to select (if None, uses threshold-based selection)
        """
        self.features = features
        self.target = target
        self.use_cross_validation = use_cross_validation
        self.random_state = random_state
        self.selected_features = None
        self.selected_feature_names = None
        self.omp_coefficients = None
        self.is_trained = False
        self.n_features_to_select = n_features_to_select
        self.omp_cv = omp_cv
        
        # Generate default feature names if not provided
        if feature_names is None and isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
        elif feature_names is None:
            n_features = features.shape[1] if np.ndim(features) > 1 else 1
            self.feature_names = [f"feature_{i+1}" for i in range(n_features)]
        else:
            self.feature_names = feature_names
        
        # Define pipeline based on selection method
        if n_features_to_select is not None:
            # For explicit feature count selection, we'll use a custom approach
            # but still define the basic pipeline for compatibility
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selector', SelectFromModel(
                    OrthogonalMatchingPursuitCV(cv=omp_cv, max_iter=n_features_to_select), 
                    max_features=n_features_to_select
                )),
                ('regressor', xgb.XGBRegressor(
                    n_jobs=-1,
                    n_estimators=n_estimators,
                    random_state=random_state,
                    max_depth=max_depth
                ))
            ])
        else:
            # Original approach with threshold-based selection
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

    def select_top_n_features(self, n_features: int) -> np.ndarray:
        """
        Select top n features based on absolute coefficient values.
        
        This method fits an OMP model directly and extracts the n most important features
        based on the magnitude of their coefficients.
        
        Args:
            n_features: Number of top features to select
            
        Returns:
            Indices of the selected top features
        """
        # Ensure features are in correct shape
        features = self._ensure_2d_features(self.features)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        # Fit OMP with cross-validation
        omp = OrthogonalMatchingPursuitCV(cv=self.omp_cv, max_iter=min(X_scaled.shape[1], 30))
        omp.fit(X_scaled, self.target)
        
        # Get absolute coefficients
        abs_coefs = np.abs(omp.coef_)
        
        # Sort features by descending absolute coefficient value
        sorted_indices = np.argsort(abs_coefs)[::-1]
        
        # Select top n features (ensure we don't exceed available features)
        n_to_select = min(n_features, len(sorted_indices))
        top_n_features = sorted_indices[:n_to_select]
        
        # Store selected feature information
        self.selected_features = top_n_features
        self.omp_coefficients = omp.coef_[top_n_features]
        
        if self.feature_names is not None:
            self.selected_feature_names = [self.feature_names[i] for i in top_n_features]
        
        return top_n_features

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

        # Check if we're working with a DataFrame
        is_dataframe_train = isinstance(x_train, pd.DataFrame)
        is_dataframe_test = isinstance(x_test, pd.DataFrame)

        # For explicit n feature selection, use our custom method
        if self.n_features_to_select is not None:
            # Get top n features
            selected_indices = self.select_top_n_features(self.n_features_to_select)
            
            # Extract only the selected features - handle DataFrames properly
            if is_dataframe_train:
                x_train_selected = x_train.iloc[:, selected_indices].values
            else:
                # Ensure features are in correct shape for numpy arrays
                x_train = self._ensure_2d_features(x_train)
                x_train_selected = x_train[:, selected_indices]
            
            if is_dataframe_test:
                x_test_selected = x_test.iloc[:, selected_indices].values
            else:
                # Ensure features are in correct shape for numpy arrays
                x_test = self._ensure_2d_features(x_test)
                x_test_selected = x_test[:, selected_indices]
            
            # Train the regressor directly on selected features
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train_selected)
            x_test_scaled = scaler.transform(x_test_selected)
            
            # Train the XGBoost regressor
            regressor = self.model.named_steps['regressor']
            regressor.fit(x_train_scaled, y_train)
            
            # Make predictions
            y_pred = regressor.predict(x_test_scaled)
            self.is_trained = True
            return y_test, y_pred
        else:
            # Use the original pipeline approach
            if is_dataframe_train:
                x_train = x_train.values
            else:
                x_train = self._ensure_2d_features(x_train)
                
            if is_dataframe_test:
                x_test = x_test.values
            else:
                x_test = self._ensure_2d_features(x_test)
                
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
        # Create KFold cross-validator
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        # Check if features is a DataFrame
        is_dataframe = isinstance(self.features, pd.DataFrame)
        
        # For n_features_to_select, we need a custom CV approach
        if self.n_features_to_select is not None:
            cv_scores = []
            
            for train_idx, test_idx in kfold.split(self.features):
                # Split data - handle DataFrames properly
                if is_dataframe:
                    X_train = self.features.iloc[train_idx].values
                    X_test = self.features.iloc[test_idx].values
                    y_train = self.target[train_idx]
                    y_test = self.target[test_idx]
                else:
                    # Ensure features are in correct format for numpy arrays
                    features = self._ensure_2d_features(self.features)
                    X_train, X_test = features[train_idx], features[test_idx]
                    y_train, y_test = self.target[train_idx], self.target[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Select features using OMP
                omp = OrthogonalMatchingPursuitCV(cv=5, max_iter=min(X_train_scaled.shape[1], 30))
                omp.fit(X_train_scaled, y_train)
                
                # Get top n features by coefficient magnitude
                abs_coefs = np.abs(omp.coef_)
                top_indices = np.argsort(abs_coefs)[::-1][:self.n_features_to_select]
                
                # Train XGBoost on selected features
                xgb_model = xgb.XGBRegressor(
                    n_jobs=-1,
                    n_estimators=self.model.named_steps['regressor'].n_estimators,
                    random_state=self.random_state,
                    max_depth=self.model.named_steps['regressor'].max_depth
                )
                xgb_model.fit(X_train_scaled[:, top_indices], y_train)
                
                # Predict and calculate score
                y_pred = xgb_model.predict(X_test_scaled[:, top_indices])
                
                # Calculate appropriate score
                if method == "mse":
                    score = mean_squared_error(y_test, y_pred)
                elif method == "rmse":
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                else:  # r2
                    score = r2_score(y_test, y_pred)
                    
                cv_scores.append(score)
            
            # Process scores
            score = np.mean(cv_scores)
        else:
            # Convert DataFrame to numpy array if needed for standard pipeline approach
            if is_dataframe:
                features_array = self.features.values
            else:
                features_array = self._ensure_2d_features(self.features)
            
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
                features_array, 
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
        figsize: Tuple[int, int] = (14, 12),
        max_name_length: int = 20,  # Limit long feature names
        label_fontsize: int = 10    # Control font size of labels
    ) -> None:
        """
        Plot feature importance from OMP coefficients and/or XGBoost.
        
        Args:
            plot_type: Type of plot to generate ('omp', 'xgboost', or 'both')
            filename: Name to save the plot (without extension)
            output_dir: Directory to save the plot in
            figsize: Figure size as (width, height)
            max_name_length: Maximum length for feature names (truncates with ...)
            label_fontsize: Font size for feature name labels
        """
        # Ensure model is trained
        if not self.is_trained:
            self.train_and_predict()
            
        if not hasattr(self, 'selected_features') or self.selected_features is None or not self.selected_features.size:
            print("No features were selected by OMP. Cannot plot feature importance.")
            return
        
        # Helper function to truncate long feature names
        def truncate_name(name, max_len):
            return name if len(name) <= max_len else name[:max_len-3] + '...'
            
        # Set up figure based on plot type
        if plot_type == "both":
            # Adjust figure height based on number of features
            n_features = len(self.selected_features)
            adjusted_height = max(12, n_features * 0.6)  # Minimum height of 12, but scales with feature count
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], adjusted_height))
            fig.tight_layout(pad=5.0)  # Increase padding between subplots
        else:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot OMP coefficients
        if plot_type in ["omp", "both"]:
            ax_omp = ax1 if plot_type == "both" else ax
            
            # Sort features by coefficient magnitude for better visualization
            sorted_indices = np.argsort(np.abs(self.omp_coefficients))[::-1]
            sorted_coefs = self.omp_coefficients[sorted_indices]
            
            # Truncate feature names if they're too long
            if self.selected_feature_names is not None:
                sorted_names = [truncate_name(self.selected_feature_names[i], max_name_length) 
                                for i in sorted_indices]
            else:
                sorted_names = [f"Feature {i}" for i in sorted_indices]
            
            # Create colormap based on coefficient sign
            colors = ['#3F7C85' if c >= 0 else '#A60A33' for c in sorted_coefs]
            
            # Plot bars with more spacing
            bar_width = 0.7  # Narrower bars for more spacing
            bars = ax_omp.bar(range(len(sorted_names)), np.abs(sorted_coefs), width=bar_width, color=colors)
            
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
            
            # Set x-ticks and labels with better spacing
            ax_omp.set_xticks(range(len(sorted_names)))
            ax_omp.set_xticklabels(sorted_names, rotation=60, ha='right', fontsize=label_fontsize)
            
            # Add some extra space at the bottom for the labels
            ax_omp.margins(y=0.2, x=0.01)
            
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
            if self.n_features_to_select is not None:
                # For custom approach, the regressor was trained directly
                xgb_model = self.model.named_steps['regressor']
                # Note: We need to fit it first if using this branch
                if not hasattr(xgb_model, 'feature_importances_'):
                    print("XGBoost model not trained yet for feature importance analysis.")
                    return
            else:
                # For pipeline approach, the regressor is in the pipeline
                xgb_model = self.model.named_steps['regressor']
            
            # For 'both' plot type, use the second axis
            ax_xgb = ax2 if plot_type == "both" else ax
            
            # Get feature importance for selected features
            if hasattr(xgb_model, 'feature_importances_'):
                # Get feature importances from model
                importances = xgb_model.feature_importances_
                
                if len(importances) > 0:
                    # Sort by importance
                    sorted_idx = np.argsort(importances)[::-1]
                    pos = np.arange(sorted_idx.shape[0])
                    
                    # Prepare feature labels with truncation if necessary
                    if self.selected_feature_names is not None and len(self.selected_feature_names) == len(importances):
                        # Properly map XGBoost importance indices to OMP-selected feature names
                        feature_labels = [truncate_name(self.selected_feature_names[i], max_name_length) 
                                         for i in sorted_idx]
                    else:
                        feature_labels = [f"Feature {i}" for i in sorted_idx]
                    
                    # Create horizontal spacing
                    bar_height = 0.7  # Control bar height
                    
                    # Plot bars with improved spacing
                    ax_xgb.barh(
                        pos, 
                        importances[sorted_idx],
                        height=bar_height,
                        align='center',
                        color='#3F7C85'
                    )
                    
                    # Add importance values as text
                    for i, importance in enumerate(importances[sorted_idx]):
                        ax_xgb.text(
                            importance + 0.01,
                            i,
                            f'{importance:.4f}',
                            va='center',
                            fontsize=8
                        )
                    
                    # Add labels with better spacing
                    ax_xgb.set_yticks(pos)
                    ax_xgb.set_yticklabels(feature_labels, fontsize=label_fontsize)
                    
                    # Add spacing at the edges
                    ax_xgb.margins(x=0.1, y=0.05)
                    
                    # Set title and labels
                    ax_xgb.set_title("XGBoost Feature Importance", fontdict=PLOT_STYLE)
                    ax_xgb.set_xlabel("Importance", fontdict=PLOT_STYLE)
                    
                    # Remove top and right spines
                    ax_xgb.spines["top"].set_visible(False)
                    ax_xgb.spines["right"].set_visible(False)
                    
                    # Add grid lines for better readability
                    ax_xgb.grid(True, axis='x', linestyle='--', alpha=0.6)
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
            if output_dir:
                # If absolute path is provided, use it directly
                if os.path.isabs(output_dir):
                    full_output_dir = output_dir
                else:
                    # Otherwise construct relative to project structure
                    base_output_dir = os.path.abspath(os.path.join("..", "..", "output", "images"))
                    full_output_dir = os.path.join(base_output_dir, output_dir)
            else:
                base_output_dir = os.path.abspath(os.path.join("..", "..", "output", "images"))
                full_output_dir = base_output_dir
            
            # Create directory if doesn't exist
            os.makedirs(full_output_dir, exist_ok=True)
            output_path = os.path.join(full_output_dir, f"{filename}.jpeg")
            
            # Save with tight layout to avoid label cutoff
            plt.tight_layout()
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
        print(f"Selected features: {len(self.selected_features) if self.selected_features is not None else 0}")
        
        if self.selected_features is not None and len(self.selected_features) > 0:
            print(f"Selection ratio: {len(self.selected_features)/len(self.feature_names):.2%}")
            
            # Print selected feature details
            print("\n----- Selected Feature Details -----")
            print("Index | Feature Name | OMP Coefficient")
            print("-" * 50)
            
            # Sort by absolute coefficient value to show importance ranking
            if self.omp_coefficients is not None and self.selected_feature_names is not None:
                # Get sorting indices based on absolute coefficient values
                sorted_idx = np.argsort(np.abs(self.omp_coefficients))[::-1]
                
                # Print features in order of importance
                for i, idx in enumerate(sorted_idx):
                    feature_idx = self.selected_features[idx]
                    feature_name = self.selected_feature_names[idx]
                    coef = self.omp_coefficients[idx]
                    print(f"{i+1:3d} | {feature_name:30s} | {coef:+.6f}")
            
            # Print XGBoost feature importance if available
            if self.n_features_to_select is None:
                xgb_model = self.model.named_steps['regressor']
                if hasattr(xgb_model, 'feature_importances_') and len(xgb_model.feature_importances_) > 0:
                    print("\n----- XGBoost Feature Importance -----")
                    # Note: XGBoost importance is for features AFTER OMP selection
                    for i, importance in enumerate(xgb_model.feature_importances_):
                        # print them in percentage 
                        print(f"Feature {i}: {importance*100:.2f}%")
        else:
            print("No features were selected.")
                
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
    
    # Create and train the model with top 5 features
    model = HistoneRegressor(features=X, target=y, feature_names=feature_names, n_features_to_select=5)
    
    # Evaluate model
    r2 = model.calculate_error(method="r2")
    print(f"R² Score with top 5 features: {r2:.3f}")
    
    # Print feature summary - this will show features in order of importance
    # Print feature summary
    model.print_feature_summary()
    
    # Plot feature importance
    model.plot_feature_importance(plot_type="both")
    
    # Plot predictions
    model.plot_predictions(title_info=("Histone Marks", "Modification Level"))
