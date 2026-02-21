from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import warnings
from typing import Literal, Optional, Dict, List
import shap

warnings.filterwarnings("ignore")
font_dict = {"fontsize": 12, "family": "serif", "weight": "bold"}


class Histone_Regressor_shap:
    """Regressor model for histone modification prediction"""

    def __init__(
        self,
        feature,
        target,
        cross_valid=True,
        test_size=0.33,
        random_state=42,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the Histone_Regressor with data and model parameters.
        Parameters
        ----------
        feature : np.array
            Feature data array
        target : np.array
            Target data array
        cross_valid : bool, optional
            Whether to perform cross-validation, by default True
        test_size : float, optional
            Proportion of data for test set, by default 0.33
        random_state : int, optional
            Random seed for reproducibility, by default 42
        feature_names : List[str], optional
            List of feature names for SHAP analysis, by default None
        """

        self.feature = feature
        self.target = target
        self.cross_valid = cross_valid
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = feature_names
        self.model = xgb.XGBRFRegressor(
            n_jobs=-1,
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
        )

        # Store CV results
        self.cv_scores_ = None
        self.cv_mean_ = None
        self.cv_std_ = None
        self.test_score_ = None

        # Store SHAP values
        self.shap_values_ = None
        self.shap_explainer_ = None

        # Create train/test split upfront, so it's available for all methods
        self._create_train_test_split()

    def _create_train_test_split(self):
        """Create train test split."""
        if np.ndim(self.feature) < 2:
            feature = np.array(self.feature).reshape(-1, 1)
        else:
            feature = np.array(self.feature)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            feature,
            self.target,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # Set default feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(self.X_train.shape[1])]

    def xgb_regressor_pred(self):
        """Train model and predict on test set."""
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        return self.y_test, y_pred

    def error_calculator_with_cv(
        self, methods: Literal["r2", "mse", "rmse"] = "r2"
    ) -> Dict:
        """
        Perform 10-fold CV on training set and final evaluation on test set.

        Returns
        -------
        Dict
            Dictionary containing cv_scores, cv_mean, cv_std, and test_score
        """
        # Define 10-fold CV strategy
        cv_strategy = KFold(n_splits=10, shuffle=True, random_state=self.random_state)

        # Determine scoring metric
        if methods == "mse":
            scoring = "neg_mean_squared_error"
        elif methods == "rmse":
            scoring = "neg_root_mean_squared_error"
        else:
            scoring = "r2"

        # Run 10-fold CV ONLY on the training set (67%)
        cv_scores = cross_val_score(
            self.model,
            X=self.X_train,
            y=self.y_train,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=-1,
            error_score="raise",
        )

        # Convert negative scores to positive for mse/rmse
        if methods in ["mse", "rmse"]:
            cv_scores = np.abs(cv_scores)

        # Store CV results
        self.cv_scores_ = cv_scores
        self.cv_mean_ = cv_scores.mean()
        self.cv_std_ = cv_scores.std()

        # Final validation on the independent test set (33%)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        if methods == "mse":
            self.test_score_ = np.square(root_mean_squared_error(self.y_test, y_pred))
        elif methods == "rmse":
            self.test_score_ = root_mean_squared_error(self.y_test, y_pred)
        else:
            self.test_score_ = r2_score(self.y_test, y_pred)

        return {
            "cv_scores": self.cv_scores_,
            "cv_mean": self.cv_mean_,
            "cv_std": self.cv_std_,
            "test_score": self.test_score_,
        }

    def error_calculator_norm(self, methods: Literal["r2", "mse", "rmse"] = "r2"):
        """Calculate the error score without cross-validation."""
        y_test, y_pred = self.xgb_regressor_pred()

        if methods == "mse":
            error = np.round(np.square(root_mean_squared_error(y_test, y_pred)), 3)
        elif methods == "rmse":
            error = np.round(root_mean_squared_error(y_test, y_pred), 3)
        else:
            error = np.round(r2_score(y_test, y_pred), 3)

        return error

    def error_calculator(self, methods: Literal["r2", "mse", "rmse"] = "r2") -> float:
        """Error calculation between the origin and predicted target data.

        Parameters
        ----------
        methods : str, optional
            error_methods, by default "r2"

        Returns
        -------
        float
            Mean CV score if cross_valid=True, else single score
        """
        if self.cross_valid:
            results = self.error_calculator_with_cv(methods=methods)
            return results
        else:
            return self.error_calculator_norm(methods=methods)

    def get_cv_results(self, methods: Literal["r2", "mse", "rmse"] = "r2") -> Dict:
        """
        Get detailed CV results including all fold scores, mean, std, and test score.

        Returns
        -------
        Dict
            Dictionary with cv_scores, cv_mean, cv_std, test_score
        """
        if self.cv_scores_ is None:
            self.error_calculator_with_cv(methods=methods)

        return {
            "cv_scores": self.cv_scores_,
            "cv_mean": self.cv_mean_,
            "cv_std": self.cv_std_,
            "test_score": self.test_score_,
        }

    def save_cv_results_to_csv(
        self,
        file_path: str,
        methods: Literal["r2", "mse", "rmse"] = "r2",
        feature_name: Optional[str] = None,
        target_name: Optional[str] = None,
        append: bool = True,
    ) -> pd.DataFrame:
        """
        Save CV results (all fold scores, mean, std, test score) to a CSV file.

        Parameters
        ----------
        file_path : str
            Path to save the CSV file
        methods : str, optional
            Scoring method, by default "r2"
        feature_name : str, optional
            Name of the feature(s) for identification
        target_name : str, optional
            Name of the target for identification
        append : bool, optional
            If True, append to existing file; if False, overwrite

        Returns
        -------
        pd.DataFrame
            DataFrame containing the saved results
        """
        # Ensure CV has been run
        if self.cv_scores_ is None:
            self.error_calculator_with_cv(methods=methods)

        # Create results dictionary
        results_dict = {
            "feature_name": feature_name or "unknown",
            "target_name": target_name or "unknown",
            "metric": methods,
            "cv_mean": np.round(self.cv_mean_, 4),
            "cv_std": np.round(self.cv_std_, 4),
            "test_score": np.round(self.test_score_, 4),
        }

        # Add individual fold scores
        for i, score in enumerate(self.cv_scores_):
            results_dict[f"fold_{i+1}"] = np.round(score, 4)

        # Create DataFrame
        results_df = pd.DataFrame([results_dict])

        # Save to CSV
        (
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.dirname(file_path)
            else None
        )

        if append and os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)

        results_df.to_csv(file_path, index=False)
        # print(f"CV results saved to: {file_path}")

        return results_df

    def print_cv_summary(self, methods: Literal["r2", "mse", "rmse"] = "r2"):
        """Print a summary of CV results."""
        if self.cv_scores_ is None:
            self.error_calculator_with_cv(methods=methods)

        print(f"\n{'='*60}")
        print(f"Cross-Validation Results Summary ({methods.upper()})")
        print(f"{'='*60}")
        print(f"10-Fold CV Score (on 67% training data):")
        print(f"  Mean: {self.cv_mean_:.4f} +/- {self.cv_std_:.4f}")
        print(f"  Individual folds: {[f'{s:.4f}' for s in self.cv_scores_]}")
        print(f"\nIndependent Test Set Score (on 33% hold-out data):")
        print(f"  Score: {self.test_score_:.4f}")
        print(f"{'='*60}\n")

    def compute_shap_values(self, use_test_data: bool = True) -> np.ndarray:
        """
        Compute SHAP values for feature importance analysis.
        """
        # Fit model if not already done
        try:
            self.model.feature_importances_
        except AttributeError:
            self.model.fit(self.X_train, self.y_train)

        # Create SHAP explainer for tree-based model
        self.shap_explainer_ = shap.TreeExplainer(self.model)

        # Compute SHAP values based on test or train data
        data = self.X_test if use_test_data else self.X_train
        self.shap_values_ = self.shap_explainer_.shap_values(data)

        return self.shap_values_

    def get_global_feature_importance(
        self, method: Literal["mean_abs", "mean", "max"] = "mean_abs"
    ) -> pd.DataFrame:
        """
        Calculate global feature importance from SHAP values.

        Parameters
        ----------
        method : str, optional
            Method to aggregate SHAP values:
            - "mean_abs": Mean absolute SHAP value (default, most common)
            - "mean": Mean SHAP value (shows direction of impact)
            - "max": Maximum absolute SHAP value

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and their importance scores, sorted descending
        """
        if self.shap_values_ is None:
            self.compute_shap_values()

        if method == "mean_abs":
            importance = np.abs(self.shap_values_).mean(axis=0)
        elif method == "mean":
            importance = self.shap_values_.mean(axis=0)
        elif method == "max":
            importance = np.abs(self.shap_values_).max(axis=0)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'mean_abs', 'mean', or 'max'."
            )

        importance_df = (
            pd.DataFrame({"feature": self.feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance_df

    def plot_shap_summary(
        self,
        plot_type: Literal["bar", "beeswarm", "violin"] = "bar",
        max_display: int = 20,
        file_name: Optional[str] = None,
        new_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Plot SHAP summary for global feature importance.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot: "bar", "beeswarm", or "violin"
        max_display : int, optional
            Maximum number of features to display
        file_name : str, optional
            If provided, save the plot to this file
        new_dir : str, optional
            Subdirectory within output/images to save the plot
        output_dir : str, optional
            Absolute path to output directory. If None, uses default relative path.
        """
        import matplotlib

        # Use non-interactive backend for thread-safety when saving files
        if file_name:
            matplotlib.use("Agg")

        if self.shap_values_ is None:
            self.compute_shap_values()

        # Create DataFrame for plotting with feature names
        data = self.X_test if hasattr(self, "X_test") else self.X_train
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)

        # Close any existing figures to prevent conflicts
        plt.close("all")

        # Create figure and axis explicitly
        fig, ax = plt.subplots(figsize=(12, 8))

        if plot_type == "bar":
            shap.summary_plot(
                self.shap_values_,
                data,
                plot_type="bar",
                max_display=max_display,
                show=False,
            )
        elif plot_type == "beeswarm":
            shap.summary_plot(
                self.shap_values_, data, max_display=max_display, show=False
            )
        elif plot_type == "violin":
            shap.summary_plot(
                self.shap_values_,
                data,
                plot_type="violin",
                max_display=max_display,
                show=False,
            )

        # Get the current figure (SHAP may have created a new one)
        fig = plt.gcf()
        fig.tight_layout()

        if file_name:
            # Use provided output_dir or construct from current file location
            if output_dir is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(
                    os.path.join(current_dir, "..", "..", "..")
                )
                output_dir = os.path.join(project_root, "output", "images")

            if new_dir:
                full_output_dir = os.path.join(output_dir, new_dir)
            else:
                full_output_dir = output_dir

            os.makedirs(full_output_dir, exist_ok=True)
            output_path = os.path.join(full_output_dir, f"{file_name}.png")

            # Save the current figure (which SHAP created)
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close("all")  # Close all figures to free memory
            print(f"SHAP plot saved to: {output_path}")
        else:
            plt.show()

    def plot_shap_waterfall(
        self,
        sample_idx: int = 0,
        file_name: Optional[str] = None,
        new_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Plot SHAP waterfall for a single prediction explanation.

        Parameters
        ----------
        sample_idx : int, optional
            Index of the sample to explain
        file_name : str, optional
            If provided, save the plot to this file
        new_dir : str, optional
            Subdirectory within output/images to save the plot
        output_dir : str, optional
            Absolute path to output directory. If None, uses default relative path.
        """
        if self.shap_explainer_ is None:
            self.compute_shap_values()

        # Get SHAP explanation for the specific sample
        explanation = self.shap_explainer_(self.X_test)
        explanation.feature_names = self.feature_names

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation[sample_idx], show=False)
        plt.tight_layout()

        if file_name:
            # Use provided output_dir or construct from current file location
            if output_dir is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(
                    os.path.join(current_dir, "..", "..", "..")
                )
                output_dir = os.path.join(project_root, "output", "images")

            if new_dir:
                full_output_dir = os.path.join(output_dir, new_dir)
            else:
                full_output_dir = output_dir

            os.makedirs(full_output_dir, exist_ok=True)
            output_path = os.path.join(full_output_dir, f"{file_name}.png")

            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"SHAP waterfall plot saved to: {output_path}")
        else:
            plt.show()

    def save_shap_importance_to_csv(
        self,
        file_path: str,
        method: Literal["mean_abs", "mean", "max"] = "mean_abs",
        target_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Save global SHAP feature importance to CSV.

        Parameters
        ----------
        file_path : str
            Path to save the CSV file
        method : str, optional
            Method to aggregate SHAP values
        target_name : str, optional
            Name of the target variable for identification

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature importance
        """
        importance_df = self.get_global_feature_importance(method=method)

        if target_name:
            importance_df["target"] = target_name

        importance_df["method"] = method

        # why is that rank needed?
        # just to have a rank column
        importance_df["rank"] = range(1, len(importance_df) + 1)

        (
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if os.path.dirname(file_path)
            else None
        )
        importance_df.to_csv(file_path, index=False)
        print(f"SHAP importance saved to: {file_path}")

        return importance_df


def score_calculator(feature_df: np.array, target_df: np.array) -> float:
    """Error score calculation"""
    ml_model = Histone_Regressor_shap(feature=feature_df, target=target_df)
    err_score = ml_model.error_calculator()
    return err_score


def score_calculator_with_save(
    feature_df: np.array,
    target_df: np.array,
    file_path: Optional[str] = None,  # Now optional
    feature_name: Optional[str] = None,
    target_name: Optional[str] = None,
    methods: str = "r2",
) -> Dict:
    """
    Calculate error score and save CV results to CSV.

    Parameters
    ----------
    feature_df : np.array
        Feature data array
    target_df : np.array
        Target data array
    file_path : str, optional
        Path to save CSV. If None, saves to 'output/files/ptm_prediction/cv_results.csv'
    feature_name : str, optional
        Name of the feature(s) for identification
    target_name : str, optional
        Name of the target for identification
    methods : str, optional
        Scoring method ('r2', 'mse', 'rmse'), by default "r2"

    Returns
    -------
    Dict
        Dictionary with cv_mean, cv_std, test_score
    """
    # Set default file path if not provided
    if file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        file_path = os.path.join(
            project_root, "output", "files", "ptm_prediction", "cv_results.csv"
        )

    ml_model = Histone_Regressor_shap(feature=feature_df, target=target_df)
    results = ml_model.get_cv_results(methods=methods)
    ml_model.save_cv_results_to_csv(
        file_path=file_path,
        methods=methods,
        feature_name=feature_name,
        target_name=target_name,
    )
    return results
