import gc
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import warnings
from typing import Literal, Optional, Dict

warnings.filterwarnings("ignore")

font_dict = {"fontsize": 12, "family": "serif", "weight": "bold"}


class Histone_Regressor:
    """Regressor model for histone modification prediction"""

    def __init__(
        self,
        feature,
        target,
        cross_valid=True,
        test_size=0.33,
        random_state=42,
        model_n_jobs: int = 1,
        cv_n_jobs: int = 1,
    ) -> None:
        self.feature = feature
        self.target = target
        self.cross_valid = cross_valid
        self.test_size = test_size
        self.random_state = random_state
        self.model_n_jobs = model_n_jobs
        self.cv_n_jobs = cv_n_jobs

        self.model = xgb.XGBRFRegressor(
            n_jobs=self.model_n_jobs,
            n_estimators=100,
            random_state=random_state,
            max_depth=10,
        )

        # Store CV results
        self.cv_scores_ = None
        self.cv_mean_ = None
        self.cv_std_ = None
        self.test_score_ = None

        # Create train/test split upfront
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
            n_jobs=self.cv_n_jobs,
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
            return results["cv_mean"]
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

    def scatterplotter(
        self, file_name: str = None, title_value: str = None, new_dir: str = None
    ):
        """Scatter plot that showing the predicted vs true values."""
        fig, ax = plt.subplots(figsize=(14, 9))
        ax.set_facecolor("#EBEBEB")
        fig.set_facecolor("#F2F2F2")

        predicted_output = self.xgb_regressor_pred()
        ax.scatter(*predicted_output, c="#3F7C85")
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="#A60A33")

        ticks_values = np.arange(
            min(predicted_output[0]), max(predicted_output[0]), dtype=int
        )
        ax.set_xticks(ticks_values)
        ax.set_yticks(ticks_values)

        labels = [f"{i}" for i in ticks_values]

        # Use test score if CV was run, otherwise calculate
        if self.test_score_ is not None:
            error = f"{self.test_score_:.2f}"
        else:
            error = f"{self.error_calculator():.2f}"

        if title_value:
            feature = title_value[0]
            target = title_value[1]
            ax.set_title(
                f" Feature: {feature}, Target: {target}, r2: {error}",
                fontdict=font_dict,
            )
        else:
            ax.set_title(f" r2: {error}", fontdict=font_dict)

        ax.set_xlabel(" True Y_values ", fontdict=font_dict)
        ax.set_ylabel(" Predicted Y_values ", fontdict=font_dict)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticklabels(labels, fontdict=font_dict)
        ax.set_yticklabels(labels, fontdict=font_dict)

        if file_name:
            ouput_dir_rel_path = os.path.join("..", "..", "output", "images")
            output_dir = os.path.abspath(ouput_dir_rel_path)

            if new_dir:
                os.makedirs(os.path.join(output_dir, new_dir), exist_ok=True)
                output_path = os.path.join(output_dir, new_dir, f"{file_name}.jpeg")
            else:
                output_path = os.path.join(output_dir, f"{file_name}.jpeg")

            fig.savefig(output_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def cleanup(self):
        try:
            del self.model
        except Exception:
            pass
        gc.collect()


def score_calculator(feature_df: np.array, target_df: np.array) -> float:
    """Error score calculation"""
    ml_model = Histone_Regressor(feature=feature_df, target=target_df)
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

    ml_model = Histone_Regressor(feature=feature_df, target=target_df)
    results = ml_model.get_cv_results(methods=methods)
    ml_model.save_cv_results_to_csv(
        file_path=file_path,
        methods=methods,
        feature_name=feature_name,
        target_name=target_name,
    )
    return results
