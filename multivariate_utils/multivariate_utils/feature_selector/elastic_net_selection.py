"""
COMPLETE ELASTIC NET FEATURE DISCOVERY PIPELINE
================================================
For N features: Predict each feature using all others as predictors
Find top K features for each target using:
  - Elastic Net with automatic hyperparameter tuning (ElasticNetCV)
  - Cross-validation for honest performance estimation
  - Shuffle-split (67% train, 33% test) with multiple splits
  - Feature importance extraction
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# from .elastic_config import Config


class Config:
    """Configuration for feature discovery pipeline"""

    # K features to select
    K_SELECT = 10
    # Train-test split
    TRAIN_SIZE = 0.67  # 67% training
    TEST_SIZE = 0.33  # 33% testing
    N_SPLITS = 5  # Number of shuffle-split iterations

    # Elastic Net hyperparameters to test
    L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # L1/L2 balance
    ALPHAS = np.logspace(-4, 1, 50)  # Regularization strength (log scale)

    # Cross-validation
    N_FOLDS = 10  # 10-fold cross-validation inside each train set

    # Other parameters
    RANDOM_STATE = 42
    MAX_ITER = 10000  # Max iterations for convergence
    VERBOSE = True  # Print progress


class ElastnetFeature:

    def __init__(self, target_mod: str, df: pd.DataFrame, config=Config):
        self.target_mod = target_mod
        self.df = df
        self.config = config
        self._train_test_data = None

    def _train_test_split_feature_matrix(self):
        """
        Split data into training and testing sets for a given target feature.

        Parameters:
        -----------
        data : pd.DataFrame
            Complete dataset with features.
        target_feature : str
            Name of the target feature to predict.
        config : Configuration object

        Returns:
        --------
        X_train, X_test, y_train, y_test : Split datasets
        """
        X = self.df.drop(columns=[self.target_mod])
        y = self.df[self.target_mod]

        # keep feature names aligned with X (exclude target)
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.config.TRAIN_SIZE,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
        )

        return X_train, X_test, y_train, y_test

    def elastic_net_feature_selection_single_target(self):
        """
        Train Elastic Net and select top K features using PERMUTATION IMPORTANCE.
        """

        if self._train_test_data is None:
            self._train_test_data = self._train_test_split_feature_matrix()

        X_train, X_test, y_train, y_test = self._train_test_data

        # scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ELASTIC NET WITH CV
        model = ElasticNetCV(
            l1_ratio=self.config.L1_RATIOS,
            alphas=self.config.ALPHAS,
            cv=self.config.N_FOLDS,
            random_state=self.config.RANDOM_STATE,
            max_iter=self.config.MAX_ITER,
            n_jobs=-1,
            verbose=0,
        )

        model.fit(X_train_scaled, y_train)

        # GET CV SCORES
        kfold = KFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE,
        )
        cv_scores = cross_val_score(
            ElasticNetCV(
                l1_ratio=self.config.L1_RATIOS,
                alphas=self.config.ALPHAS,
                cv=self.config.N_FOLDS,
                random_state=self.config.RANDOM_STATE,
                max_iter=self.config.MAX_ITER,
            ),
            X_train_scaled,
            y_train,
            cv=kfold,
            scoring="r2",
            n_jobs=-1,
        )

        # PREDICTIONS
        y_train_pred = model.predict(X_train_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        y_test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # =========================================================================
        # FEATURE SELECTION: USING PERMUTATION IMPORTANCE
        # =========================================================================

        importance_method = None
        perm_scores = None

        try:
            # Calculate permutation importance on TEST set
            perm_importance = permutation_importance(
                model,
                X_test_scaled,
                y_test,
                n_repeats=10,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
            )
            perm_scores = perm_importance.importances_mean
            importance_method = "Permutation Importance (TEST SET)"

            # ✅ USE PERMUTATION IMPORTANCE TO SELECT FEATURES
            top_k_indices = np.argsort(perm_scores)[-self.config.K_SELECT :][::-1]
            top_k_importance = perm_scores[top_k_indices]

            if self.config.VERBOSE:
                print(f"    ✓ Permutation importance calculated successfully")

        except Exception as e:
            # Fallback: Use coefficient magnitude
            print(f"    ⚠️  Permutation importance failed: {str(e)[:60]}...")
            print(f"        Falling back to coefficient magnitude")

            coef_abs = np.abs(model.coef_)
            perm_scores = coef_abs
            importance_method = "Coefficient Magnitude (FALLBACK)"

            # USE COEFFICIENTS TO SELECT FEATURES (as fallback)
            top_k_indices = np.argsort(perm_scores)[-self.config.K_SELECT :][::-1]
            top_k_importance = perm_scores[top_k_indices]

        # =========================================================================
        # ADDITIONAL METRICS: Compare permutation importance vs coefficients
        # =========================================================================

        coef_abs = np.abs(model.coef_)
        coef_ranking = np.argsort(coef_abs)[-self.config.K_SELECT :][::-1]
        selected_by_coef = [self.feature_names[i] for i in coef_ranking]
        agreement = list(
            set([self.feature_names[i] for i in top_k_indices]).intersection(
                set(selected_by_coef)
            )
        )
        disagreement = list(
            set([self.feature_names[i] for i in top_k_indices]).symmetric_difference(
                set(selected_by_coef)
            )
        )
        # For permutation importance, map top indices to names as well
        top_k_indices = np.argsort(perm_scores)[-self.config.K_SELECT :][::-1]
        top_k_features_permutation = [self.feature_names[i] for i in top_k_indices]
        top_k_importance = perm_scores[top_k_indices]

        # Build a mapping of feature -> coefficient if useful
        coef_map = {
            self.feature_names[i]: float(coef_abs[i])
            for i in range(len(self.feature_names))
        }
        # =========================================================================
        # BUILD RESULTS DICTIONARY
        # =========================================================================

        results = {
            # Target info
            "target_feature_name": self.target_mod,
            # # Models
            # 'model': model,
            # 'scaler': scaler,
            # Optimal hyperparameters
            "alpha_optimal": model.alpha_,
            "l1_ratio_optimal": model.l1_ratio_,
            # Cross-validation (training data)
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            # Training performance
            "train_r2": train_r2,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            # Test performance (unseen data)
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            # ===== PERMUTATION IMPORTANCE (PRIMARY METHOD) =====
            "importance_method": importance_method,
            "permutation_importance_scores": (
                perm_scores.tolist()
                if hasattr(perm_scores, "tolist")
                else list(perm_scores)
            ),
            "top_k_features_permutation": top_k_features_permutation,
            "top_k_importance_scores": top_k_importance,
            # ===== COEFFICIENTS (FOR REFERENCE/FALLBACK) =====
            "all_coefficients": coef_map,
            "coef_top_k_features": selected_by_coef,
            "features_selected_by_coefficients": selected_by_coef,
            "coef_abs": coef_abs.tolist(),
            # Comparison metrics
            "agreement_with_coefficients": agreement,
            "disagreement_with_coefficients": disagreement,
            # Feature selection summary
            "n_selected_total": int(np.sum(model.coef_ != 0)),
            "selected_features_mask": {
                self.feature_names[i]: bool(model.coef_[i] != 0)
                for i in range(len(self.feature_names))
            },
        }

        return results
