import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import StandardScaler


def select_top_n_omp_features(
    features: np.ndarray, target: np.ndarray, n_features: int
) -> np.ndarray:
    """
    Select top n features based on absolute coefficient values.
    This method fits an OMP model directly and extracts the n most important features
    based on the magnitude of their coefficients.

    Args:
        n_features: Number of top features to select

    Returns:
        Indices of the selected top features
    """
    if n_features > features.shape[1]:
        raise ValueError("n_features exceeds the number of available features.")

    if np.ndim(features) < 2:
        return features.reshape(-1, 1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Fit OMP with cross-validation
    omp = OrthogonalMatchingPursuitCV(cv=10, max_iter=min(X_scaled.shape[1], 30))
    omp.fit(X_scaled, target)

    # Get absolute coefficients
    abs_coefs = np.abs(omp.coef_)

    # Sort features by descending absolute coefficient value
    sorted_indices = np.argsort(abs_coefs)[::-1]

    # Select top n features (ensure we don't exceed available features)
    n_to_select = min(n_features, len(sorted_indices))
    top_n_features = sorted_indices[:n_to_select]

    # Store selected feature information
    omp_coefficients = omp.coef_[top_n_features]

    return top_n_features, omp_coefficients
