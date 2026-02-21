import numpy as np
import pandas as pd
from .data_handler import load_data


class DataPreprocessor:
    def __init__(self, data_path,
                  normalization_method='zero_mean', 
                  clip_min_value=None,
                  clip_max_value=None, 
                  impute_missing_values=True,
                  impute_strategy='mean'
                  ):
        
        self.data_path = data_path
        self.df: pd.DataFrame = load_data(self.data_path)
        self.normalization_method = normalization_method
        self.clip_min_value = clip_min_value
        self.clip_max_value = clip_max_value
        self.impute_missing_values = impute_missing_values
        self.impute_strategy = impute_strategy

    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        if self.impute_missing_values:
            if self.impute_strategy == 'mean':
                col_means = np.nanmean(data, axis=0)
                imputed_data = np.where(np.isnan(data), col_means, data)
            elif self.impute_strategy == 'median':
                col_medians = np.nanmedian(data, axis=0)
                imputed_data = np.where(np.isnan(data), col_medians, data)
            elif self.impute_strategy == 'most_frequent':
                from scipy.stats import mode
                result = mode(data, axis=0, nan_policy='omit')
                most_frequent = result.mode.flatten()
                imputed_data = np.where(np.isnan(data), most_frequent, data)
            else:
                raise ValueError(f"Unsupported imputation strategy: {self.impute_strategy}")
            return imputed_data
        else:
            mask = ~np.isnan(data).any(axis=1)
            return data[mask]

    def _zero_mean_normalization(self, data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=0)
        print(f"Mean values for zero mean normalization: {mean}")
        return data - mean
    
    def _min_max_normalization(self, data: np.ndarray) -> np.ndarray:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # avoid division by zero
        return (data - min_val) / range_val
    
    def _z_score_normalization(self, data: np.ndarray) -> np.ndarray:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # avoid division by zero
        return (data - mean) / std

    def _clip_extreme_values(self, data: np.ndarray, min_value, max_value) -> np.ndarray:
        return np.clip(data, min_value, max_value)

    def preprocess(self) -> pd.DataFrame:
        df = self.df.copy()

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.columns.difference(numeric_cols)

        data = df[numeric_cols].to_numpy(dtype=float)

        # Handle missing values
        data = self._handle_missing_values(data)

        # Normalize
        if self.normalization_method == 'zero_mean':
            data = self._zero_mean_normalization(data)
        elif self.normalization_method == 'min_max':
            data = self._min_max_normalization(data)
        elif self.normalization_method == 'z_score':
            data = self._z_score_normalization(data)

        # Clip extreme values if specified
        if self.clip_min_value is not None or self.clip_max_value is not None:
            data = self._clip_extreme_values(data, self.clip_min_value, self.clip_max_value)

        # Rebuild DataFrame
        result = pd.DataFrame(data, columns=numeric_cols)

        # print summary statistics after preprocessing
        print("Summary statistics after preprocessing:")
        print(result.describe())

        # Re-attach non-numeric columns (only rows that survived imputation)
        if len(non_numeric_cols) > 0:
            non_numeric_df = df[non_numeric_cols].iloc[:len(data)].reset_index(drop=True)
            result = pd.concat([non_numeric_df, result], axis=1)

        print("Data preprocessing completed.")
        return result