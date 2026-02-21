from multivariate_utils import (
    ProjectPaths, DataPreprocessor
)
from pathlib import Path
import click


@click.command()
@click.option("--file-path", "-f", required=True, type=click.Path(exists=False), help="Path to the input data file (CSV format).")
@click.option("--normalization-method", "-n", required=True, type=click.Choice(['zero_mean', 'min_max', 'z_score']), 
              default='zero_mean', help="Normalization method to apply to the data.")
@click.option("--clip-min-value", "-cmin", required=False, type=float, help="Minimum value for clipping extreme values.")
@click.option("--clip-max-value", "-cmax", required=False, type=float, help="Maximum value for clipping extreme values.")
@click.option("--impute-missing-values", "-impute", is_flag=True, help="Whether to impute missing values in the data.")
@click.option("--impute-strategy", "-impute-strategy", required=False, type=click.Choice(['mean', 'median', 'most_frequent']),
               default='mean', help="Strategy to use for imputing missing values.")
@click.option("--output-path", "-o", required=False, type=click.Path(), help="Path to save the preprocessed data (CSV format).")
def main(file_path, normalization_method, clip_min_value, clip_max_value, impute_missing_values, impute_strategy, output_path):
    print("Starting data preprocessing")
    print(f"File path: {file_path}")
    print(f"Normalization method: {normalization_method}")
    print(f"Clip min value: {clip_min_value}")
    print(f"Clip max value: {clip_max_value}")
    print(f"Impute missing values: {impute_missing_values}")
    print(f"Imputation strategy: {impute_strategy}")

    # getting full from absolute path to ensure it works regardless of where the script is run from
    file_path = str(Path(file_path).resolve())
    if not Path(file_path).is_file():
        print(f"Error: File not found at {file_path}")
        return
    
    preprocessor = DataPreprocessor(
        data_path=file_path,
        normalization_method=normalization_method,
        clip_min_value=clip_min_value,
        clip_max_value=clip_max_value,
        impute_missing_values=impute_missing_values,
        impute_strategy=impute_strategy
    )
    
    processed_data = preprocessor.preprocess()
    print(processed_data.head())

    if output_path is not None:
        output_path = Path(output_path).resolve()

    else: 
        output_path=ProjectPaths.get_output_files_dir("preprocessed_data.csv")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    main()