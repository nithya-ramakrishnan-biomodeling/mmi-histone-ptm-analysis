from multivariate_utils import (
    Histone_Regressor,
    load_data,
    json_file_saver,
    dict_sort
)
import gc
import click
import numpy as np
from pathlib import Path
np.random.seed(42)

@click.command()
@click.option("--feature-file", "-f", required=True, type=click.Path(exists=True), help="Path to the feature selection results file (JSON format).")
@click.option("--features-file", "-m", required=True, type=click.Path(exists=True), help="Path to the JSON file containing the features to use for regression.")
@click.option("--threshold", "-t", required=True, type=float, help="MMI threshold used for selecting features.")
@click.option("--output-path", "-o", required=False, help="Path to save the output files.", type=str)
@click.option("--n-jobs", "-n", required=False, type=int, default=1, help="Number of parallel jobs to run for regression.")
def main(feature_file, features_file, threshold, output_path, n_jobs):
    print("Starting regression prediction analysis")
    print(f"Feature file: {feature_file}")
    print(f"Features file: {features_file}")
    print(f"MMI threshold: {threshold}")
    print(f"Output path: {output_path}")
    print(f"Number of parallel jobs: {n_jobs}")
    df = load_data(feature_file)
    all_features = df.columns.tolist()
    print(f"Estimated optimal number of workers for parallel processing: {n_jobs}")

    # =======================================================
    # mmi threshold based target histone modification prediction
    # =======================================================

    data_df = load_data(feature_file)
    print(data_df.head())
    features_dict = load_data(features_file)[f"{threshold}"]  # get the features dict for the specified MMI threshold
    features = features_dict  # get the features list from the JSON
    targets = [target for target in all_features if target not in features]

    # saving results, using parent of input file if output path is not provided
    if output_path is None:
        output_path = Path(feature_file).parent.joinpath("mmi_results", "prediction_results.json")
    else:
        output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mmi_r2_dict = {}
    for target in targets:
        print(f"({targets.index(target)+1}/{len(targets)})")
        # use zero-copy numpy views (where possible)
        feature_arr = data_df.loc[:, features].to_numpy()
        target_arr = data_df[target].to_numpy()
        print(f"Predicting target: {target} using {features} features.")

        # instantiate regressor with constrained parallelism
        regressor = Histone_Regressor(
            feature=feature_arr,
            target=target_arr,
            model_n_jobs=n_jobs,
            cv_n_jobs=n_jobs,
        )
        file_path = output_path.parent.joinpath(
            f"mmi_regression_cv_results.csv",
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)
        regressor.save_cv_results_to_csv(
            file_path, append=True, feature_name=features, target_name=target
        )

        r2_score = regressor.get_cv_results()["test_score"]
        mmi_r2_dict[target] = r2_score

        # cleanup explicitly
        regressor.cleanup()
        del regressor, feature_arr, target_arr
        gc.collect()

    # sorting the dictionary based on  value (r2 score)
    mmi_r2_dict = dict_sort(mmi_r2_dict, by="value", reverse=True)

    # saving results, using parent of input file if output directory is not provided
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_file_saver(mmi_r2_dict, output_path)


if __name__ == "__main__":
    main()