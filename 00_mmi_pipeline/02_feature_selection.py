from multivariate_utils import (
    load_data,
    json_file_saver,
    mmi_feature_selector,
)
from pathlib import Path
import click
from multivariate_utils.datahandler.project_paths import ProjectPaths


# arguments for the script:
# mmi_value_path: file path for the MMI values calculated in the previous step
# threshold: threshold for feature selection, can be a single value or a list of values for multiple thresholds
# output_dir: optional, directory to save the output JSON file, if not provided, it will be saved in the same directory as the input MMI values file
@click.command()
@click.option("--mmi-value-path", "-m", required=True, type=click.Path(exists=True), help="Path to the MMI values file (JSON format).")
@click.option("--threshold", "-t", required=True, help="Threshold(s) for feature selection", type=float)
@click.option("--output-path", "-o", default=None, help="Directory to save the output JSON file.")
def main(mmi_value_path, threshold, output_path):

    click.echo(f"Starting feature selection with MMI values from: {mmi_value_path}")
    click.echo(f"Threshold for feature selection: {threshold}")

    if output_path is None:
        output_path = ProjectPaths.get_output_files_dir("mmi_results", "selected_features")
    else:
        output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mmi_value_dict = load_data(mmi_value_path)
    # for positive selection, use mode="ge" and threshold as positive value
    # for negative selection, use mode="le" and threshold as negative value
    # not equal to  is not supported currently, but can be implemented by combining "ge" and "le" modes

    if threshold < 0:
        mode = "le"
    else:
        mode = "ge"
    features, _ = mmi_feature_selector(
        threshold=threshold, mmi_data=mmi_value_dict, mode=mode
    )
    print(f"{threshold}: {set(features)}")

    threshold_feature_dict = {threshold: list(features)}
    # saving to json
    json_file_saver(threshold_feature_dict, output_path)


if __name__ == "__main__":
    main()