import pandas as pd
from joblib import delayed, Parallel
from multivariate_utils import (
    BayesianGaussSampler,
    scalar_pickler,
    ProjectPaths,
    YEAST_FILE_PATH,
    HUMAN_FILE_PATH,
    load_data,
)


def saving_scaler(data: pd.Series, hist_name: str, organism: str):
    """Function to fit and save the Standard Scaler for a given histogram data.

    Parameters
    ----------
    data : pd.Series
        Histone value series.
    hist_name : str
        Name of histone modification.
    organism : str
        Name of the organism, (e.g., 'yeast', 'human').
    output_dir : str
        Directory to save the scaler object.
    """

    output_dir = ProjectPaths.get_output_files_dir().joinpath(
        "multi_modal_fit", "scaler_pkl", organism
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_pickler(data=data.to_numpy(), hist_name=hist_name, output_path=output_dir)


def saving_model(hist_name: str, series: pd.Series, organism: str):
    """Function to fit and save the Bayesian Gaussian Mixture Model for a given histogram data.

    Parameters
    ----------
    hist_name : str
        Name of histone modification.
    series : pd.Series
        Histone value series.
    organism : str
        Name of the organism, (e.g., 'yeast', 'human').
    """

    data = series.to_numpy()
    bays_gmm = BayesianGaussSampler(data=data)
    output_dir = ProjectPaths.get_output_files_dir().joinpath(
        "multi_modal_fit", "bays_gmm_fit_pkl", organism
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hist_name.lower()}_fit.pkl"
    bays_gmm.fit_savr(file_path=output_path)


if __name__ == "__main__":

    for organism in ["yeast", "human"]:

        # organism = "yeast"  # Change to 'human' as needed
        if organism == "yeast":
            file_path = YEAST_FILE_PATH
            n_worker = 8

        elif organism == "human":
            file_path = HUMAN_FILE_PATH
            n_worker = 5

        data_multimodal = load_data(file_path)

        # ----------------------------------------------------------------------
        # Fit and save Bayesian GMM model for each histone modification
        # ----------------------------------------------------------------------
        # # Fit and save Standard Scaler for each histone modification
        # results = Parallel(n_jobs=n_worker)(
        #     delayed(saving_model)(hist, data_multimodal[hist], organism)
        #     for hist in data_multimodal.columns
        # )

        # ----------------------------------------------------------------------
        # fit and save scaler object for each histone modifications.
        results = Parallel(n_jobs=n_worker)(
            delayed(saving_scaler)(data_multimodal[hist], hist, organism)
            for hist in data_multimodal.columns
        )
