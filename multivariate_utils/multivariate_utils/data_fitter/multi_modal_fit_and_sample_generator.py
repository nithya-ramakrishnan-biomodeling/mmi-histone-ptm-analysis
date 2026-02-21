import os
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def generate_samples_orginal_sample_pdf_visualization(
    org_data, scaled_pkl_obj, gmm_pkl_obj, n_samples, bin_num=100
):

    # loading the data from the pickle object
    # pickle reader function
    data_scaled, _ = gmm_pkl_obj.sample(n_samples)
    new_samples = scaled_pkl_obj.inverse_transform(data_scaled)

    # Plot comparison
    plt.subplot(1, 2, 1)

    # plotting histogram between original data and generated samples
    plt.hist(org_data, bins=bin_num, density=True, alpha=0.5, label="Original Data")
    plt.hist(
        new_samples,
        bins=bin_num,
        density=True,
        alpha=0.5,
        label="Generated Samples",
    )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"Original Scale")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # --------------------------------------------------------------------------

    # plotting the histogram between scaled data and bayesian gmm pdf
    plt.subplot(1, 2, 2)

    # what is doing hereby x_range
    # it is creating an array of evenly spaced values between the minimum and maximum of data_scaled
    # is it bin edges?
    # I think it is just linspace for plotting the pdf curve
    x_range = np.linspace(data_scaled.min(), data_scaled.max(), bin_num).reshape(-1, 1)

    # pdf of the fitted samples
    # why are we doing this?
    # to get the probability density function values for the fitted GMM at the specified x_range points
    # The mathematical equation is p(x) = exp(score_samples(x))
    pdf = np.exp(gmm_pkl_obj.score_samples(x_range))

    # plotting the histogram of the scaled data
    plt.hist(data_scaled, bins=bin_num, density=True, alpha=0.4, label="Scaled Data")
    # Plot the PDF curve of the Bayesian GMM
    plt.plot(x_range, pdf, "r-", lw=3, label="Bayesian GMM")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # if output_path is not None:
    #     plt.savefig(f"{output_path}.png", dpi=300)
    # else:
    plt.show()


# class for the Guassian mixture model
class BayesianGaussSampler:

    def __init__(self, data):
        self.raw_data = data
        self.scaler = StandardScaler()
        self.data_scaled = self.data_scaling(data=data)
        self.raw_mean = np.mean(data)
        self.raw_std = np.std(data)
        self.scaled_mean = (self.data_scaled.mean(),)
        self.scaled_std = self.data_scaled.std()
        self.gauss_fit = None

    def data_scaling(self, data):
        """Scaling the with zero mean and unit variance for better fitting
        and to avoid converging error"""
        # Reshape and scale
        data = self.raw_data.reshape(-1, 1)
        return self.scaler.fit_transform(data)

    def data_fit(self):
        """Fitting the scaled data with Bayesian Mixture model"""

        # Bayesian GMM - automatically prunes unnecessary components
        self.bgmm = BayesianGaussianMixture(
            n_components=10,
            covariance_type="full",
            # covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1e-6,
            random_state=42,
            max_iter=10000,
            tol=1e-3,
        )

        # printing the logging info: the data is fitting
        # log_header(title="Gaussian fitting in progress...")
        # fitting the scaled data into bayesian gauss model
        self.bgmm.fit(self.data_scaled)
        self.gauss_fit = self.bgmm
        return self.gauss_fit

    def fit_savr(self, file_path: str = None):
        """ "Save fitted GMM as pickle object

        Parameters
        ----------
        file_path : str, optional
            Absolute file path to save the pickle object
        """

        if self.gauss_fit is None:
            self.gauss_fit = self.data_fit()

        return pickle_writer(self.gauss_fit, file_path)

    def data_sample(self, n_samples: int):
        """Generating samples from the multi-modal distribution"""

        if self.gauss_fit is None:
            self.gauss_fit = self.data_fit()

        self.sample_scaled, _ = self.gauss_fit.sample(n_samples=n_samples)
        self.samples_original = self.scaler.inverse_transform(self.sample_scaled)
        return self.samples_original.flatten()

    def fit_data_stats(self):

        if self.gauss_fit is None:
            self.gauss_fit = self.data_fit()

        # getting active components
        # this is the boolean values array
        active_components = self.gauss_fit.weights_ > 0.01
        weights = self.gauss_fit.weights_
        self.n_active = active_components.sum()
        print(f"active compoents are {self.n_active}")

        # Store component info
        component_info = {
            "n_active_components": int(self.n_active),
            "raw_mean": self.raw_mean,
            "raw_std": self.raw_std,
            "scaler_mean": self.scaled_mean,
            "scaler_std": self.scaled_std,
            "components": [],
        }

        for i, (weight, active) in enumerate(zip(weights, active_components)):

            print(f"Weights of the distribution{weight}, component {active}")

            if active:
                # getting stats from each active components
                mean_scaled = self.gauss_fit.means_[i, 0]
                # be careful with dimension of covariance, it depends on the Covariance type,
                # please the documentation covariance section.
                std_scaled = np.sqrt(self.gauss_fit.covariances_[i, 0, 0])

                # Transform back to original scale
                mean_original = self.scaler.inverse_transform([[mean_scaled]])[0, 0]
                std_original = std_scaled * self.scaler.scale_[0]

                print(
                    f"  Component {i+1}: weight={weight:.4f}, "
                    f"μ_scaled={mean_scaled:.2f}, σ_scaled={std_scaled:.2f}, "
                    f"μ_original={mean_original:.2f}, σ_original={std_original:.2f}"
                )

                component_info["components"].append(
                    {
                        "component_id": i + 1,
                        "weight": float(weight),
                        "mean_scaled": float(mean_scaled),
                        "std_scaled": float(std_scaled),
                        "mean_original": float(mean_original),
                        "std_original": float(std_original),
                    }
                )

        return component_info

    def visualize_fit_data(
        self,
        n_samples: int,
        bin_num=100,
    ):
        if self.gauss_fit is None:
            self.gauss_fit = self.data_fit()

        # Generate samples for visualization
        return generate_samples_orginal_sample_pdf_visualization(
            org_data=self.raw_data,
            scaled_pkl_obj=self.scaler,
            gmm_pkl_obj=self.bgmm,
            n_samples=n_samples,
            bin_num=bin_num,
        )


# sample generater from pickle object
def pickle_reader(file_path):  # This is our gmm
    with open(file_path, "rb") as file:
        return pickle.load(file)


def pickle_writer(object, file_path: str):
    with open(file_path, "wb") as file:
        pickle.dump(object, file=file, protocol=5)


def scalar_pickler(data: np.ndarray, hist_name: str, output_path: str):

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    output_file_path = os.path.join(output_path, f"{hist_name}_scaler.pkl")
    pickle_writer(scaler, output_file_path)
    del data_scaled
