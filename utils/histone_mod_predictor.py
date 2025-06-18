from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import os
import warnings
from typing import Literal

warnings.filterwarnings("ignore")

font_dict = {"fontsize": 12, "family": "serif", "weight": "bold"}


class Histone_Regressor:
    """Regrressor model for histone modification predicton"""

    def __init__(self, feature, target, cross_valid=True) -> None:
        self.feature = feature
        self.target = target
        self.cross_valid = cross_valid
        self.model = xgb.XGBRFRegressor(
            n_jobs=-1,
            n_estimators=100,
            random_state=42,
            max_depth=10,
        )

    def xgb_regressor_pred(self):

        x_train, x_test, y_train, y_test = train_test_split(
            self.feature, self.target, shuffle=True, random_state=42, test_size=0.33
        )

        if np.ndim(x_train) < 2:
            x_train = x_train.reshape(-1, 1)

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        return y_test, y_pred

    def error_calculator_norm(self, methods: Literal["r2", "mse", "rmse"] = "r2"):
        """Calculate the error score between
        predicted and true values.

        * available methods are: r2_score (r2),
        mean square_error(mse)

        Parameters
        ----------
        methods : str, optional
            _description_, by default "r2"
        """

        if methods == "mse":
            error = np.round(
                np.square(root_mean_squared_error(*self.xgb_regressor_pred())), 3
            )
            # print(f"mean squared error is: ")
            return error

        elif methods == "rmse":

            error = np.round(
                np.sqrt(root_mean_squared_error(*self.xgb_regressor_pred())), 3
            )
            # print(f"root mean squared error is: ")
            return error

        else:
            error = np.round(r2_score(*self.xgb_regressor_pred()), 3)
            # print(f" R_2 score is: ")
            return error

    def error_calculator_with_cv(self, methods: Literal["r2", "mse", "rmse"] = "r2"):

        if np.ndim(self.feature) < 2:
            feature = self.feature.reshape(-1, 1)
        else:
            feature = self.feature

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # print(methods)

        if methods == "mse":
            score_sqrt = cross_val_score(
                self.model,
                X=feature,
                y=self.target,
                scoring="neg_root_mean_squared_error",
                cv=kf,
                n_jobs=-1,
                error_score="raise",
            )

            score = np.square(score_sqrt).mean()

        elif methods == "rmse":
            score_0 = cross_val_score(
                self.model,
                X=feature,
                y=self.target,
                scoring="neg_root_mean_squared_error",
                cv=kf,
                n_jobs=-1,
                error_score="raise",
            ).mean()

            score = abs(score_0)

        else:
            score = cross_val_score(
                self.model,
                X=feature,
                y=self.target,
                scoring="r2",
                cv=kf,
                n_jobs=-1,
                error_score="raise",
            ).mean()

        return score

    def error_calculator(self, methods: Literal["r2", "mse", "rmse"] = "r2") -> float:
        """Error calculation between the origin and predicted target data.

        Parameters
        ----------
        methods : str, optional
            error_methods , by default "r2"
        """

        if self.cross_valid:

            score = self.error_calculator_with_cv(methods=methods)
        else:
            score = self.error_calculator_norm(methods=methods)

        return score

    def scatterplotter(
        self, file_name: str = None, title_value: str = None, new_dir: str = None
    ):
        """Scatter plot that showing the predcited vs true values.

        Parameters
        ----------
        file_name : str, optional
            Filename for saving, by default None
        """

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


def score_calculator(feature_df: np.array, target_df: np.array) -> float:
    """Error score calculation

    Parameters
    ----------
    feature_hits : List[str]
        feature histone mod names.
    target : List[str]
        Target histone mods names.

    Returns
    -------
    float
        Error score calculation, default r2 score
    """

    ml_model = Histone_Regressor(feature=feature_df, target=target_df)
    err_score = ml_model.error_calculator()

    return err_score
