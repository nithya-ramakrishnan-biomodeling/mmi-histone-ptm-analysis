from multivariate_utils import (
    Histone_Regressor_shap,
    YEAST_FILE_PATH,
    load_data,
    ProjectPaths,
)


if __name__ == "__main__":

    df = load_data(YEAST_FILE_PATH)
    histone_mods = df.columns.tolist()

    for target_mod in histone_mods:

        feature_mods = [mod for mod in histone_mods if mod != target_mod]
        feature_df = df[feature_mods].to_numpy()
        target_df = df[target_mod].to_numpy()

        regressor = Histone_Regressor_shap(feature=feature_df, target=target_df)
        results = regressor.get_cv_results()

        # shap analysis and plotting
        print(regressor.get_global_feature_importance())
        regressor.plot_shap_summary(
            plot_type="bar",
            output_dir=ProjectPaths().get_project_root(),
            file_name=f"shap_summary_bar_{target_mod}.png",
        )
        regressor.plot_shap_summary(
            plot_type="beeswarm",
            output_dir=ProjectPaths().get_project_root(),
            file_name=f"shap_summary_beeswarm_{target_mod}.png",
        )

        print(results)
        break
