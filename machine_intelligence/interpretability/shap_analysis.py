import shap
import pandas as pd
import torch.nn
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from machine_intelligence.utilities.viz_setup import figure_setup, five_min_to_hours
import seaborn as sns
import palettable
import numpy as np
from machine_intelligence.utilities.visualizations_interpretability import shap_heatmap_mod, shap_heatmap_values
import wandb

from machine_intelligence.models.GatedTransformer import GatedTransformer


class ModelShapWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_ts_in, x_static_in):
        self.model.eval()

        if isinstance(self.model, GatedTransformer):
            x = self.model(x_ts_in, x_static_in, "test")
        else:
            x = self.model(x_ts_in, x_static_in)

        x = torch.nn.functional.sigmoid(x)

        return x


def capture_shap_plot():
    ax = plt.gca()
    # ax.plot([0, 1], [0, 1])
    return ax


def shap_analysis(model, batch_train, batch_test, num_features, feature_names_ts, feature_names_static):
    logger.info("Starting SHAP analysis !"
                )
    explain_train_ts, explain_train_static, _ = batch_train
    explain_test_ts, explain_test_static, _ = batch_test

    # Todo: Grouping of batch in negative/positive samples and comparing them

    considered_features = [*feature_names_ts, *feature_names_static]

    number_features_explainer = num_features
    predicted_class_no_sepsis: int = 0
    predicted_class_sepsis: int = 1

    sample_int: int = 2

    num_samples_to_explain: int = 3
    x_train_explain = [explain_train_ts, explain_train_static]
    x_test_explain = [explain_test_ts[:num_samples_to_explain], explain_test_static[:num_samples_to_explain]]

    model_shap = ModelShapWrapper(model=model)
    # Todo: Determine model outputs, to group the explanations together and compare them
    explainer = shap.DeepExplainer(
        model=model_shap,
        data=x_train_explain,
    )

    explainer.feature_names = considered_features

    # check_additivity=False due to probable accumulation of rounding errors
    shap_values = explainer.shap_values(x_test_explain, check_additivity=False)
    # shap_value_new_api = explainer(x_test_explain)

    # we can build the explanation manually by splitting up the shap values we got from the old api
    # that's totally fine as well, but allows us to use the more convenient plotting of the new api

    df_explanations_ts = pd.DataFrame(data=x_test_explain[0].reshape(-1, len(feature_names_ts)),
                                      columns=feature_names_ts)

    shap_explanation_ts = shap.Explanation(shap_values[0], base_values=explainer.expected_value,
                                           data=df_explanations_ts,
                                           feature_names=feature_names_ts)

    df_sample_ts = pd.DataFrame(explain_test_ts[sample_int], columns=feature_names_ts)
    shap_explanation_ts_sample = shap.Explanation(shap_values[0][sample_int].squeeze(2),
                                                  base_values=explainer.expected_value,
                                                  feature_names=feature_names_ts, data=df_sample_ts)

    shap_explanation_static_sample = shap.Explanation(shap_values[1][sample_int].squeeze(1),
                                                      base_values=explainer.expected_value,
                                                      feature_names=feature_names_static)

    shap_explanation_static_multi = shap.Explanation(shap_values[1].squeeze(2), base_values=explainer.expected_value,
                                                     data=pd.DataFrame(data=x_test_explain[1],
                                                                       columns=feature_names_static),
                                                     feature_names=feature_names_static)

    # shap.plots.bar(shap_explanation_static_multi[0])
    # shap.plots.waterfall(shap_explanation_static_sample)
    # shap.plots.beeswarm(shap_explanation_static_multi)
    # shap.plots.bar(shap_explanation_static_multi)

    # shap.plots.violin(shap_explanation_static_multi, show=False)

    # ax_vio = capture_shap_plot()
    # fig_vio = plt.gcf()

    #  fig_vio.set_size_inches(8, 6)
    # ax_vio.set_title("SHAP Summary")
    # plt.show()

    # shap heatmap for a sample

    sample_importance = np.argsort(shap_explanation_ts_sample.abs.mean(0).values)

    sorted_feature_names_ts = [None] * len(feature_names_ts)

    for rank, idx in enumerate(sample_importance):
        sorted_feature_names_ts[rank] = feature_names_ts[idx]

    heatmap_importance_ts_features = sorted_feature_names_ts[-9:]

    df_sample_ts_heatmap = df_sample_ts.loc[:, heatmap_importance_ts_features]

    fig_heat, ax_heat, font_lb, font_anot = figure_setup(2, num_cols=1, width_cm=15, height_cm=15, sharex=True,
                                                         sharey=True)

    shap_heatmap_mod(shap_values=shap_explanation_ts_sample, ax=ax_heat[0],
                     cmap=sns.color_palette("icefire", as_cmap=True), show=False)

    sns.lineplot(df_sample_ts_heatmap, ax=ax_heat[1], palette=palettable.matplotlib.Plasma_10.mpl_colors[1:])

    ax_heat[1].sharex(ax_heat[0])

    ax_heat[0].set_xlabel("Time Step", fontproperties=font_lb)
    ax_heat[0].set_ylabel("Time Series Variables", fontproperties=font_lb)

    cbar = [ax for ax in fig_heat.axes if ax.get_label() == '<colorbar>'][0]
    # cbar_labels = ["Negative", "Positive"]
    cbar_labels = ["-", "+"]
    cbar.set_yticklabels(cbar_labels)
    cbar.set_ylabel("SHAP Feature Importance", fontproperties=font_lb)

    for tick in ax_heat[0].get_xticklabels() + ax_heat[0].get_yticklabels():
        tick.set_fontproperties(font_lb)

    ax_heat[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
        fancybox=True,
        shadow=True,
        prop=font_anot
    )

    plt.tight_layout()
    plt.show()

    # heatmap superimposed

    heat_values, heat_feature_values, heat_feature_names = shap_heatmap_values(shap_values=shap_explanation_ts_sample)
    df_heat_diy = pd.DataFrame(data=heat_values, columns=heat_feature_names)

    vmin, vmax = np.nanpercentile(heat_values.flatten(), [1, 99])
    sns.heatmap(df_heat_diy.T, cmap=sns.color_palette("icefire", as_cmap=True), vmin=min(vmin, -vmax),
                vmax=max(-vmin, vmax))

    # same row heatmap
    fig_heat_s, ax_heat_s, font_lb, font_anot = figure_setup(1, num_cols=1, width_cm=15, height_cm=15, num_colors=11,
                                                             color_offset=2, sharex=True)
    sns.heatmap(df_heat_diy.T,
                cmap=sns.color_palette("icefire", as_cmap=True),
                ax=ax_heat_s, cbar=True,
                vmin=min(vmin, -vmax),
                vmax=max(-vmin, vmax),
                )

    # Invert the y-axis to match the heatmap coordinates
    ax_heat_s.invert_yaxis()

    # Plotting time series data on the same axis with adjusted y-values
    for i, feature_name in enumerate(heat_feature_names):
        if i == len(heat_feature_names) - 1:
            break
        # Calculate the center of the heatmap row
        center_y = i
        # Plot the time series data at the center of the corresponding heatmap row
        ax_heat_s.plot(df_sample_ts_heatmap.index,
                       # ax_heat_s.scatter(df_sample_ts_heatmap.index,
                       df_sample_ts_heatmap[feature_name] / df_sample_ts_heatmap[feature_name].max() + center_y,
                       label=feature_name)

    ax_heat_s.set_xlabel("Time [h]", fontproperties=font_lb)
    ax_heat_s.set_ylabel("Time Series Features", fontproperties=font_lb)

    ax_heat_s.collections[0].colorbar.set_label("SHAP Feature Importance", fontproperties=font_lb)

    five_min_hour_formatter = ticker.FuncFormatter(func=five_min_to_hours)
    ax_heat_s.xaxis.set_major_locator(plt.MultipleLocator(1 * 12))  # 5 min  x 12
    ax_heat_s.xaxis.set_major_formatter(five_min_hour_formatter)

    for tick in ax_heat_s.get_xticklabels() + ax_heat_s.get_yticklabels():
        tick.set_fontproperties(font_anot)

    # Adding legend for line plots
    ax_heat_s.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
        fancybox=True,
        shadow=True,
        prop=font_anot
    )

    plt.tight_layout()
    plt.show()
    # global averages bar plot for time series

    # mean across batches and time steps
    shap_ts_mean = shap_explanation_ts.abs.mean((0, 1)).values.squeeze(1)
    df_shap_ts_mean = pd.DataFrame(data=shap_ts_mean, index=shap_explanation_ts.feature_names, columns=["shap_value"])
    df_shap_ts_mean_top_15 = df_shap_ts_mean.sort_values(by="shap_value", ascending=False).head(15)

    fig_ts_mean, ax_ts_mean, _, _ = figure_setup(1, 1, width_cm=14, height_cm=14)

    sns.barplot(data=df_shap_ts_mean_top_15.T, ax=ax_ts_mean, palette=palettable.matplotlib.Plasma_16.mpl_colors[1:])

    ax_ts_mean.set_xticklabels(ax_ts_mean.get_xticklabels(), rotation=60, ha="right",
                               # fontproperties=font_lb
                               )

    ax_ts_mean.set_xlabel("Time Series Features",
                          # fontproperties=font_lb
                          )
    ax_ts_mean.set_ylabel(r" mean $\| \mathrm{SHAP} \|$",
                          # fontproperties=font_lb
                          )

    plt.tight_layout()

    plt.show()

    # mean across batches for static inputs

    shap_explanation_static = shap.Explanation(shap_values[1], base_values=explainer.expected_value,
                                               data=x_test_explain[1],
                                               feature_names=feature_names_static)

    shap_static_mean = shap_explanation_static.abs.mean(0).values.squeeze(1)
    df_shap_static_mean = pd.DataFrame(data=shap_static_mean, index=shap_explanation_static.feature_names,
                                       columns=["shap_value"])
    df_shap_static_mean_top_10 = df_shap_static_mean.sort_values(by="shap_value", ascending=False).head(10)

    fig_static_mean, ax_static_mean, _, _ = figure_setup(1, 1, width_cm=14, height_cm=14)

    sns.barplot(data=df_shap_static_mean_top_10.T, ax=ax_static_mean,
                palette=palettable.matplotlib.Plasma_6.mpl_colors[1:])

    ax_static_mean.set_xticklabels(ax_static_mean.get_xticklabels(), rotation=60, ha="right",
                                   # fontproperties=font_lb
                                   )

    ax_static_mean.set_xlabel("Static Features",
                              # fontproperties=font_lb
                              )
    ax_static_mean.set_ylabel(r" mean $\| \mathrm{SHAP} \|$",
                              # fontproperties=font_lb
                              )

    plt.tight_layout()

    plt.show()

    # wandb logging
    shap_ts_mean_table = wandb.Table(dataframe=df_shap_ts_mean.T)
    shap_static_mean_table = wandb.Table(dataframe=df_shap_static_mean.T)

    wandb.log({"SHAP ts means": shap_ts_mean_table})
    wandb.log({"SHAP static means": shap_static_mean_table})

    wandb.log({"interp/shap_ts_mean": wandb.Image(fig_ts_mean)})
    wandb.log({"interp/shap_static_mean": wandb.Image(fig_static_mean)})
    wandb.log({"interp/shap_heatmap_sample": wandb.Image(fig_heat_s)})

    # df_x_test_explain = pd.DataFrame(
    #     x_test_explain[0].reshape(1, number_features_explainer), columns=considered_features
    # )

    # shap.force_plot(
    #     explainer.expected_value[0],
    #     # shap_values[0, 0, :, 0].values,
    #     df_x_test_explain,
    #     matplotlib=True,
    # )

    # df_x_test_time = pd.DataFrame(x_test_explain[0, :, :], columns=considered_features)

    # shap_values_ts = shap_values[0][0]
    # shap_values_static = shap_values[1][0]

    # shap_values_2d_no_sep = shap_values_ts[:, :, :, predicted_class_no_sepsis].reshape(
    #     -1, number_features_explainer
    # )
    # shap_values_2d_sep = shap_values_ts[:, :, :, predicted_class_sepsis].reshape(
    #     -1, number_features_explainer
    # )

    # x_test_explain_2d = x_test_explain.reshape(-1, number_features_explainer)
    # df_x_test_2d = pd.DataFrame(data=x_test_explain_2d, columns=considered_features)

    # df_x_test_sample = pd.DataFrame(data=x_test_explain[0][0], columns=feature_names_ts)
    # shap.summary_plot(shap_values_ts, df_x_test_sample)
    #
    # shap.initjs()
    #
    # shap.plots.force(base_value=explainer.expected_value[0], shap_values=shap_values_static)
    #
    # # data input via dataframes seems to make the package less finicky
    #
    # # single sample attributions
    #
    # df_sample_ts = pd.DataFrame(data=x_test_explain[0][0], columns=feature_names_ts)
    # df_sample_static = pd.DataFrame(data=[x_test_explain[1][0].numpy()], columns=feature_names_static)
    #
    # shap.plots.decision(explainer.expected_value, shap_values=shap_values_ts[:, :, 0], features=df_sample_ts,
    #                     # feature_names=feature_names_ts, link="identity", auto_size_plot=False)
    #                     feature_names=feature_names_ts, link="identity", auto_size_plot=False)
    #
    # shap.plots.decision(explainer.expected_value, shap_values=shap_values_static[:, 0], features=df_sample_static,
    #                     # feature_names=feature_names_static, link="identity", auto_size_plot=False)
    #                     feature_names=feature_names_static, link="identity", auto_size_plot=False)

    print("hallo")

    # Todo: beeswarm, mit shap values [instance, feature], reshaping auf feature über time steps oder auch batches

    #
    # # Todo: Abhängig von der Predicted Class machen, wie bei siehe highligh
    #
    # plot_shap_decision_multi = shap.multioutput_decision_plot(
    #     expected_val_list,
    #     shap_val_list,
    #     row_index=0,
    #     features=df_x_test_2d,
    #     feature_names=considered_features,
    #     legend_location="lower right",
    #     legend_labels=["no sepsis", "sepsis"],
    #     # highlight = [np.argmax(heart_predictions[row_index])],
    #     return_objects=True,
    #     show=False
    # )
    #
    # plt.show()
    #
    # fig_decision_mo = plt.figure()
    # shap.multioutput_decision_plot(
    #     expected_val_list,
    #     shap_val_list,
    #     row_index=0,
    #     features=df_x_test_2d,
    #     feature_names=considered_features,
    #     legend_location="lower right",
    #     legend_labels=["no sepsis", "sepsis"],
    #     feature_order="hclust",
    #     title="hier. clust.",
    #     return_objects=True,
    #     show=False
    # )

    # plt.savefig("mygraph.pdf", format="pdf", dpi=600, bbox_inches='tight')
    # plot_shap_decision_multi_hclust = plt.gcf()
    # plt.show()
    #
    # # wandb.log({"interp/shap_decision_no_sepsis": plot_shap_decision_no_sep})
    # wandb.log({"interp/shap_decision_sepsis": wandb.Image(fig_decision_sep)})
    #
    # # wandb.log({"interp/shap_decision_multi": plot_shap_decision_multi})
    # wandb.log({"interp/shap_decision_multi_hclust": wandb.Image(fig_decision_mo)})
    # # Todo: Bei der Heatmap muss man eventuell die features across time averagen dann könnte es gehen
