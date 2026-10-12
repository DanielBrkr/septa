import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap

from machine_intelligence.utilities.visualizations_performance import figure_setup


# from shap import shap.Explanation
# from shap.plots import colors
# from shap.utils import OpChain
# from shap.plots._utils import convert_ordering


def plot_data_vs_attributions(df_data_ts, df_sample_attribution_ts, title: str):
    # Todo: Plotting only for the top 10 features or 15
    #    fig_att_tig, axs_att = plt.subplots(2, 1)
    fig_attr_tig, axs_attr = figure_setup(num_row=2, num_cols=1, width_cm=14, height_cm=10)

    plot_sample_data = sns.scatterplot(df_data_ts, ax=axs_attr[0]).set_title(title)
    plot_sample_attr = sns.lineplot(df_sample_attribution_ts, ax=axs_attr[1])

    axs_attr[0].get_legend().remove()

    axs_attr[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()

    return fig_attr_tig


def plot_attributions_barplot(df_attributions: pd.DataFrame, title: str):
    fig_attr_sum, axs_attr_sum = figure_setup(num_row=1, num_cols=1, width_cm=14, height_cm=10)

    plot_sample_attr_sum = sns.barplot(df_attributions, ax=axs_attr_sum).set_title(title)

    axs_attr_sum.set_xticklabels(axs_attr_sum.get_xticklabels(), rotation=90)

    plt.tight_layout()

    return fig_attr_sum


def shap_heatmap_mod(shap_values, instance_order=shap.Explanation.hclust(), feature_values=shap.Explanation.abs.mean(0),
                     feature_order=None, max_display=10, cmap=shap.plots.colors.red_white_blue,
                     show=True,
                     plot_width=8, ax=None):
    """Modified version taken from the SHAP package. Create a heatmap plot of a set of SHAP values.
    """
    # sort the SHAP values matrix by rows and columns
    values = shap_values.values
    if issubclass(type(feature_values), shap.utils.OpChain):
        feature_values = feature_values.apply(shap.Explanation(values))
    if issubclass(type(feature_values), shap.Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    elif issubclass(type(feature_order), shap.utils.OpChain):
        feature_order = feature_order.apply(shap.Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise Exception(f"Unsupported feature_order: {str(feature_order)}!")
    xlabel = "Instances"
    instance_order = shap.plots._utils.convert_ordering(instance_order, shap_values)

    feature_names = np.array(shap_values.feature_names)[feature_order]
    values = shap_values.values[instance_order][:, feature_order]
    feature_values = feature_values[feature_order]

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, :max_display - 1]
        new_values[:, -1] = values[:, max_display - 1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[:max_display - 1]
        new_feature_values[-1] = feature_values[max_display - 1:].sum()
        feature_names = [
            *feature_names[:max_display - 1],
            f"Sum of {values.shape[1] - max_display + 1} other features",
        ]
        values = new_values
        feature_values = new_feature_values

    # define the plot size based on how many features we are plotting
    row_height = 0.5
    if ax is None:
        plt.gcf().set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
        ax = plt.gca()

    # plot the matrix of SHAP values as a heat map
    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / values.shape[1],
        interpolation="nearest",
        vmin=min(vmin, -vmax),
        vmax=max(-vmin, vmax),
        cmap=cmap,
    )

    # adjust the axes ticks and spines for the heat map + f(x) line chart
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines[["left", "right"]].set_visible(True)
    ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", direction="out")

    ax.set_ylim(values.shape[1] - row_height, -3)
    heatmap_yticks_pos = np.arange(values.shape[1])
    heatmap_yticks_labels = feature_names
    ax.yaxis.set_ticks(
        [-1.5, *heatmap_yticks_pos],
        [r"$f(x)$", *heatmap_yticks_labels],
        fontsize=13,
    )
    # remove the y-tick line for the f(x) label
    ax.yaxis.get_ticklines()[0].set_visible(False)

    ax.set_xlim(-0.5, values.shape[0] - 0.5)
    ax.set_xlabel(xlabel)

    # plot the f(x) line chart above the heat map
    ax.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.sum(0)
    ax.plot(
        -fx / np.abs(fx).max() - 1.5,
        color="#000000",
        linewidth=1,
    )

    # plot the bar plot on the right spine of the heat map
    bar_container = ax.barh(
        heatmap_yticks_pos,
        (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20,
        height=0.7,
        align="center",
        color="#000000",
        left=values.shape[0] * 1.0 - 0.5,
        # color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    )
    for b in bar_container:
        b.set_clip_on(False)

    # draw the color bar
    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([min(vmin, -vmax), max(-vmin, vmax)])
    cb = plt.colorbar(
        m,
        ticks=[min(vmin, -vmax), max(-vmin, vmax)],
        ax=ax,
        aspect=80,
        fraction=0.01,
        pad=0.10,  # padding between the cb and the main axes
    )
    # cb.set_label(labels["VALUE"], size=12, labelpad=-10)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    # bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    # cb.ax.set_aspect((bbox.height - 0.9) * 15)
    # cb.draw_all()

    if show:
        plt.show()

    return ax


def shap_heatmap_values(shap_values, instance_order=shap.Explanation.hclust(),
                        feature_values=shap.Explanation.abs.mean(0),
                        feature_order=None, max_display=10,
                        ):
    """Modified version taken from the SHAP package. Create a heatmap plot of a set of SHAP values.
    """
    # sort the SHAP values matrix by rows and columns
    values = shap_values.values
    if issubclass(type(feature_values), shap.utils.OpChain):
        feature_values = feature_values.apply(shap.Explanation(values))
    if issubclass(type(feature_values), shap.Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    elif issubclass(type(feature_order), shap.utils.OpChain):
        feature_order = feature_order.apply(shap.Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise Exception(f"Unsupported feature_order: {str(feature_order)}!")
    xlabel = "Instances"
    instance_order = shap.plots._utils.convert_ordering(instance_order, shap_values)

    feature_names = np.array(shap_values.feature_names)[feature_order]
    values = shap_values.values[instance_order][:, feature_order]
    feature_values = feature_values[feature_order]

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, :max_display - 1]
        new_values[:, -1] = values[:, max_display - 1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[:max_display - 1]
        new_feature_values[-1] = feature_values[max_display - 1:].sum()
        feature_names = [
            *feature_names[:max_display - 1],
            f"Sum of {values.shape[1] - max_display + 1} other features",
        ]
        values = new_values
        feature_values = new_feature_values

    return values, feature_values, feature_names
