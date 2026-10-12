import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns

import palettable
import matplotlib.ticker as ticker
from sklearn import metrics


def test_performance_fw(
    model_to_eval: torch.nn.Module,
    dataloader: torch.utils.data.dataloader.DataLoader,
    device_test: torch.device,
    perf_metrics: dict,
    plot_conf_mat: bool,
    plot_pr_curve: bool,
) -> (dict, torch.Tensor, torch.Tensor, plt.Figure, plt.Figure):
    accuracy_test = perf_metrics["acc"].to(device_test)
    specificity_test = perf_metrics["spec"].to(device_test)
    precision_test = perf_metrics["prc"].to(device_test)
    recall_test = perf_metrics["rec"].to(device_test)
    f1_test = perf_metrics["f1"].to(device_test)
    confusion_test = perf_metrics["conf"].to(device_test)
    prc_rec_crv_test = perf_metrics["prc_rec_crv"].to(device_test)

    logger.info("Starting Performance Evaluation on Testset!")

    model_to_eval.eval()
    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), unit="batch", desc="Performance Test"
    )
    with torch.inference_mode():
        for i, data in progress_bar:  # noqa
            inp_data, inp_data_static, labels = data

            inp_data: torch.Tensor = inp_data.float()
            inp_data_static: torch.Tensor = inp_data_static.float()
            labels: torch.Tensor = labels.long()  # Normal Crossentropy

            inp_data = inp_data.to(device_test)
            inp_data_static = inp_data_static.to(device_test)
            labels = labels.to(device=device_test)

            outputs_test: torch.Tensor = model_to_eval(inp_data, inp_data_static)
            _, preds = torch.max(outputs_test, 1)

            accuracy_test.update(preds, labels)
            specificity_test.update(preds, labels)
            precision_test.update(preds, labels)
            recall_test.update(preds, labels)
            f1_test.update(preds, labels)
            confusion_test.update(preds, labels)
            prc_rec_crv_test.update(preds.to(torch.double), labels.to(torch.int8))

    accuracy_on_test: torch.Tensor = accuracy_test.compute()
    specificity_on_test: torch.Tensor = specificity_test.compute()
    precision_on_test: torch.Tensor = precision_test.compute()
    recall_on_test: torch.Tensor = recall_test.compute()
    f1_on_test: torch.Tensor = f1_test.compute()
    confusion_on_test: torch.Tensor = confusion_test.compute()
    prc_rec_crv_on_test: torch.Tensor = prc_rec_crv_test.compute()

    # Todo: Mit BinaryStatsScores noch paar andere Metrics von Hand berechnen)
    results: dict = {
        "acc": accuracy_on_test,
        "spec": specificity_on_test,
        "prc": precision_on_test,
        "rec": recall_on_test,
        "f1": f1_on_test,
        "conf": confusion_on_test,
        "prc_rec_crv": prc_rec_crv_on_test,
    }
    fig_confusion, ax_confusion = confusion_test.plot()

    if plot_conf_mat:
        plt.show()
    fig_prc_rec_crv, ax_prc_rec_crv = prc_rec_crv_test.plot(score=True)
    if plot_pr_curve:
        plt.show()

    performance_report = generate_test_performance_report(torchmetrics_results=results)

    return results, labels, preds, performance_report, fig_prc_rec_crv


def generate_test_performance_report(torchmetrics_results: dict):
    """
    Take the results from above function and plot the values in a single report
    """

    def cm2inch(value):
        """converts cm to inch for plotting
        :param float value:
        """
        return value / 2.54

    def figure_setup(
        num_row=1, num_cols=1, width_cm=16, height_cm=5
    ) -> (plt.Figure, plt.Axes, plt.rc):
        # plot setup
        sns.set(style="ticks")

        fig, axes = plt.subplots(
            nrows=num_row,
            ncols=num_cols,
            dpi=300,
            figsize=(cm2inch(width_cm), cm2inch(height_cm)),
            squeeze=True,
        )

        if isinstance(axes, plt.Axes):

            axes.set_prop_cycle("color", palettable.colorbrewer.qualitative.Set3_5.mpl_colors)
            sns.despine(ax=axes)
            axes.set_axisbelow(True)

        else:
            for axs in axes.flatten():
                axs.set_prop_cycle("color", palettable.colorbrewer.qualitative.Set3_5.mpl_colors)
                sns.despine(ax=axs)
                axs.set_axisbelow(True)

        return fig, axes

    # Todo: Overview across metrics
    logger.debug("Starting Test Report Generation!")

    results = {}
    # Todo: Conversion to numpy may be better in test_performance()
    for metric in torchmetrics_results:
        if type(torchmetrics_results[metric]) is tuple:
            # we can't convert a tuple of tensors right away
            metric_array = [
                tuple_element.detach().cpu().numpy().tolist()
                for tuple_element in torchmetrics_results[metric]
            ]
            results[metric] = metric_array

        else:
            results[metric] = torchmetrics_results[metric].detach().cpu().numpy().tolist()

    scalar_labels = {
        "acc": "Accuracy",
        "spec": "Specificity",
        "prc": "Precision",
        "rec": "Recall",
        "f1": "F1",
    }

    scalar_results: dict = {key: results[key] for key in scalar_labels}

    scaler_results_df: pd.DataFrame = pd.DataFrame(data=scalar_results, index=[0])
    scaler_results_df.columns = scalar_labels.values()

    # Test Report Figure Setup
    fig_1, axs_1 = figure_setup(num_row=1, num_cols=1, width_cm=14, height_cm=14)
    fig_2, axs_2 = figure_setup(num_row=1, num_cols=1, width_cm=14, height_cm=14)
    fig_3, axs_3 = figure_setup(num_row=1, num_cols=1, width_cm=14, height_cm=14)
    color_palette = palettable.colorbrewer.qualitative.Set3_5.mpl_colors

   # fig_1.suptitle("Performance Metrics on Testset")

    axs_bars = axs_1
    # bar plots
    sns.barplot(scaler_results_df, ax=axs_bars, palette=color_palette)

    for container in axs_bars.containers:
        axs_bars.bar_label(container, fontsize=10, fmt="{:.4f}")

    axs_bars.set_ylabel("Fraction")

    # precision and recall curve
    axs_pr = axs_2

    pr_curve_x = results["prc_rec_crv"][1][::-1]
    pr_curve_y = results["prc_rec_crv"][0][::-1]
    # torchmetrics doesn't return the AUC, so we need to calculate it again
    auc: float = metrics.auc(pr_curve_x, pr_curve_y)

    # don't use sort=True, this messes up the coordinates!
    sns.lineplot(
        x=pr_curve_x, y=pr_curve_y, ax=axs_pr, sort=False, estimator=None, color="xkcd:dark teal"
    )

    axs_pr.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    props = dict(boxstyle="round", facecolor="xkcd:blue grey", alpha=0.5)

    auc_box_str: str = f"AUC: {auc:.3f}"

    # ghost line hack such that annotations gets automatically moved around on the plot depending on the curve
    # axs_pr.text(auc_box_str, fontsize=10,
    #            verticalalignment='top', bbox=props)
    auc_legend = plt.Line2D(
        [0],
        [0],
        color="xkcd:dark teal",
        marker="o",
        markerfacecolor="xkcd:dark teal",
        label=f"AUC: {auc:.3f}",
    )
    axs_pr.legend(
        handles=[auc_legend],
        loc="best",
        fancybox=True,
        facecolor="xkcd:blue grey",
        shadow=False,
        framealpha=0.5,
    )

   # axs_pr.set_title("Precision-Recall Curve")
    axs_pr.set_ylabel("Precision")
    axs_pr.set_xlabel("Recall")
    axs_pr.grid()

    # confusion matrix
    axs_conf = axs_3
    sns.heatmap(
        results["conf"],
        ax=axs_conf,
        xticklabels=["control", "sepsis in *"],
        yticklabels=["control", "sepsis in *"],
        annot=True,
        fmt="",
        annot_kws={"fontsize": 8},
        cbar=False,
    )
    axs_conf.set_xlabel("Predicted Class")
    axs_conf.set_ylabel("True Class")
    #axs_conf.set_title("Confusion Matrix")
    axs_conf.set_yticklabels(axs_conf.get_yticklabels(), rotation=90)

    plt.tight_layout()
    plt.show()

    logger.success("Done")

    return fig_1, fig_2, fig_3
